import { randomUUID } from "node:crypto";
import type { GatewayProviderConfig } from "@ccr/core/contracts/app";
import type { LLMProvider, UnifiedChatRequest } from "@ccr/core/types/llm";
import type { Transformer, TransformerContext } from "@ccr/core/types/transformer";
import { UpstreamRequestError } from "@ccr/core/gateway/internal/shared";
import Transformers from "@ccr/core/transformer";
import { sendUnifiedRequest } from "@ccr/core/utils/request";

export type TransformerAttemptInput = {
  body?: Buffer;
  headers: Record<string, string>;
  method: string;
  path: string;
  provider: GatewayProviderConfig;
  signal?: AbortSignal;
};

export type TransformerAttemptResult = {
  response: Response;
};

const bridgeLogger = {
  debug: (...args: unknown[]) => console.debug("[transformer-bridge]", ...args),
  error: (...args: unknown[]) => console.error("[transformer-bridge]", ...args),
  info: (...args: unknown[]) => console.log("[transformer-bridge]", ...args),
  warn: (...args: unknown[]) => console.warn("[transformer-bridge]", ...args)
};

let builtinRegistry: Map<string, unknown> | undefined;

function builtinTransformerClasses(): Map<string, unknown> {
  if (builtinRegistry) {
    return builtinRegistry;
  }
  const registry = new Map<string, unknown>();
  for (const candidate of Object.values(Transformers)) {
    const staticName = (candidate as { TransformerName?: string }).TransformerName;
    if (staticName) {
      registry.set(staticName, candidate);
      continue;
    }
    try {
      const instance = new (candidate as new () => Transformer)();
      if (instance?.name) {
        registry.set(instance.name, candidate);
      }
    } catch {
      // Skip builtin transformers that fail to construct during name discovery.
    }
  }
  builtinRegistry = registry;
  return registry;
}

const instanceCache = new Map<string, Transformer>();

function transformerInstanceForName(name: string, options?: any): Transformer | undefined {
  const cacheKey = `${name}:${options ? JSON.stringify(options) : ""}`;
  const cached = instanceCache.get(cacheKey);
  if (cached) {
    return cached;
  }
  const transformerClass = builtinTransformerClasses().get(name);
  if (!transformerClass) {
    return undefined;
  }
  try {
    const instance = new (transformerClass as new (options?: any) => Transformer)(options);
    if (instance && typeof instance === "object") {
      (instance as any).logger = bridgeLogger;
    }
    instanceCache.set(cacheKey, instance);
    return instance;
  } catch {
    return undefined;
  }
}

function endpointTransformerForPath(path: string): Transformer | undefined {
  for (const [name] of builtinTransformerClasses()) {
    try {
      const instance = transformerInstanceForName(name);
      if (instance?.endPoint && pathMatchesEndpoint(path, instance.endPoint)) {
        return instance;
      }
    } catch {
      // Ignore transformers whose endpoint cannot be resolved.
    }
  }
  return undefined;
}

function pathMatchesEndpoint(path: string, endPoint: string): boolean {
  return path === endPoint || path.endsWith(endPoint);
}

type TransformerChain = {
  models: Record<string, Transformer[]>;
  use: Transformer[];
};

function isTransformerEntry(value: unknown): value is string | [string, any] {
  return typeof value === "string" || (Array.isArray(value) && typeof value[0] === "string");
}

function resolveChainUse(value: unknown): Transformer[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value
    .filter(isTransformerEntry)
    .map((entry) => {
      const name = typeof entry === "string" ? entry : entry[0];
      const options = typeof entry === "string" ? undefined : entry[1];
      return transformerInstanceForName(name, options);
    })
    .filter((transformer): transformer is Transformer => Boolean(transformer));
}

function providerTransformerChain(provider: GatewayProviderConfig): TransformerChain | undefined {
  const raw = (provider as { transformer?: unknown }).transformer;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return undefined;
  }
  const use = resolveChainUse((raw as any).use);
  if (use.length === 0) {
    return undefined;
  }
  const models: Record<string, Transformer[]> = {};
  for (const [key, value] of Object.entries(raw)) {
    if (key === "use") {
      continue;
    }
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const modelChain = resolveChainUse((value as any).use);
      if (modelChain.length > 0) {
        models[key] = modelChain;
      }
    }
  }
  return { models, use };
}

function toLLMProvider(provider: GatewayProviderConfig, chain: TransformerChain): LLMProvider {
  const transformer: Record<string, unknown> = {
    use: chain.use
  };
  for (const [model, use] of Object.entries(chain.models)) {
    transformer[model] = { use };
  }
  return {
    apiKey: provider.apiKey ?? provider.api_key ?? "",
    baseUrl: provider.baseUrl ?? provider.api_base_url ?? "",
    models: provider.models ?? [],
    name: provider.name,
    transformer: transformer as unknown as LLMProvider["transformer"]
  };
}

function isResponseLike(value: any): value is Response {
  return Boolean(
    value &&
    typeof value === "object" &&
    typeof value.text === "function" &&
    typeof value.json === "function" &&
    typeof value.body !== "undefined"
  );
}

/**
 * Executes the custom provider transformer chain (v2 pipeline port) when the
 * routed provider declares one via `provider.transformer`. Returns the final
 * response converted back to the client protocol, or undefined when the
 * provider has no transformer chain (caller should use the normal upstream
 * flow).
 */
export async function executeTransformerAttempt(input: TransformerAttemptInput): Promise<TransformerAttemptResult | undefined> {
  const chain = providerTransformerChain(input.provider);
  if (!chain) {
    return undefined;
  }
  let parsedBody: any;
  try {
    parsedBody = input.body ? JSON.parse(input.body.toString("utf8")) : undefined;
  } catch {
    return undefined;
  }
  if (!parsedBody || typeof parsedBody !== "object") {
    return undefined;
  }

  const llmProvider = toLLMProvider(input.provider, chain);
  const context: TransformerContext = {
    req: {
      body: parsedBody,
      headers: input.headers,
      id: randomUUID(),
      provider: input.provider.name
    }
  };

  const isStreaming = parsedBody.stream === true;
  const endpointTransformer = endpointTransformerForPath(input.path);
  const endpointName = endpointTransformer?.name;
  const modelChain = chain.models[parsedBody.model];
  const bypass = !isStreaming &&
    Boolean(endpointName) &&
    chain.use.length === 1 &&
    chain.use[0].name === endpointName &&
    (!modelChain?.length || (modelChain.length === 1 && modelChain[0].name === endpointName));

  let requestBody: UnifiedChatRequest = parsedBody;
  let config: any = {
    signal: input.signal,
    api_key_go: input.provider.api_key_go,
    api_key_zen: input.provider.api_key_zen,
  };
  let senderTransformer: Transformer | undefined;

  if (bypass) {
    const headers = { ...input.headers };
    delete headers["content-length"];
    config.headers = headers;
  }

  // Endpoint transformer first (request normalization for the client protocol).
  if (!bypass && endpointTransformer && typeof (endpointTransformer as any).transformRequestOut === "function") {
    const transformOut = await (endpointTransformer as any).transformRequestOut(requestBody, context);
    if (transformOut?.body) {
      requestBody = transformOut.body;
      config = { ...config, ...transformOut.config };
    } else {
      requestBody = transformOut;
    }
  }

  if (!bypass && endpointTransformer && typeof (endpointTransformer as any).sendRequest === "function") {
    senderTransformer = endpointTransformer as Transformer;
  }

  // Provider-level transformers.
  for (const providerTransformer of chain.use) {
    if (!providerTransformer || typeof providerTransformer.transformRequestIn !== "function") {
      continue;
    }
    const transformIn = await providerTransformer.transformRequestIn(requestBody, llmProvider, context);
    if (transformIn?.body) {
      requestBody = transformIn.body;
      config = { ...config, ...transformIn.config };
    } else {
      requestBody = transformIn as unknown as UnifiedChatRequest;
    }
    if (typeof (providerTransformer as any).sendRequest === "function") {
      senderTransformer = providerTransformer as Transformer;
    }
  }

  // Model-specific transformers.
  if (!bypass && modelChain?.length) {
    for (const modelTransformer of modelChain) {
      if (!modelTransformer || typeof modelTransformer.transformRequestIn !== "function") {
        continue;
      }
      const transformIn = await modelTransformer.transformRequestIn(requestBody, llmProvider, context);
      if (transformIn?.body) {
        requestBody = transformIn.body;
        config = { ...config, ...transformIn.config };
      } else {
        requestBody = transformIn as unknown as UnifiedChatRequest;
      }
      if (typeof (modelTransformer as any).sendRequest === "function") {
        senderTransformer = modelTransformer as Transformer;
      }
    }
  }

  // Passthrough auth for endpoint-aligned providers.
  if (bypass && endpointTransformer && typeof (endpointTransformer as any).auth === "function") {
    const auth = await (endpointTransformer as any).auth(requestBody, llmProvider, context);
    if (auth?.body) {
      requestBody = auth.body;
      let headers = config.headers || {};
      if (auth.config?.headers) {
        headers = { ...headers, ...auth.config.headers };
        delete headers.host;
        delete auth.config.headers;
      }
      config = { ...config, ...auth.config, headers };
    } else {
      requestBody = auth;
    }
  }

  const url = config.url || (llmProvider.baseUrl ? new URL(llmProvider.baseUrl) : undefined);
  if (!url) {
    throw new UpstreamRequestError(
      `Transformer provider "${input.provider.name}" did not produce an upstream URL. ` +
      "Configure api_base_url or ensure the transformer returns a config.url.",
      { failedAttempts: [] }
    );
  }

  const configHeaders = config?.headers || {};
  const requestHeaders: Record<string, string> = {};
  if (!configHeaders["Authorization"] && !configHeaders["authorization"] && llmProvider.apiKey) {
    requestHeaders["Authorization"] = `Bearer ${llmProvider.apiKey}`;
  }
  for (const [key, value] of Object.entries(configHeaders)) {
    if (value && value !== "undefined") {
      requestHeaders[key] = value as string;
    }
  }
  for (const key in requestHeaders) {
    if (requestHeaders[key] === "undefined") {
      delete requestHeaders[key];
    }
  }
  if (config?.headers?.["x-codex-internal"] === "true") {
    delete requestHeaders["Authorization"];
  }
  const requestConfig = {
    ...config,
    headers: JSON.parse(JSON.stringify(requestHeaders))
  };

  const sender = senderTransformer && typeof (senderTransformer as any).sendRequest === "function"
    ? senderTransformer
    : undefined;
  let response: Response;
  try {
    response = sender
      ? await (sender as any).sendRequest(requestBody, requestConfig, llmProvider, context)
      : await sendUnifiedRequest(url, requestBody, requestConfig, context, bridgeLogger);
  } catch (error) {
    throw new UpstreamRequestError(
      `Transformer request to "${input.provider.name}" failed: ${error instanceof Error ? error.message : String(error)}`,
      { cause: error, failedAttempts: [] }
    );
  }

  if (!isResponseLike(response)) {
    throw new UpstreamRequestError(
      `Invalid provider response object for ${input.provider.name}`,
      { failedAttempts: [] }
    );
  }

  if (!response.ok) {
    const errorText = await response.text();
    throw new UpstreamRequestError(
      `Error from provider(${input.provider.name},${requestBody?.model}: ${response.status}): ${errorText}`,
      { failedAttempts: [] }
    );
  }

  let finalResponse: Response = response;

  // Provider-level response transformers (reverse order).
  if (!bypass && chain.use.length) {
    for (const providerTransformer of [...chain.use].reverse()) {
      if (!providerTransformer || typeof (providerTransformer as any).transformResponseOut !== "function") {
        continue;
      }
      finalResponse = await (providerTransformer as any).transformResponseOut(finalResponse, context);
    }
  }

  // Model-specific response transformers (reverse order).
  const responseModelChain = chain.models[requestBody?.model ?? parsedBody.model];
  if (!bypass && responseModelChain?.length) {
    for (const modelTransformer of [...responseModelChain].reverse()) {
      if (!modelTransformer || typeof (modelTransformer as any).transformResponseOut !== "function") {
        continue;
      }
      finalResponse = await (modelTransformer as any).transformResponseOut(finalResponse, context);
    }
  }

  // Endpoint response conversion, skipped when a chain transformer handled sending.
  if (!bypass && !senderTransformer && endpointTransformer?.transformResponseIn) {
    finalResponse = await endpointTransformer.transformResponseIn(finalResponse, context);
  }

  return { response: finalResponse };
}
