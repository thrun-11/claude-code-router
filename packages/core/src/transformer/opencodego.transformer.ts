import { UnifiedChatRequest, LLMProvider } from "@ccr/core/types/llm";
import { Transformer, TransformerContext } from "@ccr/core/types/transformer";
import { ResponsesToAnthropicStream } from "@ccr/core/transformer/codex.transformer";
import {
  buildAnthropicMessage,
  contentBlockStopEvent,
  inputJsonDeltaEvent,
  messageDeltaEvent,
  messageStartEvent,
  messageStopEvent,
  responsesOutputToContent,
  signatureDeltaEvent,
  sseEvent,
  textBlockStartEvent,
  textDeltaEvent,
  thinkingBlockStartEvent,
  thinkingDeltaEvent,
  toolUseBlockStartEvent,
} from "@ccr/core/transformer/anthropic-emitter";
import { ProxyAgent } from "undici";
import {
  isOpencodeChatCompletionsModel,
  isOpencodeFreeModel,
} from "@ccr/core/transformer/model-capabilities";
import { isFunctionTool, mapChatTools, mapResponsesTools } from "@ccr/core/transformer/server-tools";
import {
  ZEN_FINGERPRINT_CHAT_TOOLS,
  ZEN_FINGERPRINT_RESPONSES_TOOLS,
} from "@ccr/core/transformer/opencode-fingerprint-tools";
import {
  anthropicMessageToSSE,
  createExecutedSearch,
  executeWebSearchChain,
  fallbackFunctionTool,
  fallbackQueryOf,
  FALLBACK_FN_NAME,
  readWebSearchFallbackConfig,
  searchToolResultPayload,
  synthesizeSearchBlocks,
  webSearchFallbackBudget,
  type ExecutedSearch,
} from "@ccr/core/transformer/websearch-fallback";

const CHAT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions";
// Free tier moved to /inference/v1 (MITM-verified 2026-10-13, CLI 1.18.35):
// chat frees 200 on /inference/v1/chat/completions, Responses frees
// (muse-spark-*) 200 on /inference/v1/responses with native tools.
// /zen/v1/* now answers 403 FreeTierError to the same bytes.
const FREE_CHAT_ENDPOINT = "https://opencode.ai/inference/v1/chat/completions";
const GO_RESPONSES_ENDPOINT = "https://opencode.ai/zen/go/v1/responses";
const ZEN_RESPONSES_ENDPOINT = "https://opencode.ai/inference/v1/responses";

// Harness parity with sst/opencode `packages/opencode/src/session/llm/request.ts:18,177-204`.
// MITM-captured 2026-10-13 (CLI 1.18.35, /tmp/cap5.har + /tmp/fresh.mitm):
//   chat:      User-Agent: opencode/1.18.35 ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14
//   responses: User-Agent: opencode/1.18.35 ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14
// plus `x-opencode-session` AND `x-opencode-session-id` (same ses_… value,
// stable per conversation), fresh `x-opencode-request` (msg_… per HTTP call),
// `x-opencode-client: cli`, `x-opencode-project` (repo hash or `global`),
// `prompt_cache_key` = ses id, native tools + `tool_choice: auto`,
// `store: false`, `stream: true` — otherwise 403 FreeTierError.
//
// CRITICAL: ses_/msg_ IDs must use the exact SortableID generator from
// `packages/schema/src/identifier.ts` (sequential per-ms counter shared
// across ses+msg). Random counters 403 (verified live: exact replay with
// captured IDs 200s, fresh random-counter IDs 403s).
//
// The `opencode/<ver>` part auto-refreshes from the npm registry and a
// FreeTierError triggers a refresh + retry. Override entirely via
// `OPENCODE_HARNESS_UA` env or `opencode_user_agent` provider config.
const OPENCODE_HARNESS_VERSION_DEFAULT = "1.18.35";
const OPENCODE_HARNESS_CHAT_SUFFIX_DEFAULT = "ai-sdk/provider-utils/4.0.23 runtime/bun/1.3.14";
const OPENCODE_HARNESS_RESPONSES_SUFFIX_DEFAULT = "ai-sdk/provider-utils/4.0.40 runtime/bun/1.3.14";
const OPENCODE_HARNESS_UA_REFRESH_MS = 24 * 60 * 60 * 1000;

let cachedHarnessVersion = process.env["OPENCODE_HARNESS_VERSION"] || OPENCODE_HARNESS_VERSION_DEFAULT;
let cachedHarnessChatSuffix =
  process.env["OPENCODE_HARNESS_UA_SUFFIX"] || OPENCODE_HARNESS_CHAT_SUFFIX_DEFAULT;
let cachedHarnessResponsesSuffix =
  process.env["OPENCODE_HARNESS_UA_SUFFIX"] || OPENCODE_HARNESS_RESPONSES_SUFFIX_DEFAULT;
let lastHarnessUaRefresh = 0;
let harnessUaRefreshInFlight: Promise<void> | undefined;

function buildHarnessUA(kind: "chat" | "responses" = "responses"): string {
  const suffix = kind === "chat" ? cachedHarnessChatSuffix : cachedHarnessResponsesSuffix;
  return `opencode/${cachedHarnessVersion} ${suffix}`;
}

export function getOpencodeUserAgent(config?: any, kind?: "chat" | "responses"): string {
  const override = config?.opencode_user_agent || process.env["OPENCODE_HARNESS_UA"];
  if (typeof override === "string" && override.trim()) return override.trim();
  const inferred = typeof config?.harness_ua_kind === "string" ? config.harness_ua_kind : kind;
  return buildHarnessUA(inferred === "chat" ? "chat" : "responses");
}

function compareSemver(a: string, b: string): number {
  const pa = a.split(".").map((x) => parseInt(x, 10) || 0);
  const pb = b.split(".").map((x) => parseInt(x, 10) || 0);
  for (let i = 0; i < 3; i++) {
    if (pa[i] !== pb[i]) return pa[i] < pb[i] ? -1 : 1;
  }
  return 0;
}

async function fetchLatestOpencodeVersion(): Promise<string | undefined> {
  try {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 10000);
    try {
      const res = await fetch("https://registry.npmjs.org/opencode-ai/latest", {
        signal: ctrl.signal,
        headers: { Accept: "application/json" },
      });
      if (!res.ok) return undefined;
      const data = (await res.json().catch(() => undefined)) as any;
      const v = typeof data?.version === "string" ? data.version.trim() : "";
      return /^\d+\.\d+\.\d+(-[\w.]+)?$/.test(v) ? v : undefined;
    } finally {
      clearTimeout(timer);
    }
  } catch {
    return undefined;
  }
}

export async function refreshOpencodeUserAgent(reason: string, logger?: any): Promise<string> {
  if (!harnessUaRefreshInFlight) {
    harnessUaRefreshInFlight = (async () => {
      const latest = await fetchLatestOpencodeVersion();
      lastHarnessUaRefresh = Date.now();
      if (latest && compareSemver(cachedHarnessVersion, latest) < 0) {
        const prev = cachedHarnessVersion;
        cachedHarnessVersion = latest;
        logger?.info?.(`[opencode-go] harness UA refreshed (${reason}): ${prev} -> ${latest}`);
      }
    })().finally(() => {
      harnessUaRefreshInFlight = undefined;
    });
  }
  await harnessUaRefreshInFlight;
  return buildHarnessUA();
}

export function maybeRefreshOpencodeUserAgent(logger?: any): void {
  if (Date.now() - lastHarnessUaRefresh < OPENCODE_HARNESS_UA_REFRESH_MS) return;
  void refreshOpencodeUserAgent("periodic", logger).catch(() => {});
}

export function resetOpencodeUserAgentForTests(): void {
  cachedHarnessVersion = OPENCODE_HARNESS_VERSION_DEFAULT;
  cachedHarnessChatSuffix = OPENCODE_HARNESS_CHAT_SUFFIX_DEFAULT;
  cachedHarnessResponsesSuffix = OPENCODE_HARNESS_RESPONSES_SUFFIX_DEFAULT;
  lastHarnessUaRefresh = 0;
  harnessUaRefreshInFlight = undefined;
  resetOpencodeIdCounterForTests();
}

// Exact SortableID generator from `packages/schema/src/identifier.ts`
// (~/opencode checkout). The counter is SEQUENTIAL per ms and SHARED across
// ses_/msg_ calls — random counters 403 (verified live 2026-10-13: exact
// replay with captured IDs 200s, fresh random-counter IDs 403s, exact
// sequential-counter IDs 200s).
const OPENCODE_ID_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const OPENCODE_ID_LENGTH = 26;

let opencodeIdLastTimestamp = 0;
let opencodeIdCounter = 0;

export function resetOpencodeIdCounterForTests(): void {
  opencodeIdLastTimestamp = 0;
  opencodeIdCounter = 0;
}

function createOpencodeId(descending: boolean, timestamp = Date.now()): string {
  if (timestamp !== opencodeIdLastTimestamp) {
    opencodeIdLastTimestamp = timestamp;
    opencodeIdCounter = 0;
  }
  opencodeIdCounter++;
  const current = BigInt(timestamp) * 0x1000n + BigInt(opencodeIdCounter);
  const value = descending ? ~current : current;
  const time = Array.from({ length: 6 }, (_, index) =>
    Number((value >> BigInt(40 - 8 * index)) & 0xffn)
      .toString(16)
      .padStart(2, "0"),
  ).join("");
  const cryptoObj = (globalThis as any)?.crypto;
  let suffix: string;
  if (cryptoObj?.getRandomValues) {
    const bytes = cryptoObj.getRandomValues(new Uint8Array(OPENCODE_ID_LENGTH - 12));
    suffix = Array.from(bytes as Uint8Array, (byte: number) => OPENCODE_ID_BASE62[byte % 62]).join("");
  } else {
    suffix = Array.from({ length: OPENCODE_ID_LENGTH - 12 }, () =>
      OPENCODE_ID_BASE62[Math.floor(Math.random() * 62)],
    ).join("");
  }
  return time + suffix;
}

function createOpencodeSessionId(timestamp = Date.now()): string {
  return `ses_${createOpencodeId(true, timestamp)}`;
}

function createOpencodeRequestId(timestamp = Date.now()): string {
  return `msg_${createOpencodeId(false, timestamp)}`;
}

// Stable ses_ per caller string (e.g. Claude Code UUID): minted once with
// the exact generator above and cached, so stickiness keeps valid IDs
// instead of synthetic hashes the gate rejects.
const stableOpencodeSessionCache = new Map<string, string>();

function stableOpencodeSessionIdFrom(input: string): string {
  const hit = stableOpencodeSessionCache.get(input);
  if (hit) return hit;
  const minted = createOpencodeSessionId();
  stableOpencodeSessionCache.set(input, minted);
  if (stableOpencodeSessionCache.size > 5000) {
    const oldest = stableOpencodeSessionCache.keys().next();
    if (!oldest.done) stableOpencodeSessionCache.delete(oldest.value);
  }
  return minted;
}

function headerValue(headers: Record<string, string | undefined>, name: string): string | undefined {
  if (headers[name] !== undefined) return headers[name];
  const lower = name.toLowerCase();
  for (const [key, value] of Object.entries(headers)) {
    if (key.toLowerCase() === lower && value !== undefined) return value;
  }
  return undefined;
}

function stripGatewayModelSuffix(model: string): string {
  return model.trim().replace(/\[1m\]$/i, "").trim();
}

function isValidOpencodeId(value: string, prefix: "ses" | "msg"): boolean {
  return new RegExp(`^${prefix}_[0-9a-f]{12}[0-9A-Za-z]{14}$`).test(value);
}

// Fingerprint tools the /inference/v1 free gate requires (MITM + probes
// 2026-10-15): the request must declare these LOWERCASE names with
// BYTE-VERBATIM official definitions (matching is case-sensitive —
// capitalized `Bash` does NOT satisfy `bash` — and since 2026-10-15 dummy
// schemas/descriptions 403; only the captured definitions 200). Extra
// caller tools ride verbatim (verified tolerated). Injected definitions
// are cloaked from downstream responses.
const ZEN_FINGERPRINT_TOOLS = [
  "bash",
  "edit",
  "glob",
  "grep",
  "read",
  "skill",
  "task",
  "todowrite",
  "webfetch",
  "websearch",
  "write",
] as const;

function zenToolNameOf(tool: any): string {
  if (!tool || typeof tool !== "object") return "";
  if (typeof tool.name === "string") return tool.name;
  const fn = (tool as any).function;
  if (fn && typeof fn.name === "string") return fn.name;
  return "";
}

// Inject missing fingerprint tools into a wire `tools` array. Returns the
// exact lowercase names THIS request injected (for downstream cloaking).
// Caller tools are never renamed or dropped.
function ensureZenFingerprintTools(
  tools: any[] | undefined,
  makePlaceholder: (name: string) => any,
): { tools: any[]; injected: string[] } {
  const list = Array.isArray(tools) ? [...tools] : [];
  const present = new Set(list.map(zenToolNameOf).filter(Boolean));
  const injected: string[] = [];
  for (const name of ZEN_FINGERPRINT_TOOLS) {
    if (!present.has(name)) {
      list.push(makePlaceholder(name));
      injected.push(name);
    }
  }
  return { tools: list, injected };
}

function trackZenInjected(body: any, injected: string[], callerToolNames: string[] = []): void {
  Object.defineProperty(body, "__zenInjected", {
    value: injected,
    enumerable: false,
    writable: true,
  });
  Object.defineProperty(body, "__zenCallerToolNames", {
    value: callerToolNames,
    enumerable: false,
    writable: true,
  });
}

function zenInjectedOf(body: any): string[] {
  const v = (body as any)?.__zenInjected;
  return Array.isArray(v) ? v : [];
}

function zenCallerToolNamesOf(body: any): string[] {
  const v = (body as any)?.__zenCallerToolNames;
  return Array.isArray(v) ? v : [];
}

// Non-enumerable tracking is lost by object spread; re-attach it after
// building derived wire bodies (search fallback turns).
function carryZenTracking(source: any, target: any): any {
  for (const key of ["__zenInjected", "__zenCallerToolNames"]) {
    const value = source?.[key];
    if (value !== undefined) {
      Object.defineProperty(target, key, { value, enumerable: false, writable: true });
    }
  }
  return target;
}

// Injected placeholders that case-insensitively match a caller tool
// (`bash` vs `Bash`, `read` vs `Read`, ...) are remapped on the way out
// instead of dropped: the model may legitimately pick the lowercase zen
// name, while Claude Code only knows the caller's spelling.
function zenInjectedRemap(body: any): Record<string, string> {
  const callers = new Map(
    zenCallerToolNamesOf(body).map((name) => [name.toLowerCase(), name] as const),
  );
  const remap: Record<string, string> = {};
  for (const name of zenInjectedOf(body)) {
    const caller = callers.get(name.toLowerCase());
    if (caller) remap[name] = caller;
  }
  return remap;
}

function combineChatUsage(total: any, add: any): any {
  if (!add) return total;
  if (!total) return add;
  const cached = (usage: any) => usage?.prompt_tokens_details?.cached_tokens || 0;
  return {
    ...total,
    prompt_tokens: (total.prompt_tokens || 0) + (add.prompt_tokens || 0),
    completion_tokens: (total.completion_tokens || 0) + (add.completion_tokens || 0),
    total_tokens: (total.total_tokens || 0) + (add.total_tokens || 0),
    prompt_tokens_details: {
      ...(total.prompt_tokens_details || {}),
      cached_tokens: cached(total) + cached(add),
    },
  };
}

// Free chat models can invoke the injected zen fingerprint placeholders
// (`todowrite`, `glob`, ...). With no caller equivalent, cloaking would drop
// the tool_use and the client would see a bare end_turn — Claude Code then
// stalls mid-task right after a thinking block. Feed a synthetic tool result
// back and let the model retry with real tools in the same client request.
function appendZenPlaceholderFeedback(
  body: any,
  calls: any[],
  unhandled: Set<string>,
  callerToolNames: string[],
): any {
  const available = callerToolNames.length
    ? callerToolNames.slice(0, 40).join(", ")
    : "no client tools are available";
  const messages = [...(body.messages ?? [])];
  messages.push({
    role: "assistant",
    content: null,
    tool_calls: calls.map((tc) => ({
      id: tc.id,
      type: "function",
      function: { name: tc.function?.name || "", arguments: tc.function?.arguments || "{}" },
    })),
  });
  for (const tc of calls) {
    const name = tc.function?.name || "";
    messages.push({
      role: "tool",
      tool_call_id: tc.id,
      content: unhandled.has(name)
        ? `Tool "${name}" is not available in this session. Available tools: ${available}. ` +
          "Re-issue this step using an available tool, or reply to the user directly without calling a tool."
        : `Tool "${name}" is unavailable in this session; continue with an available tool.`,
    });
  }
  return { ...body, messages };
}

async function collectSSEDataPayloads(response: Response): Promise<string[]> {
  const payloads: string[] = [];
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const pushLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith("data:")) return;
    const data = trimmed.slice(5).trim();
    if (!data || data === "[DONE]") return;
    payloads.push(data);
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) pushLine(line);
    }
    if (buffer.trim()) pushLine(buffer);
  } finally {
    try {
      reader.releaseLock();
    } catch {}
  }
  return payloads;
}

function assembleChatCompletion(payloads: string[], fallbackModel: string): any {
  let id = `chatcmpl_${Date.now()}`;
  let model = fallbackModel;
  let created = Math.floor(Date.now() / 1000);
  let content = "";
  let reasoning = "";
  let finish: string | null = null;
  const toolCalls = new Map<number, { id?: string; name?: string; args: string }>();
  let usage: any = undefined;
  for (const p of payloads) {
    let chunk: any;
    try {
      chunk = JSON.parse(p);
    } catch {
      continue;
    }
    if (chunk.error) throw new Error(JSON.stringify(chunk.error).slice(0, 500));
    if (typeof chunk.id === "string") id = chunk.id;
    if (typeof chunk.model === "string") model = chunk.model;
    if (typeof chunk.created === "number") created = chunk.created;
    if (chunk.usage) usage = chunk.usage;
    const choice = chunk.choices?.[0];
    if (!choice) continue;
    const d = choice.delta || {};
    if (typeof d.content === "string") content += d.content;
    if (typeof d.reasoning_content === "string") reasoning += d.reasoning_content;
    for (const tc of d.tool_calls || []) {
      const idx = tc.index ?? 0;
      const cur = toolCalls.get(idx) || { args: "" };
      if (typeof tc.id === "string") cur.id = tc.id;
      if (typeof tc.function?.name === "string") cur.name = tc.function.name;
      if (typeof tc.function?.arguments === "string") cur.args += tc.function.arguments;
      toolCalls.set(idx, cur);
    }
    if (typeof choice.finish_reason === "string") finish = choice.finish_reason;
  }
  const message: any = { role: "assistant", content: content || null };
  if (reasoning) message.reasoning_content = reasoning;
  if (toolCalls.size) {
    message.tool_calls = [...toolCalls.entries()]
      .sort((a, b) => a[0] - b[0])
      .map(([_, tc], i) => ({
        id: tc.id || `call_${i}`,
        type: "function",
        function: { name: tc.name || "", arguments: tc.args || "" },
      }));
  }
  return {
    id,
    object: "chat.completion",
    created,
    model,
    choices: [{ index: 0, message, finish_reason: finish || "stop" }],
    usage: usage || { prompt_tokens: 0, completion_tokens: 0 },
  };
}

function assembleResponses(payloads: string[]): any {
  for (const p of payloads) {
    let chunk: any;
    try {
      chunk = JSON.parse(p);
    } catch {
      continue;
    }
    if (chunk.type === "response.completed" && chunk.response) return chunk.response;
    if (chunk.type === "response.failed" || chunk.type === "response.incomplete") {
      throw new Error(
        `upstream responses stream ended with ${chunk.type}`.slice(0, 500),
      );
    }
    if (chunk.error) throw new Error(JSON.stringify(chunk.error).slice(0, 500));
  }
  throw new Error("upstream responses stream ended without response.completed");
}

export class OpencodeGoTransformer implements Transformer {
  name = "opencode-go";
  logger?: any;

  private isChatCompletionsModel(model: string): boolean {
    return isOpencodeChatCompletionsModel(model);
  }

  private isFreeModel(model: string): boolean {
    return isOpencodeFreeModel(model);
  }

  private chatEndpointForModel(model: string): string {
    return this.isFreeModel(model) ? FREE_CHAT_ENDPOINT : CHAT_ENDPOINT;
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    _provider: LLMProvider,
    _context: TransformerContext,
  ): Promise<any> {
    if (request.tools) {
      request.tools = request.tools.map((tool) => {
        if (!isFunctionTool(tool)) return tool;
        if (tool.function?.parameters) {
          const cleanedParams = this.cleanJsonSchema(
            tool.function.parameters as Record<string, any>
          );
          return {
            type: "function" as const,
            function: {
              name: tool.function.name,
              description: tool.function.description,
              parameters: cleanedParams as any,
            },
          };
        }
        return tool;
      });
    }
    return request;
  }

  async sendRequest(
    requestInput: UnifiedChatRequest,
    config: any,
    _provider: LLMProvider,
    context: TransformerContext,
  ): Promise<Response> {
    // Gateway variants like `model[1m]` (1M-context flag) must never reach the
    // transformer logic or the upstream body: Zen validates the bare id, and
    // the suffix breaks free-tier detection (`endsWith("-free")`) and endpoint
    // selection. Normalize on a copy; never mutate the caller's request.
    const request = { ...requestInput, model: stripGatewayModelSuffix(requestInput.model || "") };
    const model = request.model;
    // Free-tier topology (MITM-verified 2026-10-13 on /inference/v1):
    // muse-spark-*-free are RESPONSES-only (chat 400s ModelProtocolUnsupported),
    // all other free models are CHAT-only (responses 400s the same way).
    if (this.isFreeModel(model)) {
      if (model.startsWith("muse-spark")) {
        const body = this.unifiedToResponses(request);
        return this.fetchEndpoint(body, ZEN_RESPONSES_ENDPOINT, config, request.stream ?? false, context);
      }
      const freeChatEndpoint = FREE_CHAT_ENDPOINT;
      const searchBudget = webSearchFallbackBudget((request as any).tools);
      if (searchBudget && readWebSearchFallbackConfig()) {
        const body = this.unifiedToOpenAI(request, model);
        return this.sendChatWithSearchFallback(
          body,
          freeChatEndpoint,
          config,
          request.stream ?? false,
          context,
          searchBudget.maxUses,
        );
      }
      const body = this.unifiedToOpenAI(request, model);
      return this.fetchEndpoint(body, freeChatEndpoint, config, request.stream ?? false, context);
    }
    // Muse Spark paid models use Responses API per docs/zen and docs/go
    if (model.startsWith("muse-spark")) {
      const endpoint = GO_RESPONSES_ENDPOINT;
      const body = this.unifiedToResponses(request);
      return this.fetchEndpoint(body, endpoint, config, request.stream ?? false, context);
    }
    const endpoint = this.chatEndpointForModel(model);

    // Chat-completions has no native web search. When the client declared a
    // server web_search tool and search engines are configured, run the
    // gateway-executed fallback loop instead of the legacy silent drop.
    // Native paths (Responses web_search, Gemini google_search) never reach
    // here. With no search keys configured, fetchEndpoint keeps the legacy
    // drop behavior (mapChatTools warns per tool).
    const searchBudget = webSearchFallbackBudget((request as any).tools);
    if (searchBudget && readWebSearchFallbackConfig()) {
      const body = this.unifiedToOpenAI(request, model);
      return this.sendChatWithSearchFallback(
        body,
        endpoint,
        config,
        request.stream ?? false,
        context,
        searchBudget.maxUses,
      );
    }

    if (this.isChatCompletionsModel(model)) {
      const body = this.unifiedToOpenAI(request, model);
      return this.fetchEndpoint(body, endpoint, config, request.stream ?? false, context);
    }

    const body = this.unifiedToOpenAI(request, model);
    return this.fetchEndpoint(body, endpoint, config, request.stream ?? false, context);
  }

  private async fetchUpstream(
    body: any,
    url: string,
    config: any,
    isStream: boolean,
    context: TransformerContext,
  ): Promise<Response> {
    const reqLog = (context.req as any)?.log;
    maybeRefreshOpencodeUserAgent(reqLog);
    // config.headers first so harness UA + opencode headers always win,
    // mirroring harness `...model.headers, ...plugin.headers` ordering where
    // the base harness headers are authoritative.
    const rawConfigHeaders = { ...(config.headers || {}) };
    // Gate rejects SDK fingerprints (verified 2026-10-11).
    for (const key of Object.keys(rawConfigHeaders)) {
      if (key.toLowerCase().startsWith("x-stainless-")) delete rawConfigHeaders[key];
    }
    const isResponsesUrl = url.includes("/responses");
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...rawConfigHeaders,
      "User-Agent": getOpencodeUserAgent(config, isResponsesUrl ? "responses" : "chat"),
    };

    const isFree = this.isFreeModel(body.model);
    const endpointApiKey = isFree ? config.api_key_zen : config.api_key_go;
    if (endpointApiKey) {
      headers["Authorization"] = `Bearer ${endpointApiKey}`;
    } else if (config.api_key) {
      headers["Authorization"] = `Bearer ${config.api_key}`;
    } else if (isFree) {
      // Keyless free tier: official CLI sends `Bearer public`.
      headers["Authorization"] = `Bearer public`;
    }

    const reqHeaders = context.req?.headers || {};

    // Harness parity: session is `ses_…` stable per conversation, request is a
    // fresh `msg_…` per HTTP call. Never reuse the session id as request id
    // and never send `ccr_…` (wrong prefix + unstable breaks sticky routing
    // and prompt-cache reuse, both of which the server monitors).
    const passthroughSession = headerValue(reqHeaders, "x-opencode-session");
    let opencodeSession: string;
    if (passthroughSession && passthroughSession.trim()) {
      const trimmed = passthroughSession.trim();
      opencodeSession = isValidOpencodeId(trimmed, "ses") ? trimmed : stableOpencodeSessionIdFrom(trimmed);
    } else {
      const rawBody = (context.req as any)?.body;
      const rawMeta = rawBody?.metadata?.user_id;
      let candidate: string | undefined;
      if (typeof rawMeta === "string") {
        try {
          const parsed = JSON.parse(rawMeta);
          candidate = parsed.session_id || parsed.device_id;
        } catch {}
      }
      if (!candidate && typeof rawBody?.session_id === "string") {
        candidate = rawBody.session_id;
      }
      if (!candidate && typeof (context.req as any)?.id === "string") {
        candidate = (context.req as any).id;
      }
      opencodeSession = candidate?.trim()
        ? isValidOpencodeId(candidate.trim(), "ses")
          ? candidate.trim()
          : stableOpencodeSessionIdFrom(candidate.trim())
        : createOpencodeSessionId();
    }
    headers["x-opencode-session"] = opencodeSession;
    // Official CLI sends both headers with the same value (MITM 2026-10-13).
    headers["x-opencode-session-id"] = opencodeSession;

    // Harness always sends project (`global` when no git repo). Omitting it
    // is itself a non-harness signal.
    const passthroughProject = headerValue(reqHeaders, "x-opencode-project");
    headers["x-opencode-project"] =
      passthroughProject && passthroughProject.trim() ? passthroughProject.trim() : "global";
    // Fresh unique request id per upstream call, like harness `input.user.id`.
    const passthroughRequest = headerValue(reqHeaders, "x-opencode-request");
    headers["x-opencode-request"] =
      passthroughRequest && isValidOpencodeId(passthroughRequest.trim(), "msg")
        ? passthroughRequest.trim()
        : createOpencodeRequestId();
    headers["x-opencode-client"] = headerValue(reqHeaders, "x-opencode-client") || "cli";
    const parentSession = headerValue(reqHeaders, "x-parent-session-id");
    if (parentSession && parentSession.trim()) {
      headers["x-parent-session-id"] = parentSession.trim();
    }

    // Free-tier gate: official CLI always sends stream:true; non-streaming
    // bodies get 403 FreeTierError. Force stream:true upstream for free
    // models; when the downstream client asked non-stream, the SSE is
    // consumed below and collapsed back into a single response.
    // prompt_cache_key = ses id (official wire shape, MITM 2026-10-13).
    const isResponsesEndpoint = url.includes("/responses");
    if (isFree && isResponsesEndpoint && body && typeof body === "object" && !body.prompt_cache_key) {
      body.prompt_cache_key = opencodeSession;
    }
    const forceStream = isFree && !isStream;
    const upstreamBody = forceStream
      ? {
          ...body,
          stream: true,
          ...(isResponsesEndpoint ? {} : { stream_options: { include_usage: true } }),
        }
      : body;

    // Upstream body-size cliff (verified 2026-09-25 from request logs):
    // zen chat bodies >~2MB get a bare 400 "invalid request" (1.71MB passed,
    // 2.37MB+ failed repeatedly). Log early so the next 400 doesn't require
    // another log-diving session; the fix is client-side (compact context).
    const upstreamBodyJson = JSON.stringify(upstreamBody);
    if (upstreamBodyJson.length > 1_800_000) {
      reqLog?.warn?.({
        opencodeGoOversizeBody: true,
        opencodeGoBodyBytes: upstreamBodyJson.length,
        opencodeGoModel: body?.model || "unknown",
      });
    }

    const fetchOptions: RequestInit & { dispatcher?: any } = {
      method: "POST",
      headers,
      body: upstreamBodyJson,
    };

    if (config.httpsProxy) {
      fetchOptions.dispatcher = new ProxyAgent(new URL(config.httpsProxy).toString());
    }

    let response = await fetch(url, fetchOptions);

    // Reactive UA refresh: a FreeTierError may mean our cached harness triple
    // went stale (new opencode release). Refresh once and retry once.
    if (isFree && (response.status === 403 || response.status === 400)) {
      const peek = await response
        .clone()
        .text()
        .catch(() => "");
      if (peek.includes("FreeTierError")) {
        const freshUA = await refreshOpencodeUserAgent("free-tier-error", reqLog).catch(() =>
          getOpencodeUserAgent(config, isResponsesEndpoint ? "responses" : "chat"),
        );
        headers["User-Agent"] = config?.opencode_user_agent || freshUA;
        fetchOptions.headers = headers;
        response = await fetch(url, fetchOptions);
      }
    }

    if (!response.ok && response.status === 429) {
      const maxRetries = 5;
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        const retryAfterMs = Number.parseFloat(response.headers.get("retry-after-ms") || "");
        const retryAfterSec = Number.parseFloat(response.headers.get("retry-after") || "");
        const base = 2000 * Math.pow(2, attempt - 1);
        const jitter = Math.ceil(base * 0.25 * Math.random());
        const delayMs = Number.isFinite(retryAfterMs) && retryAfterMs > 0
          ? retryAfterMs
          : Number.isFinite(retryAfterSec) && retryAfterSec > 0
            ? Math.ceil(retryAfterSec * 1000)
            : Math.min(base + jitter, 30_000);
        reqLog?.warn?.({
          opencodeGoRateLimited: true,
          opencodeGoRetryInMs: delayMs,
          opencodeGoAttempt: attempt,
        });
        await new Promise((resolve) => setTimeout(resolve, delayMs));
        response = await fetch(url, fetchOptions);
        if (response.ok || response.status !== 429) {
          break;
        }
      }
    }

    if (!response.ok) {
      const errorBody = await response.text().catch(() => "no body");
      reqLog?.error({
        opencodeGoStatus: response.status,
        opencodeGoError: errorBody.substring(0, 1500),
      });
      return new Response(errorBody, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    return response;
  }

  private async fetchEndpoint(
    body: any,
    url: string,
    config: any,
    isStream: boolean,
    context: TransformerContext,
  ): Promise<Response> {
    const reqLog = (context.req as any)?.log;
    const isFree = this.isFreeModel(body.model);
    const forceStream = isFree && !isStream;
    const response = await this.fetchUpstream(body, url, config, isStream, context);
    if (!response.ok) {
      const errorBody = await response.text().catch(() => "no body");
      reqLog?.error({
        opencodeGoStatus: response.status,
        opencodeGoError: errorBody.substring(0, 1500),
      });
      return new Response(errorBody, {
        status: response.status,
        statusText: response.statusText,
        headers: response.headers,
      });
    }

    const isResponses = url.includes("/responses");
    const zenInjected = isFree ? zenInjectedOf(body) : [];
    const zenRemap = isFree ? zenInjectedRemap(body) : {};
    // Buffered exactly once here so both the streaming and the collapse paths
    // can run placeholder recovery (internal retry) before conversion.
    const freeData =
      isFree && !isResponses && zenInjected.length && response.body
        ? await this.assembleFreeChatWithPlaceholderRecovery(response, body, url, config, isStream, context)
        : undefined;
    if (isStream) {
      if (isFree) {
        if (!freeData && !response.body) {
          return response;
        }
        // Free tier: injected fingerprint calls must never reach downstream
        // (the model may invoke the placeholder tools). Buffer, assemble,
        // cloak, then emit clean Anthropic SSE.
        return this.bufferFreeStreamToAnthropic(
          response,
          body,
          context,
          zenInjected,
          isResponses,
          freeData,
          zenRemap,
        );
      }
      if (!response.body) {
        return response;
      }
      if (isResponses) {
        // Convert Responses SSE to Anthropic SSE. The gateway returns our
        // bytes directly to Claude Code (a custom sendRequest skips endpoint
        // response conversion), so passing raw response.* events through
        // yields zero parseable stream events client-side — the same failure
        // mode the Codex transformer had.
        const converter = new ResponsesToAnthropicStream(body.model || "unknown", {
          onUnknownEvent: (type) =>
            this.logger?.warn?.(`[opencode-go] Ignoring unknown upstream event type: ${type}`),
        });
        const convertedStream = this.convertResponsesStreamToAnthropic(
          response.body,
          converter,
        );
        return new Response(convertedStream, {
          status: response.status,
          statusText: response.statusText,
          headers: {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            Connection: "keep-alive",
          },
        });
      }
      const convertedStream = this.convertOpenAIStreamToAnthropic(response.body, context, (type) =>
        this.logger?.warn?.(`[opencode-go] Ignoring unknown upstream delta shape: ${type}`),
      );
      return new Response(convertedStream, {
        status: response.status,
        statusText: response.statusText,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }

    let data: any;
    if (forceStream) {
      if (freeData) {
        data = freeData;
      } else {
        // Upstream was forced to stream (free-tier gate) while downstream wants
        // a single response: consume the SSE and collapse it back.
        if (!response.body) {
          return new Response(JSON.stringify({ error: "empty upstream response" }), {
            status: 502,
            headers: { "Content-Type": "application/json" },
          });
        }
        try {
          const payloads = await collectSSEDataPayloads(response);
          data = isResponses
            ? assembleResponses(payloads)
            : assembleChatCompletion(payloads, body.model || "unknown");
        } catch (e: any) {
          reqLog?.error({
            opencodeGoStatus: response.status,
            opencodeGoError: `forced-stream collapse failed: ${e?.message || e}`.substring(0, 500),
          });
          return new Response(JSON.stringify({ error: e?.message || "upstream stream error" }), {
            status: 502,
            headers: { "Content-Type": "application/json" },
          });
        }
      }
    } else {
      data = await response.json();
    }
    if (isResponses) {
      const anthropicResponse = this.convertResponsesToAnthropic(data, zenInjected, zenRemap);
      return new Response(JSON.stringify(anthropicResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: { "Content-Type": "application/json" },
      });
    }
    const anthropicResponse = this.convertOpenAIResponseToAnthropic(data, zenInjected, zenRemap);
    return new Response(JSON.stringify(anthropicResponse), {
      status: response.status,
      statusText: response.statusText,
      headers: { "Content-Type": "application/json" },
    });
  }

  // Free chat models sometimes invoke the injected zen fingerprint
  // placeholders (`todowrite`, `glob`, `task`, ...). Cloaking drops those
  // tool_use blocks, and the client would see a bare end_turn — Claude Code
  // then stalls mid-task right after a thinking block. Before conversion,
  // feed the model synthetic tool results for the unavailable placeholders
  // and retry upstream (bounded) so it can re-issue with real tools inside
  // the same client request. Usage is summed across attempts.
  private async assembleFreeChatWithPlaceholderRecovery(
    response: Response,
    body: any,
    url: string,
    config: any,
    isStream: boolean,
    context: TransformerContext,
  ): Promise<any> {
    const reqLog = (context.req as any)?.log;
    const injectedSet = new Set(zenInjectedOf(body));
    const remap = zenInjectedRemap(body);
    const callerToolNames = zenCallerToolNamesOf(body);
    const maxRetries = 2;
    let data = assembleChatCompletion(
      await collectSSEDataPayloads(response),
      body.model || "unknown",
    );
    let usage = data.usage;
    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      const calls: any[] = data?.choices?.[0]?.message?.tool_calls ?? [];
      const unhandled = calls.filter(
        (tc) => injectedSet.has(tc?.function?.name) && !remap[tc.function.name],
      );
      if (!unhandled.length) break;
      const hasUsableCall = calls.some(
        (tc) => !injectedSet.has(tc?.function?.name) || remap[tc?.function?.name],
      );
      // Real/remappable calls keep the turn alive; remaining placeholders drop.
      if (hasUsableCall) break;
      reqLog?.warn?.({
        opencodeGoPlaceholderRetry: attempt,
        opencodeGoPlaceholderTools: unhandled.map((tc) => tc.function.name),
      });
      const retryBody = appendZenPlaceholderFeedback(
        body,
        calls,
        new Set(unhandled.map((tc) => tc.function?.name || "")),
        callerToolNames,
      );
      let retryResponse: Response;
      try {
        retryResponse = await this.fetchUpstream(retryBody, url, config, isStream, context);
      } catch (e: any) {
        reqLog?.error?.({
          opencodeGoPlaceholderRetryFailed: String(e?.message || e).substring(0, 300),
        });
        break;
      }
      if (!retryResponse.ok || !retryResponse.body) {
        reqLog?.error?.({ opencodeGoPlaceholderRetryStatus: retryResponse.status });
        break;
      }
      const retryData = assembleChatCompletion(
        await collectSSEDataPayloads(retryResponse),
        body.model || "unknown",
      );
      usage = combineChatUsage(usage, retryData.usage);
      data = retryData;
    }
    if (usage) data.usage = usage;
    return data;
  }

  private async bufferFreeStreamToAnthropic(
    response: Response,
    body: any,
    context: TransformerContext,
    injected: string[],
    isResponses: boolean,
    preData?: any,
    remap: Record<string, string> = {},
  ): Promise<Response> {
    const reqLog = (context.req as any)?.log;
    try {
      let data = preData;
      if (!data) {
        const payloads = await collectSSEDataPayloads(response);
        data = isResponses
          ? assembleResponses(payloads)
          : assembleChatCompletion(payloads, body.model || "unknown");
      }
      const anthropicResponse = isResponses
        ? this.convertResponsesToAnthropic(data, injected, remap)
        : this.convertOpenAIResponseToAnthropic(data, injected, remap);
      const sse = anthropicMessageToSSE({
        id: anthropicResponse.id,
        model: anthropicResponse.model,
        content: anthropicResponse.content,
        stop_reason: anthropicResponse.stop_reason,
        usage: anthropicResponse.usage,
      });
      return new Response(sse, {
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    } catch (e: any) {
      reqLog?.error?.({
        opencodeGoStatus: 502,
        opencodeGoError: `free stream buffer failed: ${e?.message || e}`.substring(0, 500),
      });
      return new Response(JSON.stringify({ error: e?.message || "upstream stream error" }), {
        status: 502,
        headers: { "Content-Type": "application/json" },
      });
    }
  }

  /**
   * Gateway-executed web search for the chat path (see websearch-fallback).
   * Upstream turns are always buffered (isStream=false into fetchEndpoint,
   * which already handles the free-tier forceStream collapse), then the
   * final transcript — executed searches as native citation blocks plus the
   * model's last turn — is returned as Anthropic SSE or JSON depending on
   * what downstream asked for.
   */
  private async sendChatWithSearchFallback(
    body: any,
    endpoint: string,
    config: any,
    downstreamStream: boolean,
    context: TransformerContext,
    maxUses: number,
  ): Promise<Response> {
    const searchConfig = readWebSearchFallbackConfig();
    if (!searchConfig) {
      return this.fetchEndpoint(body, endpoint, config, downstreamStream, context);
    }
    const reqLog = (context.req as any)?.log;
    const warn = (message: string) => {
      if (this.logger?.warn) this.logger.warn(`[opencode-go] ${message}`);
      else reqLog?.warn?.({ websearchFallback: message });
    };

    const chatMessages: any[] = [...(body.messages ?? [])];
    const chatTools: any[] = [...(body.tools ?? []), fallbackFunctionTool()];
    const executed: ExecutedSearch[] = [];
    const usage = { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0 };
    const maxTurns = Math.min(searchConfig.maxTurns, maxUses + 2);
    let finalTurn: any = null;
    let lastTurn: any = null;

    const toChatToolCalls = (toolUses: any[]) =>
      toolUses.map((t: any) => ({
        id: t.id,
        type: "function" as const,
        function: { name: t.name, arguments: JSON.stringify(t.input ?? {}) },
      }));
    const turnText = (turn: any): string | null => {
      const parts = (turn?.content ?? [])
        .filter((b: any) => b?.type === "text" && typeof b.text === "string")
        .map((b: any) => b.text);
      return parts.length > 0 ? parts.join("") : null;
    };

    for (let turn = 0; turn <= maxTurns && !finalTurn; turn++) {
      const turnBody = carryZenTracking(body, {
        ...body,
        messages: chatMessages,
        tools: chatTools,
        tool_choice: body.tool_choice ?? "auto",
      });
      const res = await this.fetchEndpoint(turnBody, endpoint, config, false, context);
      if (!res.ok) return res;
      let anth: any = null;
      try {
        anth = await res.json();
      } catch {
        anth = null;
      }
      if (!anth || typeof anth !== "object" || !Array.isArray((anth as any).content)) {
        return new Response(JSON.stringify(anth ?? { error: "unparseable upstream response" }), {
          status: 502,
          headers: { "Content-Type": "application/json" },
        });
      }
      lastTurn = anth;
      usage.input_tokens += anth.usage?.input_tokens || 0;
      usage.output_tokens += anth.usage?.output_tokens || 0;
      usage.cache_read_input_tokens += anth.usage?.cache_read_input_tokens || 0;

      const toolUses = anth.content.filter((b: any) => b?.type === "tool_use");
      const searchCalls = toolUses.filter((t: any) => t?.name === FALLBACK_FN_NAME);
      const realCalls = toolUses.filter((t: any) => t?.name !== FALLBACK_FN_NAME);
      if (searchCalls.length === 0) {
        finalTurn = anth;
        break;
      }

      chatMessages.push({
        role: "assistant",
        content: turnText(anth),
        tool_calls: toChatToolCalls(toolUses),
      });

      if (executed.length >= maxUses) {
        for (const sc of searchCalls) {
          chatMessages.push({
            role: "tool",
            tool_call_id: sc.id,
            content:
              "Web search budget exhausted (max_uses). Answer using the results gathered so far.",
          });
        }
        if (realCalls.length > 0) {
          finalTurn = { ...anth, stop_reason: "tool_use" };
        }
        continue;
      }

      for (const sc of searchCalls) {
        const query = fallbackQueryOf(sc);
        if (!query) {
          chatMessages.push({
            role: "tool",
            tool_call_id: sc.id,
            content: "Search query must be a non-empty string.",
          });
          continue;
        }
        try {
          const { results, engine } = await executeWebSearchChain(query, searchConfig, (eng, msg) =>
            warn(`search engine ${eng} failed, trying next: ${msg}`),
          );
          const done = createExecutedSearch(query, engine, results);
          executed.push(done);
          chatMessages.push({
            role: "tool",
            tool_call_id: sc.id,
            content: searchToolResultPayload(done),
          });
        } catch (e: any) {
          reqLog?.error?.({
            websearchFallbackFailed: String(e?.message || e).slice(0, 300),
          });
          chatMessages.push({
            role: "tool",
            tool_call_id: sc.id,
            content:
              "Web search is unavailable right now. Answer from knowledge and say so explicitly.",
          });
        }
      }

      if (realCalls.length > 0) {
        // Mixed turn: fallback searches executed above; real tools belong to
        // the client, so surface everything and let it continue the loop.
        finalTurn = { ...anth, stop_reason: "tool_use" };
      }
    }

    if (!finalTurn) {
      warn(`search loop hit turn cap (${maxTurns}); surfacing last turn`);
      finalTurn = lastTurn ?? {
        id: `msg_${Date.now()}`,
        model: body.model || "unknown",
        content: [],
        stop_reason: "end_turn",
        usage: { input_tokens: 0, output_tokens: 0 },
      };
    }

    const blocks: any[] = [...synthesizeSearchBlocks(executed)];
    for (const b of finalTurn.content ?? []) {
      // Fallback calls are already represented as server_tool_use blocks.
      if (b?.type === "tool_use" && b?.name === FALLBACK_FN_NAME) continue;
      blocks.push(b);
    }
    const hasRealToolUse = blocks.some((b: any) => b?.type === "tool_use");
    const message = {
      id: finalTurn.id || `msg_${Date.now()}`,
      type: "message",
      role: "assistant",
      model: body.model || finalTurn.model || "unknown",
      content: blocks,
      stop_reason: hasRealToolUse ? "tool_use" : finalTurn.stop_reason || "end_turn",
      stop_sequence: null,
      usage,
    };

    if (downstreamStream) {
      const sse = anthropicMessageToSSE({
        id: message.id,
        model: message.model,
        content: message.content,
        stop_reason: message.stop_reason,
        usage: message.usage,
      });
      return new Response(sse, {
        status: 200,
        headers: {
          "Content-Type": "text/event-stream",
          "Cache-Control": "no-cache",
          Connection: "keep-alive",
        },
      });
    }
    return new Response(JSON.stringify(message), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  }

  private unifiedToOpenAI(unified: UnifiedChatRequest, model?: string): any {
    const messages: any[] = [];
    const supportsImages = !model?.startsWith("deepseek-");
    for (const msg of unified.messages) {
      if (msg.role === "system") {
        const text = typeof msg.content === "string" ? msg.content : null;
        if (text) {
          messages.push({ role: "system", content: text });
        }
        continue;
      }

      if (msg.role === "assistant") {
        const m: any = { role: "assistant" };
        if (typeof msg.content === "string") {
          m.content = msg.content;
        } else {
          m.content = null;
        }
        if ((msg as any).thinking?.content) {
          m.reasoning_content = (msg as any).thinking.content;
        } else if (msg.tool_calls?.length) {
          m.reasoning_content = " ";
        }
        if (msg.tool_calls?.length) {
          m.tool_calls = msg.tool_calls.map((tc) => ({
            id: tc.id,
            type: "function" as const,
            function: { name: tc.function.name, arguments: tc.function.arguments },
          }));
        }
        messages.push(m);
        continue;
      }

      if (msg.role === "tool") {
        messages.push({
          role: "tool",
          content: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
          tool_call_id: msg.tool_call_id,
        });
        continue;
      }

      const m: any = { role: "user" };
      if (typeof msg.content === "string") {
        m.content = msg.content;
      } else if (Array.isArray(msg.content)) {
        m.content = msg.content.map((part: any) => {
          if (part.type === "image_url") {
            if (supportsImages) {
              return { type: "image_url", image_url: { url: part.image_url.url } };
            }
            return { type: "text", text: "[Image]" };
          }
          return { type: "text", text: part.text || "" };
        });
      } else {
        m.content = "";
      }
      messages.push(m);
    }

    const body: any = {
      model: unified.model,
      messages,
      stream: unified.stream ?? false,
    };

    if (unified.max_tokens) body.max_tokens = unified.max_tokens;
    if (unified.temperature !== undefined) body.temperature = unified.temperature;
    // Chat completions has no server-side tool types: drop unrepresentable
    // entries (with a warning) instead of emitting malformed tools that fail
    // the whole request. Function tools pass through byte-identical.
    if (unified.tools?.length) {
      const mapped = mapChatTools(unified.tools, (message) => this.logger?.warn?.(`[opencode-go] ${message}`));
      if (mapped) body.tools = mapped;
    }
    if (this.isFreeModel(unified.model)) {
      // /inference/v1 free gate requires the lowercase fingerprint set
      // with byte-verbatim official definitions (dummy schemas 403).
      const callerToolNames = (body.tools ?? []).map(zenToolNameOf).filter(Boolean);
      const chatDefs = new Map(
        ZEN_FINGERPRINT_CHAT_TOOLS.map((t: any) => [t?.function?.name, t]),
      );
      const { tools, injected } = ensureZenFingerprintTools(body.tools, (name) =>
        structuredClone(chatDefs.get(name)),
      );
      body.tools = tools;
      trackZenInjected(body, injected, callerToolNames);
      if (unified.tool_choice) body.tool_choice = unified.tool_choice;
      else if (body.tools.length) body.tool_choice = "auto";
    } else if (unified.tool_choice) {
      body.tool_choice = unified.tool_choice;
    }

    if (unified.reasoning?.enabled) {
      if (unified.model.startsWith("qwen-")) {
        body.enable_thinking = true;
      } else {
        body.thinking = { type: "enabled" };
        if (unified.reasoning.effort) {
          body.reasoning_effort = unified.reasoning.effort;
        }
      }
    }

    if (body.stream) {
      body.stream_options = { include_usage: true };
    }

    return body;
  }

  private unifiedToResponses(unified: UnifiedChatRequest): any {
    const input: any[] = [];
    for (const msg of unified.messages) {
      if (msg.role === "system") {
        const text = typeof msg.content === "string" ? msg.content : "";
        if (text) input.push({ role: "system", content: [{ type: "input_text", text }] });
        continue;
      }
      if (msg.role === "assistant") {
        const items: any[] = [];
        // Deliberately one-way: echoed array content (e.g. `thinking`
        // blocks with gateway-issued signatures) is dropped rather than
        // sent back as model thought. Text and tool calls round-trip.
        if (typeof msg.content === "string" && msg.content) {
          items.push({ role: "assistant", content: [{ type: "output_text", text: msg.content }] });
        }
        if (msg.tool_calls?.length) {
          for (const tc of msg.tool_calls) {
            items.push({
              type: "function_call",
              call_id: tc.id,
              name: tc.function.name,
              arguments: tc.function.arguments,
            });
          }
        }
        if (items.length) input.push(...items);
        continue;
      }
      if (msg.role === "tool") {
        input.push({
          type: "function_call_output",
          call_id: msg.tool_call_id,
          output: typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content),
        });
        continue;
      }
      // user
      if (typeof msg.content === "string") {
        input.push({ role: "user", content: [{ type: "input_text", text: msg.content }] });
      } else if (Array.isArray(msg.content)) {
        const parts = msg.content.map((part: any) => {
          if (part.type === "image_url" && part.image_url?.url) {
            return { type: "input_image", image_url: part.image_url.url };
          }
          return { type: "input_text", text: part.text || "" };
        });
        input.push({ role: "user", content: parts });
      } else {
        input.push({ role: "user", content: [{ type: "input_text", text: "" }] });
      }
    }
    const body: any = {
      model: unified.model,
      input,
      stream: unified.stream ?? false,
    };
    // Native tools ride verbatim on /inference/v1 (MITM 2026-10-13:
    // official main request carries native tools + tool_choice auto),
    // plus the lowercase fingerprint definitions the free gate requires
    // byte-verbatim (MITM 2026-10-15).
    if (unified.tools?.length) {
      const mapped = mapResponsesTools(unified.tools, (message) =>
        this.logger?.warn?.(`[opencode-go] ${message}`),
      );
      if (mapped) body.tools = mapped;
    }
    if (this.isFreeModel(unified.model)) {
      const callerToolNames = (body.tools ?? []).map(zenToolNameOf).filter(Boolean);
      const responsesDefs = new Map(
        ZEN_FINGERPRINT_RESPONSES_TOOLS.map((t: any) => [t?.name, t]),
      );
      const { tools, injected } = ensureZenFingerprintTools(body.tools, (name) =>
        structuredClone(responsesDefs.get(name)),
      );
      body.tools = tools;
      trackZenInjected(body, injected, callerToolNames);
      // Zen responses supports only auto-or-absent (other values 400).
      const choice = unified.tool_choice;
      if (typeof choice === "string" && choice.trim().toLowerCase() === "auto") {
        body.tool_choice = "auto";
      } else if (body.tools.length) {
        body.tool_choice = "auto";
      }
    } else if (unified.tool_choice) {
      body.tool_choice = unified.tool_choice;
    }
    if (this.isFreeModel(unified.model)) {
      this.applyInferenceFreeShape(body);
    } else if (unified.temperature !== undefined) {
      body.temperature = unified.temperature;
    }
    return body;
  }

  private applyInferenceFreeShape(body: any): void {
    // Wire shape of the official CLI on /inference/v1/responses
    // (MITM 2026-10-13): model, input, max_output_tokens, store:false,
    // include:[reasoning.encrypted_content], prompt_cache_key=ses,
    // tools, tool_choice, stream. prompt_cache_key is set in fetchEndpoint
    // once the session id is known; defaults that are always safe go here.
    delete body.temperature;
    delete body.top_p;
    delete body.response_format;
    delete body.max_tokens;
    delete body.max_completion_tokens;
    if (body.parallel_tool_calls === false) delete body.parallel_tool_calls;
    if (body.store === undefined) body.store = false;
    if (body.max_output_tokens === undefined) body.max_output_tokens = 32000;
    if (body.include === undefined) body.include = ["reasoning.encrypted_content"];
  }

  private extractZenToolCallsFromText(text: string): {
    cleanedText: string;
    toolCalls: Array<{ id: string; name: string; input: unknown }>;
  } {
    const toolCalls: Array<{ id: string; name: string; input: unknown }> = [];
    if (!text || !text.includes("```tool")) return { cleanedText: text, toolCalls };
    const fence = /```tool\s*\n([\s\S]*?)```/g;
    let counter = 0;
    const cleanedText = text
      .replace(fence, (_m, jsonText) => {
        try {
          const parsed = JSON.parse(String(jsonText).trim()) as { name?: unknown; arguments?: unknown };
          if (typeof parsed.name === "string" && parsed.name.length > 0) {
            counter += 1;
            let input: unknown = {};
            if (parsed.arguments !== undefined) {
              if (typeof parsed.arguments === "string") {
                try {
                  input = JSON.parse(parsed.arguments);
                } catch {
                  input = {};
                }
              } else if (typeof parsed.arguments === "object" && parsed.arguments !== null) {
                input = parsed.arguments;
              }
            }
            toolCalls.push({
              id: `call_zen_${Date.now()}_${counter}`,
              name: parsed.name,
              input,
            });
            return "";
          }
        } catch {}
        return _m;
      })
      .trim();
    return { cleanedText, toolCalls };
  }

  private applyZenTextProtocol(content: any[]): boolean {
    let hasToolUse = false;
    for (const block of content) {
      if (block?.type !== "text" || typeof block.text !== "string") continue;
      const { cleanedText, toolCalls } = this.extractZenToolCallsFromText(block.text);
      if (toolCalls.length === 0) continue;
      block.text = cleanedText;
      for (const tc of toolCalls) {
        content.push({ type: "tool_use", id: tc.id, name: tc.name, input: tc.input });
        hasToolUse = true;
      }
    }
    // Drop empty text blocks left after extraction
    for (let i = content.length - 1; i >= 0; i--) {
      if (content[i]?.type === "text" && !content[i].text) content.splice(i, 1);
    }
    return hasToolUse;
  }

  private cloakInjectedToolUse(
    content: any[],
    injected: string[],
    remap: Record<string, string> = {},
  ): boolean {
    if (!injected.length) return false;
    const drop = new Set(injected);
    let removed = false;
    for (let i = content.length - 1; i >= 0; i--) {
      const block = content[i];
      if (block?.type === "tool_use" && typeof block.name === "string" && drop.has(block.name)) {
        const callerName = remap[block.name];
        if (callerName) {
          block.name = callerName;
        } else {
          content.splice(i, 1);
          removed = true;
        }
      }
    }
    return removed;
  }

  private convertResponsesToAnthropic(
    data: any,
    injected: string[] = [],
    remap: Record<string, string> = {},
  ): any {
    const { content, hasToolUse: nativeToolUse } = responsesOutputToContent(data.output);
    // fallback to direct output_text if no structured output
    if (content.length === 0 && data.output_text) {
      content.push({ type: "text", text: data.output_text });
    }
    const textToolUse = this.applyZenTextProtocol(content);
    this.cloakInjectedToolUse(content, injected, remap);
    const hasToolUse =
      (nativeToolUse || textToolUse) && content.some((b: any) => b?.type === "tool_use");
    const usage = data.usage || {};
    return buildAnthropicMessage({
      id: data.id,
      model: data.model,
      content,
      stopReason: hasToolUse ? "tool_use" : "end_turn",
      usage: {
        input_tokens: usage.input_tokens ?? usage.prompt_tokens ?? 0,
        output_tokens: usage.output_tokens ?? usage.completion_tokens ?? 0,
        cache_read_input_tokens: usage.input_tokens_details?.cached_tokens || 0,
      },
    });
  }

  private convertResponsesStreamToAnthropic(
    stream: ReadableStream<Uint8Array>,
    converter: ResponsesToAnthropicStream,
  ): ReadableStream {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();

    return new ReadableStream({
      async start(controller) {
        let isClosed = false;
        const safeEnqueue = (data: string) => {
          if (isClosed) return;
          try {
            controller.enqueue(encoder.encode(data));
          } catch {
            isClosed = true;
          }
        };
        const safeClose = () => {
          if (isClosed) return;
          isClosed = true;
          try {
            controller.close();
          } catch {}
        };
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            for (const line of converter.push(chunk)) {
              safeEnqueue(line);
            }
            if (isClosed) break;
          }
          for (const line of converter.finish()) {
            safeEnqueue(line);
          }
        } catch (e) {
          if (!isClosed) {
            try {
              controller.error(e);
            } catch {}
            isClosed = true;
          }
        } finally {
          try {
            reader.releaseLock();
          } catch {}
          safeClose();
        }
      },
      cancel() {
        try {
          reader.cancel();
        } catch {}
      },
    });
  }

  private convertOpenAIStreamToAnthropic(
    stream: ReadableStream<Uint8Array>,
    _context: TransformerContext,
    onUnknownEvent?: (type: string) => void,
  ): ReadableStream {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();
    const warnedTypes = new Set<string>();
    const reportUnknown = (type: string) => {
      if (!type || warnedTypes.has(type)) return;
      warnedTypes.add(type);
      try {
        onUnknownEvent?.(type);
      } catch {
        // Reporting must never break conversion.
      }
    };

    return new ReadableStream({
      async start(controller) {
        const messageId = `msg_${Date.now()}`;
        let model = "unknown";
        let hasStarted = false;
        let isClosed = false;
        let isFinished = false;
        let usage: any = null;

        let nextBlockIdx = 0;

        let primaryBlockType: string | null = null;
        let primaryBlockIndex = -1;

        const toolCallBlocks = new Map<number, number>();

        const safeEnqueue = (data: Uint8Array) => {
          if (!isClosed) {
            try {
              controller.enqueue(data);
            } catch (e: any) {
              if (
                e instanceof TypeError &&
                e.message.includes("Controller is already closed")
              ) {
                isClosed = true;
              }
            }
          }
        };

        const sendLine = (line: string) => {
          safeEnqueue(encoder.encode(line));
        };

        const closePrimaryBlock = () => {
          if (primaryBlockIndex >= 0) {
            sendLine(contentBlockStopEvent(primaryBlockIndex));
            primaryBlockIndex = -1;
            primaryBlockType = null;
          }
        };

        const closeAllToolCallBlocks = () => {
          for (const [, blockIdx] of toolCallBlocks) {
            sendLine(contentBlockStopEvent(blockIdx));
          }
          toolCallBlocks.clear();
        };

        const closeAllBlocks = () => {
          closePrimaryBlock();
          closeAllToolCallBlocks();
        };

        const allocBlockIndex = () => {
          const idx = nextBlockIdx;
          nextBlockIdx++;
          return idx;
        };

        const sendMessageStart = () => {
          sendLine(messageStartEvent(messageId, model));
          hasStarted = true;
        };

        const emitFinish = (stopReason: string, usageData?: any) => {
          if (isFinished) return;
          isFinished = true;

          if (primaryBlockType === "thinking") {
            sendLine(signatureDeltaEvent(primaryBlockIndex, Date.now().toString()));
          }

          closeAllBlocks();

          if (!hasStarted) {
            sendMessageStart();
          }

          const stopReasonMap: Record<string, string> = {
            stop: "end_turn",
            length: "max_tokens",
            tool_calls: "tool_use",
            content_filter: "stop_sequence",
          };
          const anthropicStop = stopReasonMap[stopReason] || "end_turn";

          sendLine(
            messageDeltaEvent(anthropicStop as "end_turn", {
              input_tokens:
                (usageData?.prompt_tokens || 0) -
                (usageData?.prompt_tokens_details?.cached_tokens || 0),
              output_tokens: usageData?.completion_tokens || 0,
              cache_read_input_tokens:
                usageData?.prompt_tokens_details?.cached_tokens || 0,
            })
          );
          sendLine(messageStopEvent());

          try {
            controller.close();
          } catch {}
          isClosed = true;
        };

        try {
          let buffer = "";

          while (true) {
            if (isClosed || isFinished) break;

            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            for (const line of lines) {
              if (isClosed || isFinished || !line.trim()) continue;
              if (!line.startsWith("data:")) continue;

              const rawData = line.slice(5).trim();
              if (rawData === "[DONE]" || rawData === "") continue;

              try {
                const chunk = JSON.parse(rawData);
                if (chunk.error) {
                  // Preserve this converter's error shape (message, not error key).
                  sendLine(
                    sseEvent("error", {
                      type: "error",
                      message: {
                        type: "api_error",
                        message: JSON.stringify(chunk.error),
                      },
                    })
                  );
                  continue;
                }

                if (chunk.model) model = chunk.model;
                if (chunk.usage) usage = chunk.usage;

                const choice = chunk.choices?.[0];
                if (!choice) continue;

                const delta = choice.delta || {};
                const finishReason = choice.finish_reason;

                if (
                  !delta.reasoning_content &&
                  !delta.content &&
                  !delta.tool_calls &&
                  !finishReason &&
                  Object.keys(delta).length > 0
                ) {
                  reportUnknown(`choice-delta:${Object.keys(delta).sort().join(",")}`);
                }

                if (delta.reasoning_content) {
                  if (!hasStarted) sendMessageStart();

                  if (primaryBlockType !== "thinking") {
                    closePrimaryBlock();
                    const idx = allocBlockIndex();
                    sendLine(thinkingBlockStartEvent(idx));
                    primaryBlockIndex = idx;
                    primaryBlockType = "thinking";
                  }

                  sendLine(thinkingDeltaEvent(primaryBlockIndex, delta.reasoning_content));
                }

                if (delta.content) {
                  if (!hasStarted) sendMessageStart();

                  if (primaryBlockType === "thinking") {
                    sendLine(signatureDeltaEvent(primaryBlockIndex, Date.now().toString()));
                    closePrimaryBlock();
                  }

                  if (primaryBlockType !== "text") {
                    closePrimaryBlock();
                    const idx = allocBlockIndex();
                    sendLine(textBlockStartEvent(idx));
                    primaryBlockIndex = idx;
                    primaryBlockType = "text";
                  }

                  sendLine(textDeltaEvent(primaryBlockIndex, delta.content));
                }

                if (delta.tool_calls) {
                  if (!hasStarted) sendMessageStart();

                  if (primaryBlockType === "thinking") {
                    sendLine(signatureDeltaEvent(primaryBlockIndex, Date.now().toString()));
                    closePrimaryBlock();
                  }

                  for (const tc of delta.tool_calls) {
                    const tcIndex = tc.index ?? 0;

                    if (!toolCallBlocks.has(tcIndex) && tc.id && tc.function?.name) {
                      const idx = allocBlockIndex();
                      sendLine(toolUseBlockStartEvent(idx, tc.id, tc.function.name));
                      toolCallBlocks.set(tcIndex, idx);
                    }

                    if (tc.function?.arguments) {
                      const blockIdx = toolCallBlocks.get(tcIndex);
                      if (blockIdx !== undefined) {
                        sendLine(inputJsonDeltaEvent(blockIdx, tc.function.arguments));
                      }
                    }
                  }
                }

                if (finishReason && !isFinished) {
                  emitFinish(finishReason, usage);
                  break;
                }
              } catch {
                // Skip unparseable lines
              }
            }
          }

          if (!isFinished) {
            emitFinish("stop", usage);
          }
        } catch (e) {
          if (!isClosed) {
            try {
              controller.error(e);
            } catch {}
          }
        } finally {
          try {
            reader.releaseLock();
          } catch {}
          if (!isClosed) {
            try {
              controller.close();
            } catch {}
          }
        }
      },
      cancel() {
        try {
          reader.cancel();
        } catch {}
      },
    });
  }

  private convertOpenAIResponseToAnthropic(
    data: any,
    injected: string[] = [],
    remap: Record<string, string> = {},
  ): any {
    const choice = data.choices?.[0];
    if (!choice) {
      return {
        id: `msg_${Date.now()}`,
        type: "message",
        role: "assistant",
        model: data.model || "unknown",
        content: [],
        stop_reason: "end_turn",
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0, cache_read_input_tokens: 0 },
      };
    }

    const content: any[] = [];
    const msg = choice.message || {};

    if (msg.reasoning_content) {
      content.push({
        type: "thinking",
        thinking: msg.reasoning_content,
        signature: Date.now().toString(),
      });
    }

    if (msg.content) {
      content.push({ type: "text", text: msg.content });
    }

    if (msg.tool_calls) {
      for (const tc of msg.tool_calls) {
        let input: any = {};
        try {
          input = JSON.parse(tc.function.arguments || "{}");
        } catch {
          input = {};
        }
        content.push({
          type: "tool_use",
          id: tc.id,
          name: tc.function.name,
          input,
        });
      }
    }

    if (msg.annotations) {
      for (const ann of msg.annotations) {
        if (ann.url_citation) {
          const id = `srvtoolu_${Date.now()}`;
          content.push({
            type: "server_tool_use",
            id,
            name: "web_search",
            input: { query: "" },
          });
          content.push({
            type: "web_search_tool_result",
            tool_use_id: id,
            content: [
              {
                type: "web_search_result",
                url: ann.url_citation.url,
                title: ann.url_citation.title,
              },
            ],
          });
        }
      }
    }

    this.cloakInjectedToolUse(content, injected, remap);
    // Text-protocol fences may also appear in chat text; surface them.
    this.applyZenTextProtocol(content);
    this.cloakInjectedToolUse(content, injected, remap);

    const stopReasonMap: Record<string, string> = {
      stop: "end_turn",
      length: "max_tokens",
      tool_calls: "tool_use",
      content_filter: "stop_sequence",
    };
    const mappedStop = stopReasonMap[choice.finish_reason] || "end_turn";
    const stop_reason =
      mappedStop === "tool_use" && !content.some((b: any) => b?.type === "tool_use")
        ? "end_turn"
        : mappedStop;

    return {
      id: `msg_${Date.now()}`,
      type: "message",
      role: "assistant",
      model: data.model || "unknown",
      content,
      stop_reason,
      stop_sequence: null,
      usage: {
        input_tokens:
          (data.usage?.prompt_tokens || 0) -
          (data.usage?.prompt_tokens_details?.cached_tokens || 0),
        output_tokens: data.usage?.completion_tokens || 0,
        cache_read_input_tokens: data.usage?.prompt_tokens_details?.cached_tokens || 0,
      },
    };
  }

  // Force Responses-format for Muse Spark (docs/zen + docs/go: muse-spark* only on /responses)
  // This keeps Muse Spark working via CCR even when provider advertises chat/completions.

  private cleanJsonSchema(schema: Record<string, any>): Record<string, any> {
    if (!schema || typeof schema !== "object") {
      return schema;
    }
    if (Array.isArray(schema)) {
      return (schema as any[]).map((item) =>
        typeof item === "object" && item !== null
          ? this.cleanJsonSchema(item as Record<string, any>)
          : item,
      ) as any;
    }

    const cleaned: Record<string, any> = {};

    for (const [key, value] of Object.entries(schema)) {
      if (key === "$ref" || key === "$schema" || key === "$id") {
        continue;
      }
      if (key === "definitions" || key === "$defs") {
        continue;
      }

      if (["required", "enum", "anyOf", "oneOf", "allOf"].includes(key)) {
        if (value === undefined || value === null) {
          continue;
        }
        if (Array.isArray(value)) {
          cleaned[key] = value.map((item) =>
            typeof item === "object" && item !== null
              ? this.cleanJsonSchema(item)
              : item,
          );
        } else if (typeof value === "string") {
          cleaned[key] = [value];
        } else if (typeof value === "object") {
          cleaned[key] = Object.values(value).map((item) =>
            typeof item === "object" && item !== null
              ? this.cleanJsonSchema(item)
              : item,
          );
        }
        continue;
      }

      if (typeof value === "object" && value !== null) {
        cleaned[key] = this.cleanJsonSchema(value);
      } else {
        cleaned[key] = value;
      }
    }

    return cleaned;
  }
}
