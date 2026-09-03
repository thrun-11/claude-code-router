type UpstreamRequest = {
  body: unknown;
  headers: Record<string, string>;
  method?: string;
  url: string;
};

type ProviderPluginRequestInput = {
  request?: {
    headers?: Record<string, string | string[] | undefined>;
  };
  upstreamRequest: UpstreamRequest;
};

const MIN_OUTPUT_TOKENS = 256;

function clampBody(body: unknown): unknown {
  if (!body || typeof body !== "object" || Array.isArray(body)) return body;
  const record = body as Record<string, unknown>;
  let changed = false;
  for (const key of ["max_output_tokens", "max_tokens"]) {
    const v = record[key];
    if (typeof v === "number" && Number.isFinite(v) && v < MIN_OUTPUT_TOKENS) {
      delete record[key];
      changed = true;
    }
  }
  return changed ? { ...record } : body;
}

export function createGatewayPlugin() {
  return {
    providerHooks: [
      {
        key: "ccr-upstream-min-output-tokens",
        transformRequest(input: ProviderPluginRequestInput) {
          const url = input?.upstreamRequest?.url || "";
          let pathname = url;
          try {
            pathname = new URL(url).pathname;
          } catch {}
          if (!pathname.endsWith("/responses")) {
            return { ok: true as const, value: input.upstreamRequest };
          }
          const body = input.upstreamRequest.body;
          let parsed: unknown;
          const wasString = typeof body === "string";
          try {
            parsed = wasString ? JSON.parse(body as string) : body;
          } catch {
            return { ok: true as const, value: input.upstreamRequest };
          }
          const clamped = clampBody(parsed);
          if (clamped === parsed && !wasString) {
            return {
              ok: true as const,
              value: { ...input.upstreamRequest, body: clamped },
            };
          }
          return {
            ok: true as const,
            value: {
              ...input.upstreamRequest,
              body: wasString ? JSON.stringify(clamped) : clamped,
            },
          };
        },
      },
    ],
  };
}
