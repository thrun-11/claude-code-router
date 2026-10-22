import {
  contentBlockStartEvent,
  contentBlockStopEvent,
  inputJsonDeltaEvent,
  messageDeltaEvent,
  messageStartEvent,
  messageStopEvent,
  sseEvent,
  textBlockStartEvent,
  textDeltaEvent,
  thinkingBlockStartEvent,
  thinkingDeltaEvent,
} from "@ccr/core/transformer/anthropic-emitter";
import { isWebSearchTool } from "@ccr/core/transformer/server-tools";

/**
 * Gateway-executed web search fallback for chat-completions upstreams.
 *
 * Chat-completions has no server-side search primitive, so an Anthropic
 * `web_search_*` server tool cannot be forwarded there (see `toChatTool`,
 * which drops it). Instead of dropping, this module lets the gateway act
 * as the tool executor, LiteLLM-`websearch_interception` style:
 *
 *   1. The server tool is swapped for a `ccr_web_search` function tool.
 *   2. When the model calls it, the gateway runs the query against a
 *      configured search-engine chain (free tiers: Brave, Tavily).
 *   3. Results are fed back upstream as `role: "tool"` messages in a
 *      bounded server-side loop, then surfaced downstream as native
 *      `server_tool_use` + `web_search_tool_result` blocks so Anthropic
 *      clients (Claude Code subagents) see the semantics they expect.
 *
 * Native search paths are untouched: Responses `web_search` and Gemini
 * `google_search` grounding keep working exactly as before. This module
 * only fires on the chat path, and only when search engines are configured
 * (env, below). With no keys set, behavior is the legacy silent drop.
 */

export const FALLBACK_FN_NAME = "ccr_web_search";
const DEFAULT_MAX_RESULTS = 5;
const DEFAULT_MAX_TURNS = 5;
const ENGINE_TIMEOUT_MS = 25_000;
const SNIPPET_MAX_CHARS = 500;

export type WebSearchResult = {
  title: string;
  url: string;
  snippet: string;
};

export type SearchEngineSpec = {
  name: string;
  key: string;
};

export type WebSearchFallbackConfig = {
  engines: SearchEngineSpec[];
  maxResults: number;
  maxTurns: number;
};

export type ExecutedSearch = {
  query: string;
  engine: string;
  results: WebSearchResult[];
  /** Gateway-issued `srvtoolu_…` id used in the synthesized transcript. */
  toolUseId: string;
};

/**
 * Env config. `CCR_WEBSEARCH_ENGINES` is a comma list drawn from
 * `brave,tavily` (default order). Returns null when no usable engine key
 * is present — the caller must then keep the legacy drop behavior.
 */
export function readWebSearchFallbackConfig(env: NodeJS.ProcessEnv = process.env): WebSearchFallbackConfig | null {
  const order = (env["CCR_WEBSEARCH_ENGINES"] || "brave,tavily")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  const keys: Record<string, string | undefined> = {
    brave: env["CCR_WEBSEARCH_BRAVE_KEY"]?.trim(),
    tavily: env["CCR_WEBSEARCH_TAVILY_KEY"]?.trim(),
  };
  const engines: SearchEngineSpec[] = [];
  for (const name of order) {
    const key = keys[name];
    if (name === "brave" || name === "tavily") {
      if (key) engines.push({ name, key });
    }
  }
  if (engines.length === 0) return null;
  const maxResults = clampInt(env["CCR_WEBSEARCH_MAX_RESULTS"], DEFAULT_MAX_RESULTS, 1, 10);
  const maxTurns = clampInt(env["CCR_WEBSEARCH_MAX_TURNS"], DEFAULT_MAX_TURNS, 1, 10);
  return { engines, maxResults, maxTurns };
}

function clampInt(raw: string | undefined, fallback: number, min: number, max: number): number {
  const n = Number.parseInt(raw || "", 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.min(max, Math.max(min, n));
}

/**
 * If unified `tools` carries an Anthropic web-search server tool, return
 * its search budget (`max_uses`, default 5). Otherwise null.
 */
export function webSearchFallbackBudget(tools: any[] | undefined): { maxUses: number } | null {
  if (!Array.isArray(tools)) return null;
  for (const tool of tools) {
    if (isWebSearchTool(tool)) {
      const raw = Number(tool?.max_uses ?? tool?.maxUses ?? 5);
      const maxUses = Number.isFinite(raw) ? Math.min(10, Math.max(1, Math.floor(raw))) : 5;
      return { maxUses };
    }
  }
  return null;
}

/** Function-tool stand-in sent upstream on the chat path. */
export function fallbackFunctionTool(): any {
  return {
    type: "function",
    function: {
      name: FALLBACK_FN_NAME,
      description:
        "Search the live web for current or external information. Use when the user asks about recent events, facts beyond training data, or anything requiring up-to-date sources. Results return title, URL and snippet per hit.",
      parameters: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "The web search query.",
          },
        },
        required: ["query"],
        additionalProperties: false,
      },
    },
  };
}

export function isFallbackFnCall(toolCall: any): boolean {
  return toolCall?.function?.name === FALLBACK_FN_NAME;
}

export function fallbackQueryOf(toolCall: any): string | null {
  // Anthropic surface shape (what the loop actually sees).
  const direct = toolCall?.input?.query;
  if (typeof direct === "string" && direct.trim()) return direct.trim();
  // Raw chat-completions shape (unit-test / pre-conversion use).
  let args: any = null;
  try {
    args = JSON.parse(toolCall?.function?.arguments || "{}");
  } catch {
    return null;
  }
  const query = args?.query;
  return typeof query === "string" && query.trim() ? query.trim() : null;
}

// ---------------------------------------------------------------------------
// Engines
// ---------------------------------------------------------------------------

export function parseBraveResponse(json: any): WebSearchResult[] {
  const items = json?.web?.results;
  if (!Array.isArray(items)) throw new Error("unexpected Brave response shape");
  return items
    .filter((r: any) => typeof r?.url === "string" && r.url)
    .map((r: any) => ({
      title: typeof r.title === "string" && r.title ? r.title : r.url,
      url: r.url,
      snippet: typeof r.description === "string" ? r.description.slice(0, SNIPPET_MAX_CHARS) : "",
    }));
}

export function parseTavilyResponse(json: any): WebSearchResult[] {
  const items = json?.results;
  if (!Array.isArray(items)) throw new Error("unexpected Tavily response shape");
  return items
    .filter((r: any) => typeof r?.url === "string" && r.url)
    .map((r: any) => ({
      title: typeof r.title === "string" && r.title ? r.title : r.url,
      url: r.url,
      snippet: typeof r.content === "string" ? r.content.slice(0, SNIPPET_MAX_CHARS) : "",
    }));
}

async function braveWebSearch(query: string, key: string, maxResults: number): Promise<WebSearchResult[]> {
  const url =
    `https://api.search.brave.com/res/v1/web/search?q=${encodeURIComponent(query)}` +
    `&count=${maxResults}&result_filter=web&text_decorations=0&safesearch=moderate`;
  const res = await fetch(url, {
    headers: { Accept: "application/json", "X-Subscription-Token": key },
    signal: AbortSignal.timeout(ENGINE_TIMEOUT_MS),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`brave ${res.status}: ${body.slice(0, 200)}`);
  }
  return parseBraveResponse(await res.json());
}

async function tavilyWebSearch(query: string, key: string, maxResults: number): Promise<WebSearchResult[]> {
  const res = await fetch("https://api.tavily.com/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      api_key: key,
      query,
      search_depth: "basic",
      max_results: maxResults,
      include_answer: false,
      include_raw_content: false,
    }),
    signal: AbortSignal.timeout(ENGINE_TIMEOUT_MS),
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`tavily ${res.status}: ${body.slice(0, 200)}`);
  }
  return parseTavilyResponse(await res.json());
}

const ENGINES: Record<string, typeof braveWebSearch> = {
  brave: braveWebSearch,
  tavily: tavilyWebSearch,
};

/**
 * Run the query against each configured engine in order; first success
 * wins. Throws an aggregated error only when every engine fails.
 */
export async function executeWebSearchChain(
  query: string,
  config: WebSearchFallbackConfig,
  onEngineError?: (engine: string, message: string) => void,
): Promise<{ results: WebSearchResult[]; engine: string }> {
  const failures: string[] = [];
  for (const spec of config.engines) {
    const run = ENGINES[spec.name];
    if (!run) continue;
    try {
      const results = await run(query, spec.key, config.maxResults);
      return { results, engine: spec.name };
    } catch (e: any) {
      const message = e?.name === "TimeoutError" ? "timeout" : String(e?.message || e).slice(0, 200);
      failures.push(`${spec.name}: ${message}`);
      onEngineError?.(spec.name, message);
    }
  }
  throw new Error(`all web search engines failed (${failures.join("; ") || "no engines"})`);
}

// ---------------------------------------------------------------------------
// Transcript synthesis (Anthropic surface)
// ---------------------------------------------------------------------------

export function createExecutedSearch(query: string, engine: string, results: WebSearchResult[]): ExecutedSearch {
  return {
    query,
    engine,
    results,
    toolUseId: `srvtoolu_fallback_${Date.now().toString(36)}${Math.floor(Math.random() * 0xffff).toString(36)}`,
  };
}

/** Executed searches as native citation blocks, in call order. */
export function synthesizeSearchBlocks(searches: ExecutedSearch[]): any[] {
  const blocks: any[] = [];
  for (const s of searches) {
    blocks.push({
      type: "server_tool_use",
      id: s.toolUseId,
      name: "web_search",
      input: { query: s.query },
    });
    blocks.push({
      type: "web_search_tool_result",
      tool_use_id: s.toolUseId,
      content: s.results.map((r) => ({
        type: "web_search_result",
        title: r.title,
        url: r.url,
      })),
    });
  }
  return blocks;
}

/** Chat `role: "tool"` payload handed back upstream after a search. */
export function searchToolResultPayload(search: ExecutedSearch): string {
  return JSON.stringify({
    results: search.results.map((r) => ({ title: r.title, url: r.url, snippet: r.snippet })),
    engine: search.engine,
  });
}

/**
 * Serialize a complete Anthropic message (content blocks + stop_reason +
 * usage) to SSE. Used when downstream streams but the search loop had to
 * buffer upstream turns.
 */
export function anthropicMessageToSSE(message: {
  id: string;
  model: string;
  content: any[];
  stop_reason: string;
  usage: { input_tokens: number; output_tokens: number; cache_read_input_tokens?: number };
}): string {
  const out: string[] = [messageStartEvent(message.id, message.model)];
  message.content.forEach((block: any, index: number) => {
    switch (block?.type) {
      case "thinking":
        out.push(thinkingBlockStartEvent(index));
        if (block.thinking) out.push(thinkingDeltaEvent(index, block.thinking));
        out.push(contentBlockStopEvent(index));
        break;
      case "text":
        out.push(textBlockStartEvent(index));
        if (block.text) out.push(textDeltaEvent(index, block.text));
        out.push(contentBlockStopEvent(index));
        break;
      case "tool_use":
        out.push(contentBlockStartEvent(index, {
          type: "tool_use",
          id: block.id,
          name: block.name,
          input: {},
        }));
        if (block.input && Object.keys(block.input).length > 0) {
          out.push(inputJsonDeltaEvent(index, JSON.stringify(block.input)));
        }
        out.push(contentBlockStopEvent(index));
        break;
      case "server_tool_use":
      case "web_search_tool_result":
        // Server-executed blocks stream atomically, mirroring the Responses
        // citation path: full block on start, no deltas.
        out.push(contentBlockStartEvent(index, block));
        out.push(contentBlockStopEvent(index));
        break;
      default:
        out.push(sseEvent("content_block_start", {
          type: "content_block_start",
          index,
          content_block: { type: "text", text: "" },
        }));
        out.push(contentBlockStopEvent(index));
        break;
    }
  });
  out.push(messageDeltaEvent(message.stop_reason as any, message.usage));
  out.push(messageStopEvent());
  return out.join("");
}
