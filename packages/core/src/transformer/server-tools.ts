import type { UnifiedTool } from "@ccr/core/types/llm";

/**
 * Root-cause fix for malformed upstream tools.
 *
 * The unified request model used to assume every tool is a function tool, so
 * Anthropic server-side tools (`web_search_*`, `code_execution_*`, ...) were
 * flattened into `{type: "function", function: {...}}` entries with missing
 * schemas. Upstreams reject those (e.g. Zen: `tools[0] did not match any
 * supported type`), failing the entire request.
 *
 * Contract:
 * - Producers preserve server tools with their original `type`.
 * - Every emitter maps through these helpers: function tools keep their
 *   exact legacy shape; server tools become the protocol-native equivalent
 *   or are skipped (with `onSkip`) when the protocol cannot represent them.
 *   Skipping one exotic tool degrades gracefully; emitting a malformed entry
 *   400s the whole request.
 */

export function isFunctionTool(tool: any): tool is { type: "function"; function: any } {
  if (!tool || typeof tool !== "object") return false;
  if (tool.type === "function") return true;
  // Legacy shape without an explicit type but with a function payload.
  if ((tool.type === undefined || tool.type === "custom") && tool.function != null) {
    return true;
  }
  return false;
}

export function serverToolKind(tool: any): string | undefined {
  if (!tool || typeof tool !== "object" || isFunctionTool(tool)) return undefined;
  return typeof tool.type === "string" ? tool.type : undefined;
}

/**
 * True for Anthropic server web-search tools in any shape we preserve or see:
 * an explicit `web_search*` type, or a function-shaped entry named web_search
 * WITHOUT parameters (the legacy flattened form). A real function that
 * happens to be named web_search *with* a schema stays a function.
 */
export function isWebSearchTool(tool: any): boolean {
  if (!tool || typeof tool !== "object") return false;
  const t = tool.type;
  if (typeof t === "string" && t !== "function" && t !== "custom") {
    return /^web_search(_|$)/i.test(t);
  }
  const fn = t === "function" || t === undefined || t === "custom" ? tool.function : undefined;
  const params = fn?.parameters ?? tool.input_schema ?? tool.parameters;
  const name = typeof fn?.name === "string" ? fn.name : typeof tool.name === "string" ? tool.name : undefined;
  return (
    typeof name === "string" &&
    /^web_search$/i.test(name) &&
    (params === undefined || params === null)
  );
}

/** Map one unified tool to an OpenAI Responses `tools[]` entry, or null to skip. */
export function toResponsesTool(tool: UnifiedTool | any): any | null {
  if (isWebSearchTool(tool)) {
    // Responses has a native server-side web search tool. Optional
    // provider-specific options (max_uses, filters) have no equivalent here.
    return { type: "web_search" };
  }
  if (isFunctionTool(tool)) {
    const fn = tool.function;
    if (!fn || typeof fn.name !== "string" || !fn.name) return null;
    return {
      type: "function",
      name: fn.name,
      description: fn.description,
      parameters: fn.parameters,
    };
  }
  return null;
}

/** Map one unified tool to an OpenAI chat-completions `tools[]` entry, or null to skip. */
export function toChatTool(tool: UnifiedTool | any): any | null {
  if (isFunctionTool(tool)) {
    return tool;
  }
  // Chat completions has no server-side tool types; anything else would be
  // rejected upstream, so drop it rather than emit a malformed entry.
  return null;
}

/** Map one unified tool to an Anthropic Messages `tools[]` entry, or null to skip. */
export function toClaudeTool(tool: UnifiedTool | any): any | null {
  if (isFunctionTool(tool)) {
    return {
      name: tool.function.name,
      description: tool.function.description,
      input_schema: tool.function.parameters,
    };
  }
  // Server tools originated on the Anthropic surface: pass through verbatim.
  if (serverToolKind(tool)) {
    const { ...rest } = tool;
    return rest;
  }
  return null;
}

function dropReason(tool: any): string {
  const kind = serverToolKind(tool) || typeof tool?.type === "string" ? tool?.type : typeof tool;
  const name = tool?.function?.name || tool?.name;
  return `unsupported tool kind '${kind}'${name ? ` (name '${name}')` : ""}`;
}

/** Map + filter a tool list for Responses, warning on skipped entries. */
export function mapResponsesTools(
  tools: UnifiedTool[] | undefined,
  onSkip?: (message: string) => void,
): any[] | undefined {
  if (!tools?.length) return undefined;
  const out: any[] = [];
  for (const tool of tools) {
    const mapped = toResponsesTool(tool);
    if (mapped) out.push(mapped);
    else onSkip?.(`dropping ${dropReason(tool)}: no Responses equivalent`);
  }
  return out.length > 0 ? out : undefined;
}

/** Map + filter a tool list for chat-completions, warning on skipped entries. */
export function mapChatTools(
  tools: UnifiedTool[] | undefined,
  onSkip?: (message: string) => void,
): any[] | undefined {
  if (!tools?.length) return undefined;
  const out: any[] = [];
  for (const tool of tools) {
    const mapped = toChatTool(tool);
    if (mapped) out.push(mapped);
    else onSkip?.(`dropping ${dropReason(tool)}: chat-completions has no server tools`);
  }
  return out.length > 0 ? out : undefined;
}

/** Map + filter a tool list for Anthropic Messages, warning on skipped entries. */
export function mapClaudeTools(
  tools: UnifiedTool[] | undefined,
  onSkip?: (message: string) => void,
): any[] | undefined {
  if (!tools?.length) return undefined;
  const out: any[] = [];
  for (const tool of tools) {
    const mapped = toClaudeTool(tool);
    if (mapped) out.push(mapped);
    else onSkip?.(`dropping ${dropReason(tool)}: no Anthropic equivalent`);
  }
  return out.length > 0 ? out : undefined;
}
