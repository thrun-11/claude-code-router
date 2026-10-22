import { MIN_SIGNATURE_LENGTH } from "./constants";
import { cacheThinkingSignature } from "./thinking-utils";

export interface AnthropicResponse {
  id: string;
  type: string;
  role: string;
  content: any[];
  model: string;
  stop_reason: string | null;
  stop_sequence: string | null;
  usage: {
    input_tokens: number;
    output_tokens: number;
    cache_read_input_tokens: number;
    cache_creation_input_tokens: number;
  };
}

export function convertGoogleToAnthropic(
  googleResponse: any,
  model: string
): AnthropicResponse {
  const response = googleResponse.response || googleResponse;
  const candidates = response.candidates || [];
  const firstCandidate = candidates[0] || {};
  const content = firstCandidate.content || {};
  const parts = content.parts || [];

  const anthropicContent: any[] = [];
  let hasToolCalls = false;

  for (const part of parts) {
    if (part.text !== undefined) {
      if (part.thought === true) {
        const signature = part.thoughtSignature || "";
        if (signature && signature.length >= MIN_SIGNATURE_LENGTH) {
          cacheThinkingSignature(signature, part.text);
        }
        anthropicContent.push({
          type: "thinking",
          thinking: part.text,
          signature,
        });
      } else {
        anthropicContent.push({
          type: "text",
          text: part.text,
        });
      }
    } else if (part.functionCall) {
      const toolId =
        part.functionCall.id ||
        `toolu_${crypto.randomUUID?.() || Math.random().toString(36).slice(2, 26)}`;
      const toolUseBlock: any = {
        type: "tool_use",
        id: toolId,
        name: part.functionCall.name,
        input: part.functionCall.args || {},
      };

      if (part.thoughtSignature && part.thoughtSignature.length >= MIN_SIGNATURE_LENGTH) {
        toolUseBlock.thoughtSignature = part.thoughtSignature;
      }

      anthropicContent.push(toolUseBlock);
      hasToolCalls = true;
    } else if (part.inlineData) {
      anthropicContent.push({
        type: "image",
        source: {
          type: "base64",
          media_type: part.inlineData.mimeType,
          data: part.inlineData.data,
        },
      });
    } else if (part.functionResponse) {
      const responseObj = part.functionResponse.response;
      let toolContent = "";
      if (typeof responseObj === "string") {
        toolContent = responseObj;
      } else if (responseObj?.text?.text) {
        toolContent = responseObj.text.text;
      } else if (responseObj?.content?.json) {
        toolContent = responseObj.content.json;
      } else {
        toolContent = JSON.stringify(responseObj);
      }
      anthropicContent.push({
        type: "tool_result",
        tool_use_id: part.functionResponse.id,
        content: toolContent,
      });
    }
  }

  appendGroundingCitations(firstCandidate, anthropicContent);

  const finishReason = firstCandidate.finishReason;
  // Tool calls win over STOP: Gemini reports finishReason STOP even when
  // the turn produced functionCall parts, and the client only runs tools
  // when stop_reason is tool_use.
  let stopReason = "end_turn";
  if (hasToolCalls) {
    stopReason = "tool_use";
  } else if (finishReason === "MAX_TOKENS") {
    stopReason = "max_tokens";
  } else if (finishReason === "STOP") {
    stopReason = "end_turn";
  }

  const usageMetadata = response.usageMetadata || {};
  const promptTokens = usageMetadata.promptTokenCount || 0;
  const cachedTokens = usageMetadata.cachedContentTokenCount || 0;

  return {
    id: `msg_${randomId()}`,
    type: "message",
    role: "assistant",
    content: anthropicContent.length > 0 ? anthropicContent : [{ type: "text", text: "" }],
    model,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: promptTokens - cachedTokens,
      output_tokens: usageMetadata.candidatesTokenCount || 0,
      cache_read_input_tokens: cachedTokens,
      cache_creation_input_tokens: 0,
    },
  };
}

function randomId(): string {
  return Math.random().toString(36).slice(2, 18) + Math.random().toString(36).slice(2, 18);
}

/**
 * Translates Gemini grounding metadata into Anthropic citation blocks.
 * `webSearchQueries` become server_tool_use blocks (one per query);
 * `groundingChunks[].web` become a single aggregated web_search_tool_result
 * (deduplicated by URL). Emits nothing when there is no grounding data, and
 * never flips stop_reason (server tools execute provider-side).
 */
export function appendGroundingCitations(candidate: any, anthropicContent: any[]): void {
  const grounding = candidate?.groundingMetadata;
  if (!grounding || typeof grounding !== "object") return;
  const queries: string[] = [];
  const seenQueries = new Set<string>();
  for (const query of Array.isArray(grounding.webSearchQueries) ? grounding.webSearchQueries : []) {
    if (typeof query !== "string" || !query.trim() || seenQueries.has(query)) continue;
    seenQueries.add(query);
    queries.push(query);
  }
  const seen = new Set<string>();
  const sources: Array<{ title: string; url: string }> = [];
  for (const chunk of Array.isArray(grounding.groundingChunks) ? grounding.groundingChunks : []) {
    const web = chunk?.web;
    if (!web || typeof web.uri !== "string" || !web.uri) continue;
    if (seen.has(web.uri)) continue;
    seen.add(web.uri);
    sources.push({
      title: typeof web.title === "string" && web.title ? web.title : web.uri,
      url: web.uri,
    });
  }
  if (queries.length === 0 && sources.length === 0) return;
  const useIds: string[] = [];
  for (const query of queries) {
    const id = `srvtoolu_grounding_${randomId()}`;
    useIds.push(id);
    anthropicContent.push({
      type: "server_tool_use",
      id,
      name: "web_search",
      input: { query },
    });
  }
  if (sources.length > 0) {
    anthropicContent.push({
      type: "web_search_tool_result",
      tool_use_id: useIds[0] ?? `srvtoolu_${Date.now()}`,
      content: sources.map((source) => ({
        type: "web_search_result",
        title: source.title,
        url: source.url,
      })),
    });
  }
}