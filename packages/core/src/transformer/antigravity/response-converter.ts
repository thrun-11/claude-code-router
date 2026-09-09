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