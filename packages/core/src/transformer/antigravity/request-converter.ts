import {
  UnifiedChatRequest,
} from "@ccr/core/types/llm";
import {
  getModelFamily,
  isThinkingModel,
  GEMINI_MAX_OUTPUT_TOKENS,
} from "./constants";
import {
  convertContentToParts,
  convertRole,
} from "./content-converter";
import { sanitizeSchema, cleanSchema, cleanCacheControl } from "./schema-sanitizer";
import {
  hasGeminiHistory,
  hasUnsignedThinkingBlocks,
  needsThinkingRecovery,
  closeToolLoopForThinking,
  filterUnsignedThinkingBlocks,
  reorderAssistantContent,
  restoreThinkingSignatures,
  clampGeminiThinkingBudget,
} from "./thinking-utils";

export interface GoogleRequest {
  contents: any[];
  generationConfig: any;
  systemInstruction?: any;
  tools?: any[];
  toolConfig?: any;
  sessionId?: string;
}

export function convertAnthropicToGoogle(
  request: UnifiedChatRequest
): GoogleRequest {
  const req = request as any;
  const messages = cleanCacheControl(req.messages || []);

  const { max_tokens, temperature, top_p, top_k, stop_sequences, tools, thinking } = req;
  const modelName = request.model || "";
  const modelFamily = getModelFamily(modelName);
  const isClaudeModel = modelFamily === "claude";
  const isGeminiModel = modelFamily === "gemini";
  const isThinking = isThinkingModel(modelName);

  const googleRequest: GoogleRequest = {
    contents: [],
    generationConfig: {},
  };

  const systemMessages = messages.filter((m) => m.role === "system");
  if (systemMessages.length > 0) {
    const systemParts: any[] = [];
    for (const sysMsg of systemMessages) {
      if (typeof sysMsg.content === "string") {
        if (sysMsg.content.trim()) {
          systemParts.push({ text: sysMsg.content.trim() });
        }
      } else if (Array.isArray(sysMsg.content)) {
        for (const block of sysMsg.content) {
          if (block?.type === "text" && block.text?.trim()) {
            systemParts.push({ text: block.text.trim() });
          }
        }
      }
    }

    if (systemParts.length > 0) {
      const combinedSystemText = systemParts.map(p => p.text).join("\n\n");
      googleRequest.systemInstruction = { parts: [{ text: combinedSystemText }] };
    }
  }

  if (isClaudeModel && isThinking && tools && tools.length > 0) {
    const hint =
      "Interleaved thinking is enabled. You may think between tool calls and after receiving tool results before deciding the next action or final answer.";
    const parts = googleRequest.systemInstruction.parts;
    const lastPart = parts[parts.length - 1];
    if (lastPart?.text) {
      lastPart.text = `${lastPart.text}\n\n${hint}`;
    } else {
      parts.push({ text: hint });
    }
  }

  let processedMessages = messages;

  if (isGeminiModel && isThinking && needsThinkingRecovery(messages)) {
    processedMessages = closeToolLoopForThinking(messages, "gemini");
  }

  const needsClaudeRecovery =
    hasGeminiHistory(messages) || hasUnsignedThinkingBlocks(messages);
  if (
    isClaudeModel &&
    isThinking &&
    needsClaudeRecovery &&
    needsThinkingRecovery(messages)
  ) {
    processedMessages = closeToolLoopForThinking(messages, "claude");
  }

  for (const msg of processedMessages) {
    if (msg.role === "system") continue;

    let msgContent = msg.content;

    if (
      (msg.role === "assistant" || msg.role === "model") &&
      Array.isArray(msgContent)
    ) {
      msgContent = restoreThinkingSignatures(msgContent);
      msgContent = reorderAssistantContent(msgContent);
    }

    const parts = convertContentToParts(msgContent, isClaudeModel, isGeminiModel);

    if (parts.length === 0) {
      parts.push({ text: "." });
    }

    const role = convertRole(msg.role);

    if (role) {
      googleRequest.contents.push({
        role,
        parts,
      });
    }
  }

  if (isClaudeModel) {
    googleRequest.contents = filterUnsignedThinkingBlocks(googleRequest.contents);
  }

  if (max_tokens !== undefined) {
    googleRequest.generationConfig.maxOutputTokens = max_tokens;
  }
  if (temperature !== undefined) {
    googleRequest.generationConfig.temperature = temperature;
  }
  if (top_p !== undefined) {
    googleRequest.generationConfig.topP = top_p;
  }
  if (top_k !== undefined) {
    googleRequest.generationConfig.topK = top_k;
  }
  if (stop_sequences && stop_sequences.length > 0) {
    googleRequest.generationConfig.stopSequences = stop_sequences;
  }

  if (isGeminiModel && !modelName.includes("3.1") && !modelName.includes("2.5") && googleRequest.generationConfig.maxOutputTokens > GEMINI_MAX_OUTPUT_TOKENS) {
    googleRequest.generationConfig.maxOutputTokens = GEMINI_MAX_OUTPUT_TOKENS;
  }

  if (isThinking) {
    if (isClaudeModel) {
      const thinkingBudget = thinking?.budget_tokens || 32000;
      googleRequest.generationConfig.thinkingConfig = {
        includeThoughts: true,
        thinkingBudget: thinkingBudget,
      };

      const currentMaxTokens = googleRequest.generationConfig.maxOutputTokens;
      if (currentMaxTokens && currentMaxTokens <= thinkingBudget) {
        googleRequest.generationConfig.maxOutputTokens = thinkingBudget + 8192;
      }
    } else if (isGeminiModel) {
      let thinkingBudget = clampGeminiThinkingBudget(modelName, thinking?.budget_tokens);
      const maxOutputTokens = googleRequest.generationConfig.maxOutputTokens;
      if (typeof maxOutputTokens === "number" && maxOutputTokens > 0) {
        // Thinking tokens count against maxOutputTokens: without headroom
        // the model burns the whole budget thinking and the answer
        // truncates mid-stream (only a fresh-budget "resume" continues it).
        thinkingBudget = Math.min(thinkingBudget, Math.max(1024, maxOutputTokens - 4096));
      }
      googleRequest.generationConfig.thinkingConfig = {
        includeThoughts: true,
        thinkingBudget,
      };
    }
  }

  if (tools && tools.length > 0) {
    const functionDeclarations = tools.map((tool: any, idx: number) => {
      const name =
        tool.name ||
        tool.function?.name ||
        (tool as any).custom?.name ||
        `tool-${idx}`;
      const description =
        tool.description ||
        tool.function?.description ||
        (tool as any).custom?.description ||
        "";
      const schema =
        tool.input_schema ||
        tool.function?.input_schema ||
        tool.function?.parameters ||
        (tool as any).custom?.input_schema ||
        tool.parameters ||
        { type: "object" };

      let parameters = sanitizeSchema(schema);
      parameters = cleanSchema(parameters);

      return {
        name: String(name).replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64),
        description,
        parameters,
      };
    });

    if (functionDeclarations.length > 0) {
      googleRequest.tools = [{ functionDeclarations }];

      if (isClaudeModel) {
        googleRequest.toolConfig = {
          functionCallingConfig: {
            mode: "AUTO",
          },
        };
      }
    }
  }

  return googleRequest;
}