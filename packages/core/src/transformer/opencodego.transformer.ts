import { UnifiedChatRequest, UnifiedMessage, UnifiedTool } from "@/types/llm";
import { Transformer } from "../types/transformer";

export class OpencodeGoTransformer implements Transformer {
  name = "opencode-go";
  endPoint = "/zen/go/v1/chat/completions";

  private readonly MINIMAX_MODELS = ["minimax-m2.5", "minimax-m2.7"];

  async transformRequestIn(
    request: UnifiedChatRequest
  ): Promise<UnifiedChatRequest> {
    (this as any).endPoint = this.getEndpoint(request.model);

    if (request.tools) {
      request.tools = request.tools.map((tool) => {
        if (tool.function?.parameters) {
          const cleanedParams = this.cleanJsonSchema(
            tool.function.parameters as Record<string, any>
          );
          const cleanedTool: UnifiedTool = {
            type: "function",
            function: {
              name: tool.function.name,
              description: tool.function.description,
              parameters: cleanedParams as {
                type: "object";
                properties: Record<string, any>;
                required?: string[];
                additionalProperties?: boolean;
                $schema?: string;
              },
            },
          };
          return cleanedTool;
        }
        return tool;
      });

      if (this.needsPlaceholderReasoning(request.model)) {
        request = this.addPlaceholderReasoning(request);
      }
    }

    if (this.isDeepSeekModel(request.model)) {
      request = this.handleDeepSeekThinking(request);
    }

    return request;
  }

  private getEndpoint(model: string): string {
    if (this.MINIMAX_MODELS.includes(model)) {
      return "/zen/go/v1/messages";
    }
    return "/zen/go/v1/chat/completions";
  }

  private isDeepSeekModel(modelId: string): boolean {
    return modelId.startsWith("deepseek-");
  }

  private needsPlaceholderReasoning(modelId: string): boolean {
    return modelId.startsWith("kimi-");
  }

  private hasThinkingInHistory(messages: UnifiedMessage[]): boolean {
    for (const msg of messages) {
      if (msg.role !== "assistant") continue;
      if ((msg as any).thinking && (msg as any).thinking.content) {
        return true;
      }
      const content = msg.content;
      if (typeof content === "string") {
        continue;
      }
      if (Array.isArray(content)) {
        for (const block of content) {
          if ((block as any).type === "thinking") return true;
        }
      }
    }
    return false;
  }

  private addPlaceholderReasoning(
    request: UnifiedChatRequest
  ): UnifiedChatRequest {
    const updatedMessages = request.messages.map((msg) => {
      if (msg.role === "assistant" && msg.tool_calls && msg.tool_calls.length > 0) {
        return {
          ...msg,
          thinking: msg.thinking || { content: " " },
        } as UnifiedMessage;
      }
      return msg;
    });

    return {
      ...request,
      messages: updatedMessages,
    };
  }

  private handleDeepSeekThinking(
    request: UnifiedChatRequest
  ): UnifiedChatRequest {
    const hasThinking = this.hasThinkingInHistory(request.messages);

    if (!hasThinking) {
      const updated: UnifiedChatRequest = { ...request };
      (updated as any).thinking = { type: "disabled" };
      return updated;
    }

    const updated: UnifiedChatRequest = {
      ...request,
      messages: request.messages.map((msg) => {
        if (msg.role === "assistant") {
          const hasToolCalls = msg.tool_calls && msg.tool_calls.length > 0;
          const existingThinking = (msg as any).thinking?.content;

          if (hasToolCalls || !existingThinking) {
            return {
              ...msg,
              reasoning_content: " ",
            };
          }
        }
        return msg;
      }),
    };

    (updated as any).reasoning_effort = "high";
    (updated as any).thinking = { type: "enabled" };

    return updated;
  }

  private cleanJsonSchema(schema: Record<string, any>): Record<string, any> {
    if (!schema || typeof schema !== "object") {
      return schema;
    }

    const cleaned: Record<string, any> = {};

    for (const [key, value] of Object.entries(schema)) {
      if (key === "$ref") {
        continue;
      }
      if (key === "$schema" || key === "$id") {
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
              : item
          );
        } else if (typeof value === "string") {
          cleaned[key] = [value];
        } else if (typeof value === "object") {
          cleaned[key] = Object.values(value).map((item) =>
            typeof item === "object" && item !== null
              ? this.cleanJsonSchema(item)
              : item
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