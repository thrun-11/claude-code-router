import { UnifiedChatRequest, LLMProvider } from "@/types/llm";
import { Transformer, TransformerContext } from "@/types/transformer";
import { ProxyAgent } from "undici";

const CHAT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions";

export class OpencodeGoTransformer implements Transformer {
  name = "opencode-go";
  logger?: any;

  private isChatCompletionsModel(model: string): boolean {
    return (
      model.startsWith("deepseek-") ||
      model.startsWith("kimi-") ||
      model.startsWith("glm-") ||
      model.startsWith("mimo-") ||
      model.startsWith("qwen-") ||
      model.startsWith("minimax-")
    );
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    context: TransformerContext,
  ): Promise<any> {
    if (request.tools) {
      request.tools = request.tools.map((tool) => {
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
    request: UnifiedChatRequest,
    config: any,
    provider: LLMProvider,
    context: TransformerContext,
  ): Promise<Response> {
    const model = request.model;

    if (this.isChatCompletionsModel(model)) {
      const body = this.unifiedToOpenAI(request, model);
      return this.fetchEndpoint(body, CHAT_ENDPOINT, config, request.stream ?? false, context);
    }

    const body = this.unifiedToOpenAI(request, model);
    return this.fetchEndpoint(body, CHAT_ENDPOINT, config, request.stream ?? false, context);
  }

  private async fetchEndpoint(
    body: any,
    url: string,
    config: any,
    isStream: boolean,
    context: TransformerContext,
  ): Promise<Response> {
    const reqLog = (context.req as any)?.log;
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "User-Agent": "opencode/2.1.131",
      ...(config.headers || {}),
    };

    if (config.api_key) {
      headers["Authorization"] = `Bearer ${config.api_key}`;
    }

    const reqHeaders = context.req?.headers || {};

    let opencodeSession = reqHeaders["x-opencode-session"];
    if (!opencodeSession) {
      const rawBody = (context.req as any)?.body;
      const rawMeta = rawBody?.metadata?.user_id;
      if (typeof rawMeta === "string") {
        try {
          const parsed = JSON.parse(rawMeta);
          opencodeSession = parsed.session_id || parsed.device_id;
        } catch {}
      }
      if (!opencodeSession) {
        opencodeSession = (context.req as any)?.id || `ccr_${Date.now()}`;
      }
    }
    headers["x-opencode-session"] = opencodeSession;

    if (reqHeaders["x-opencode-project"]) {
      headers["x-opencode-project"] = reqHeaders["x-opencode-project"];
    }
    headers["x-opencode-request"] = opencodeSession;
    headers["x-opencode-client"] = reqHeaders["x-opencode-client"] || "cli";

    const fetchOptions: RequestInit & { dispatcher?: any } = {
      method: "POST",
      headers,
      body: JSON.stringify(body),
    };

    if (config.httpsProxy) {
      fetchOptions.dispatcher = new ProxyAgent(new URL(config.httpsProxy).toString());
    }

    const response = await fetch(url, fetchOptions);

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

    if (isStream) {
      if (!response.body) {
        return response;
      }
      const convertedStream = this.convertOpenAIStreamToAnthropic(response.body, context);
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

    const data = await response.json();
    const anthropicResponse = this.convertOpenAIResponseToAnthropic(data);
    return new Response(JSON.stringify(anthropicResponse), {
      status: response.status,
      statusText: response.statusText,
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
    if (unified.tools?.length) body.tools = unified.tools;
    if (unified.tool_choice) body.tool_choice = unified.tool_choice;

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

  private convertOpenAIStreamToAnthropic(
    stream: ReadableStream<Uint8Array>,
    context: TransformerContext,
  ): ReadableStream {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();

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

        const sendEvent = (event: string, data: any) => {
          safeEnqueue(
            encoder.encode(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`),
          );
        };

        const closePrimaryBlock = () => {
          if (primaryBlockIndex >= 0) {
            sendEvent("content_block_stop", {
              type: "content_block_stop",
              index: primaryBlockIndex,
            });
            primaryBlockIndex = -1;
            primaryBlockType = null;
          }
        };

        const closeAllToolCallBlocks = () => {
          for (const [, blockIdx] of toolCallBlocks) {
            sendEvent("content_block_stop", {
              type: "content_block_stop",
              index: blockIdx,
            });
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
          sendEvent("message_start", {
            type: "message_start",
            message: {
              id: messageId,
              type: "message",
              role: "assistant",
              content: [],
              model,
              stop_reason: null,
              stop_sequence: null,
              usage: { input_tokens: 0, output_tokens: 0 },
            },
          });
          hasStarted = true;
        };

        const emitFinish = (stopReason: string, usageData?: any) => {
          if (isFinished) return;
          isFinished = true;

          if (primaryBlockType === "thinking") {
            sendEvent("content_block_delta", {
              type: "content_block_delta",
              index: primaryBlockIndex,
              delta: {
                type: "signature_delta",
                signature: Date.now().toString(),
              },
            });
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

          sendEvent("message_delta", {
            type: "message_delta",
            delta: { stop_reason: anthropicStop, stop_sequence: null },
            usage: {
              input_tokens:
                (usageData?.prompt_tokens || 0) -
                (usageData?.prompt_tokens_details?.cached_tokens || 0),
              output_tokens: usageData?.completion_tokens || 0,
              cache_read_input_tokens:
                usageData?.prompt_tokens_details?.cached_tokens || 0,
            },
          });
          sendEvent("message_stop", { type: "message_stop" });

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
                  sendEvent("error", {
                    type: "error",
                    message: {
                      type: "api_error",
                      message: JSON.stringify(chunk.error),
                    },
                  });
                  continue;
                }

                if (chunk.model) model = chunk.model;
                if (chunk.usage) usage = chunk.usage;

                const choice = chunk.choices?.[0];
                if (!choice) continue;

                const delta = choice.delta || {};
                const finishReason = choice.finish_reason;

                if (delta.reasoning_content) {
                  if (!hasStarted) sendMessageStart();

                  if (primaryBlockType !== "thinking") {
                    closePrimaryBlock();
                    const idx = allocBlockIndex();
                    sendEvent("content_block_start", {
                      type: "content_block_start",
                      index: idx,
                      content_block: { type: "thinking", thinking: "" },
                    });
                    primaryBlockIndex = idx;
                    primaryBlockType = "thinking";
                  }

                  sendEvent("content_block_delta", {
                    type: "content_block_delta",
                    index: primaryBlockIndex,
                    delta: {
                      type: "thinking_delta",
                      thinking: delta.reasoning_content,
                    },
                  });
                }

                if (delta.content) {
                  if (!hasStarted) sendMessageStart();

                  if (primaryBlockType === "thinking") {
                    sendEvent("content_block_delta", {
                      type: "content_block_delta",
                      index: primaryBlockIndex,
                      delta: {
                        type: "signature_delta",
                        signature: Date.now().toString(),
                      },
                    });
                    closePrimaryBlock();
                  }

                  if (primaryBlockType !== "text") {
                    closePrimaryBlock();
                    const idx = allocBlockIndex();
                    sendEvent("content_block_start", {
                      type: "content_block_start",
                      index: idx,
                      content_block: { type: "text", text: "" },
                    });
                    primaryBlockIndex = idx;
                    primaryBlockType = "text";
                  }

                  sendEvent("content_block_delta", {
                    type: "content_block_delta",
                    index: primaryBlockIndex,
                    delta: { type: "text_delta", text: delta.content },
                  });
                }

                if (delta.tool_calls) {
                  if (!hasStarted) sendMessageStart();

                  if (primaryBlockType === "thinking") {
                    sendEvent("content_block_delta", {
                      type: "content_block_delta",
                      index: primaryBlockIndex,
                      delta: {
                        type: "signature_delta",
                        signature: Date.now().toString(),
                      },
                    });
                    closePrimaryBlock();
                  }

                  for (const tc of delta.tool_calls) {
                    const tcIndex = tc.index ?? 0;

                    if (!toolCallBlocks.has(tcIndex) && tc.id && tc.function?.name) {
                      const idx = allocBlockIndex();
                      sendEvent("content_block_start", {
                        type: "content_block_start",
                        index: idx,
                        content_block: {
                          type: "tool_use",
                          id: tc.id,
                          name: tc.function.name,
                          input: {},
                        },
                      });
                      toolCallBlocks.set(tcIndex, idx);
                    }

                    if (tc.function?.arguments) {
                      const blockIdx = toolCallBlocks.get(tcIndex);
                      if (blockIdx !== undefined) {
                        sendEvent("content_block_delta", {
                          type: "content_block_delta",
                          index: blockIdx,
                          delta: {
                            type: "input_json_delta",
                            partial_json: tc.function.arguments,
                          },
                        });
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

  private convertOpenAIResponseToAnthropic(data: any): any {
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

    const stopReasonMap: Record<string, string> = {
      stop: "end_turn",
      length: "max_tokens",
      tool_calls: "tool_use",
      content_filter: "stop_sequence",
    };

    return {
      id: `msg_${Date.now()}`,
      type: "message",
      role: "assistant",
      model: data.model || "unknown",
      content,
      stop_reason: stopReasonMap[choice.finish_reason] || "end_turn",
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

  private cleanJsonSchema(schema: Record<string, any>): Record<string, any> {
    if (!schema || typeof schema !== "object") {
      return schema;
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
