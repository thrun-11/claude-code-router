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

const CHAT_ENDPOINT = "https://opencode.ai/zen/go/v1/chat/completions";
const FREE_CHAT_ENDPOINT = "https://opencode.ai/zen/v1/chat/completions";
const GO_RESPONSES_ENDPOINT = "https://opencode.ai/zen/go/v1/responses";
const ZEN_RESPONSES_ENDPOINT = "https://opencode.ai/zen/v1/responses";

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
    _provider: LLMProvider,
    context: TransformerContext,
  ): Promise<Response> {
    const model = request.model;
    // Muse Spark models use Responses API per docs/zen and docs/go
    if (model.startsWith("muse-spark")) {
      const isFree = this.isFreeModel(model);
      const endpoint = isFree ? ZEN_RESPONSES_ENDPOINT : GO_RESPONSES_ENDPOINT;
      const body = this.unifiedToResponses(request);
      return this.fetchEndpoint(body, endpoint, config, request.stream ?? false, context);
    }
    const endpoint = this.chatEndpointForModel(model);

    if (this.isChatCompletionsModel(model)) {
      const body = this.unifiedToOpenAI(request, model);
      return this.fetchEndpoint(body, endpoint, config, request.stream ?? false, context);
    }

    const body = this.unifiedToOpenAI(request, model);
    return this.fetchEndpoint(body, endpoint, config, request.stream ?? false, context);
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

    const isFree = this.isFreeModel(body.model);
    const endpointApiKey = isFree ? config.api_key_zen : config.api_key_go;
    if (endpointApiKey) {
      headers["Authorization"] = `Bearer ${endpointApiKey}`;
    } else if (config.api_key) {
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

    let response = await fetch(url, fetchOptions);

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

    const isResponses = url.includes("/responses");
    if (isStream) {
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

    const data = await response.json();
    if (isResponses) {
      const anthropicResponse = this.convertResponsesToAnthropic(data);
      return new Response(JSON.stringify(anthropicResponse), {
        status: response.status,
        statusText: response.statusText,
        headers: { "Content-Type": "application/json" },
      });
    }
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
    if (unified.tools?.length) {
      body.tools = unified.tools.map((tool) => ({
        type: "function",
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
      }));
    }
    if (unified.tool_choice) body.tool_choice = unified.tool_choice;
    if (unified.temperature !== undefined) body.temperature = unified.temperature;
    return body;
  }

  private convertResponsesToAnthropic(data: any): any {
    const { content, hasToolUse } = responsesOutputToContent(data.output);
    // fallback to direct output_text if no structured output
    if (content.length === 0 && data.output_text) {
      content.push({ type: "text", text: data.output_text });
    }
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
