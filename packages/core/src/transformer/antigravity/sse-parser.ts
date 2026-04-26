import { convertGoogleToAnthropic } from "./response-converter";

export interface SSEEvent {
  type: string;
  index?: number;
  delta?: any;
  message?: any;
  content_block?: any;
  usage?: any;
}

export async function* streamSSEResponse(
  response: Response,
  model: string
): AsyncGenerator<SSEEvent, void, unknown> {
  const reader = response.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";
  const messageId = `msg_${Date.now().toString(16)}${Math.random().toString(16).slice(2, 18)}`;

  let hasEmittedStart = false;
  let blockIndex = 0;
  let currentBlockType: "thinking" | "text" | "tool_use" | "image" | null = null;
  let currentThinkingSignature = "";

  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let stopReason: string | null = null;

  const flushSignature = (): SSEEvent | null => {
    if (currentThinkingSignature) {
      const ev: SSEEvent = {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "signature_delta", signature: currentThinkingSignature },
      };
      currentThinkingSignature = "";
      return ev;
    }
    return null;
  };

  const closeBlock = (): SSEEvent | null => {
    if (currentBlockType === null) return null;
    const ev: SSEEvent = { type: "content_block_stop", index: blockIndex };
    blockIndex++;
    currentBlockType = null;
    return ev;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;

      const jsonText = line.slice(5).trim();
      if (!jsonText) continue;

      try {
        const data = JSON.parse(jsonText);
        const innerResponse = data.response || data;

        const usage = innerResponse.usageMetadata;
        if (usage) {
          inputTokens = usage.promptTokenCount || inputTokens;
          outputTokens = usage.candidatesTokenCount || outputTokens;
          cacheReadTokens = usage.cachedContentTokenCount || cacheReadTokens;
        }

        const candidates = innerResponse.candidates || [];
        const firstCandidate = candidates[0] || {};
        const content = firstCandidate.content || {};
        const parts = content.parts || [];

        if (!hasEmittedStart && parts.length > 0) {
          hasEmittedStart = true;
          yield {
            type: "message_start",
            message: {
              id: messageId,
              type: "message",
              role: "assistant",
              content: [],
              model,
              stop_reason: null,
              stop_sequence: null,
              usage: {
                input_tokens: inputTokens - cacheReadTokens,
                output_tokens: 0,
                cache_read_input_tokens: cacheReadTokens,
                cache_creation_input_tokens: 0,
              },
            },
          };
        }

        for (const part of parts) {
          if ((part as any).thought === true) {
            const text = (part as any).text || "";
            const signature = (part as any).thoughtSignature || "";

            if (currentBlockType !== "thinking") {
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
              currentBlockType = "thinking";
              currentThinkingSignature = "";
              yield {
                type: "content_block_start",
                index: blockIndex,
                content_block: { type: "thinking", thinking: "" },
              };
            }

            if (signature && signature.length >= 50) {
              currentThinkingSignature = signature;
            }

            if (text) {
              yield {
                type: "content_block_delta",
                index: blockIndex,
                delta: { type: "thinking_delta", thinking: text },
              };
            }
          } else if ((part as any).text !== undefined) {
            const text = (part as any).text;

            if (currentBlockType !== "text") {
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
              currentBlockType = "text";
              yield {
                type: "content_block_start",
                index: blockIndex,
                content_block: { type: "text", text: "" },
              };
            }

            yield {
              type: "content_block_delta",
              index: blockIndex,
              delta: { type: "text_delta", text },
            };
          } else if ((part as any).functionCall) {
            const fc = (part as any).functionCall;

            if (currentBlockType !== null) {
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
            }

            currentBlockType = "tool_use";
            stopReason = "tool_use";

            const toolId = fc.id || `toolu_${Math.random().toString(36).slice(2, 14)}`;

            yield {
              type: "content_block_start",
              index: blockIndex,
              content_block: {
                type: "tool_use",
                id: toolId,
                name: fc.name,
                input: {},
              },
            };

            yield {
              type: "content_block_delta",
              index: blockIndex,
              delta: {
                type: "input_json_delta",
                partial_json: JSON.stringify(fc.args || {}),
              },
            };
          } else if ((part as any).inlineData) {
            const img = (part as any).inlineData;

            if (currentBlockType !== null) {
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
            }

            yield {
              type: "content_block_start",
              index: blockIndex,
              content_block: {
                type: "image",
                source: {
                  type: "base64",
                  media_type: img.mimeType,
                  data: img.data,
                },
              },
            };
            yield { type: "content_block_stop", index: blockIndex };
            blockIndex++;
            currentBlockType = null;
          }
        }

        if (firstCandidate.finishReason && !stopReason) {
          if (firstCandidate.finishReason === "MAX_TOKENS") {
            stopReason = "max_tokens";
          } else if (firstCandidate.finishReason === "STOP") {
            stopReason = "end_turn";
          }
        }
      } catch {
        continue;
      }
    }
  }

  if (!hasEmittedStart) {
    const messageId2 = `msg_${Date.now().toString(16)}${Math.random().toString(16).slice(2, 18)}`;
    yield {
      type: "message_start",
      message: {
        id: messageId2,
        type: "message",
        role: "assistant",
        content: [],
        model,
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
          cache_read_input_tokens: 0,
          cache_creation_input_tokens: 0,
        },
      },
    };
    yield {
      type: "content_block_start",
      index: 0,
      content_block: { type: "text", text: "" },
    };
    yield {
      type: "content_block_delta",
      index: 0,
      delta: {
        type: "text_delta",
        text: "[No response - please try again]",
      },
    };
    yield { type: "content_block_stop", index: 0 };
    yield {
      type: "message_delta",
      delta: { stop_reason: "end_turn", stop_sequence: null },
      usage: { output_tokens: 0 },
    };
    yield { type: "message_stop" };
    return;
  }

  const closeEv = closeBlock();
  if (closeEv) yield closeEv;
  const sigEv = flushSignature();
  if (sigEv) yield sigEv;

  yield {
    type: "message_delta",
    delta: { stop_reason: stopReason || "end_turn", stop_sequence: null },
    usage: {
      output_tokens: outputTokens,
      cache_read_input_tokens: cacheReadTokens,
      cache_creation_input_tokens: 0,
    },
  };

  yield { type: "message_stop" };
}

export async function accumulateSSEToResponse(response: Response, model: string): Promise<Response> {
  const parts: any[] = [];
  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let stopReason: string | null = null;

  const reader = response.body?.getReader();
  if (!reader) {
    return new Response(JSON.stringify({ error: "No response body" }), {
      status: 500,
      headers: { "content-type": "application/json" },
    });
  }

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data:")) continue;

      const jsonText = line.slice(5).trim();
      if (!jsonText) continue;

      try {
        const data = JSON.parse(jsonText);
        const innerResponse = data.response || data;

        const usage = innerResponse.usageMetadata;
        if (usage) {
          inputTokens = usage.promptTokenCount || inputTokens;
          outputTokens = usage.candidatesTokenCount || outputTokens;
          cacheReadTokens = usage.cachedContentTokenCount || cacheReadTokens;
        }

        const candidates = innerResponse.candidates || [];
        const firstCandidate = candidates[0] || {};
        const content = firstCandidate.content || {};
        const partsArr = content.parts || [];

        for (const part of partsArr) {
          parts.push(part);
        }

        if (firstCandidate.finishReason) {
          if (firstCandidate.finishReason === "MAX_TOKENS") {
            stopReason = "max_tokens";
          } else if (firstCandidate.finishReason === "STOP") {
            stopReason = "end_turn";
          }
        }
      } catch {
        continue;
      }
    }
  }

  if (parts.length === 0) {
    const fallback = convertGoogleToAnthropic(
      {
        response: {
          candidates: [
            {
              content: { parts: [{ text: "" }] },
              finishReason: "STOP",
            },
          ],
        },
      },
      model
    );
    return new Response(JSON.stringify(fallback), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  }

  const googleResponse = {
    response: {
      candidates: [
        {
          content: { parts },
          finishReason: stopReason?.toUpperCase() || "STOP",
          usageMetadata: {
            promptTokenCount: inputTokens,
            candidatesTokenCount: outputTokens,
            cachedContentTokenCount: cacheReadTokens,
          },
        },
      ],
    },
  };

  const anthropic = convertGoogleToAnthropic(googleResponse, model);
  return new Response(JSON.stringify(anthropic), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}

export function sseToResponse(
  response: Response,
  model: string
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const event of streamSSEResponse(response, model)) {
          const eventLine = `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
          controller.enqueue(encoder.encode(eventLine));
        }
        controller.close();
      } catch (err) {
        controller.error(err);
      }
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
    },
  });
}