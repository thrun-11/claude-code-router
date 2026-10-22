import { MIN_SIGNATURE_LENGTH } from "./constants";
import { convertGoogleToAnthropic } from "./response-converter";
import { cacheThinkingSignature } from "./thinking-utils";

export interface SSEEvent {
  type: string;
  index?: number;
  delta?: any;
  message?: any;
  content_block?: any;
  usage?: any;
  error?: any;
}

export interface StreamResponseOptions {
  onUnknownPart?: (info: string) => void;
}

export interface GroundingAccumulator {
  queries: string[];
  seenQueries: Set<string>;
  sources: Array<{ title: string; url: string }>;
  seenUrls: Set<string>;
}

export function createGroundingAccumulator(): GroundingAccumulator {
  return { queries: [], seenQueries: new Set(), sources: [], seenUrls: new Set() };
}

/** Merges one candidate's groundingMetadata into the accumulator (deduped). */
export function mergeGroundingMetadata(acc: GroundingAccumulator, candidate: any): void {
  const grounding = candidate?.groundingMetadata;
  if (!grounding || typeof grounding !== "object") return;
  for (const query of Array.isArray(grounding.webSearchQueries) ? grounding.webSearchQueries : []) {
    if (typeof query !== "string" || !query.trim() || acc.seenQueries.has(query)) continue;
    acc.seenQueries.add(query);
    acc.queries.push(query);
  }
  for (const chunk of Array.isArray(grounding.groundingChunks) ? grounding.groundingChunks : []) {
    const web = chunk?.web;
    if (!web || typeof web.uri !== "string" || !web.uri || acc.seenUrls.has(web.uri)) continue;
    acc.seenUrls.add(web.uri);
    acc.sources.push({
      title: typeof web.title === "string" && web.title ? web.title : web.uri,
      url: web.uri,
    });
  }
}

export async function* streamSSEResponse(
  response: Response,
  model: string,
  options?: StreamResponseOptions,
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
  let currentThinkingText = "";
  const warnedShapes = new Set<string>();

  let inputTokens = 0;
  let outputTokens = 0;
  let cacheReadTokens = 0;
  let stopReason: string | null = null;
  // Tracks emitted tool calls by stable upstream id. Proxies/aggregators
  // sometimes replay the same functionCall part across chunks; re-emitting
  // the full args each time concatenates duplicate JSON downstream
  // (InputValidationError). Entries map id -> { block index, args JSON }.
  const emittedTools = new Map<string, { index: number; args: string }>();
  const grounding = createGroundingAccumulator();

  const reportUnknownPart = (part: any): void => {
    const shape = Object.keys(part || {}).sort().join(",");
    if (!shape || warnedShapes.has(shape)) return;
    warnedShapes.add(shape);
    try {
      options?.onUnknownPart?.(shape);
    } catch {
      // Reporting must never break conversion.
    }
  };

  const flushSignature = (): SSEEvent | null => {
    if (currentThinkingSignature) {
      if (currentThinkingText) {
        cacheThinkingSignature(currentThinkingSignature, currentThinkingText);
      }
      const ev: SSEEvent = {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "signature_delta", signature: currentThinkingSignature },
      };
      currentThinkingSignature = "";
      currentThinkingText = "";
      return ev;
    }
    currentThinkingText = "";
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
        mergeGroundingMetadata(grounding, firstCandidate);

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
              // Signature first: it belongs to the thinking block being
              // closed (same index); emitting after closeBlock would
              // misindex it onto the next block.
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
              currentBlockType = "thinking";
              currentThinkingSignature = "";
              currentThinkingText = "";
              yield {
                type: "content_block_start",
                index: blockIndex,
                content_block: { type: "thinking", thinking: "" },
              };
            }

            if (signature && signature.length >= MIN_SIGNATURE_LENGTH) {
              currentThinkingSignature = signature;
            }

            if (text) {
              currentThinkingText += text;
              yield {
                type: "content_block_delta",
                index: blockIndex,
                delta: { type: "thinking_delta", thinking: text },
              };
            }
          } else if ((part as any).text !== undefined) {
            const text = (part as any).text;

            if (currentBlockType !== "text") {
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
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
            const argsJson = JSON.stringify(fc.args || {});

            // Dedupe replays: aggregators sometimes resend the same
            // functionCall part across chunks. Re-emitting the full args
            // would concatenate duplicate JSON downstream
            // (InputValidationError), so identical repeats are skipped.
            // Genuinely different args under a reused id fall through as a
            // new call to preserve data. No-id parts cannot be proven
            // duplicates, so they always emit (previous behavior).
            if (typeof fc.id === "string" && fc.id) {
              const seen = emittedTools.get(fc.id);
              if (seen && seen.args === argsJson) continue;
            }

            if (currentBlockType !== null) {
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
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
                partial_json: argsJson,
              },
            };

            if (typeof fc.id === "string" && fc.id) {
              emittedTools.set(fc.id, { index: blockIndex, args: argsJson });
            }
          } else if ((part as any).inlineData) {
            const img = (part as any).inlineData;

            if (currentBlockType !== null) {
              const sigEv = flushSignature();
              if (sigEv) yield sigEv;
              const closeEv = closeBlock();
              if (closeEv) yield closeEv;
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
          } else {
            reportUnknownPart(part);
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
    yield {
      type: "error",
      error: {
        type: "api_error",
        message: "Antigravity returned no usable events for this request.",
      },
    };
    return;
  }

  const sigEv = flushSignature();
  if (sigEv) yield sigEv;
  const closeEv = closeBlock();
  if (closeEv) yield closeEv;

  // Grounded searches collected across chunks flush here, after live content
  // and before the terminal delta — same ordering as the Responses path.
  // Citations aggregate into one result block linked to the first query.
  if (grounding.queries.length > 0 || grounding.sources.length > 0) {
    let firstUseId: string | undefined;
    for (const query of grounding.queries) {
      const id = `srvtoolu_grounding_${Math.random().toString(36).slice(2, 14)}`;
      firstUseId ??= id;
      yield {
        type: "content_block_start",
        index: blockIndex,
        content_block: { type: "server_tool_use", id, name: "web_search", input: {} },
      };
      yield {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "input_json_delta", partial_json: JSON.stringify({ query }) },
      };
      yield { type: "content_block_stop", index: blockIndex };
      blockIndex++;
    }
    if (grounding.sources.length > 0) {
      const content = grounding.sources.map((source) => ({
        type: "web_search_result",
        title: source.title,
        url: source.url,
      }));
      yield {
        type: "content_block_start",
        index: blockIndex,
        content_block: {
          type: "web_search_tool_result",
          tool_use_id: firstUseId ?? `srvtoolu_${Date.now()}`,
          content,
        },
      };
      yield {
        type: "content_block_delta",
        index: blockIndex,
        delta: { type: "input_json_delta", partial_json: JSON.stringify(content) },
      };
      yield { type: "content_block_stop", index: blockIndex };
      blockIndex++;
    }
  }

  yield {
    type: "message_delta",
    delta: { stop_reason: stopReason || "end_turn", stop_sequence: null },
    usage: {
      input_tokens: inputTokens - cacheReadTokens,
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
  const grounding = createGroundingAccumulator();

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
        mergeGroundingMetadata(grounding, firstCandidate);
        const content = firstCandidate.content || {};
        const partsArr = content.parts || [];

        for (const part of partsArr) {
          // Same replay protection as the streaming path: an identical
          // functionCall id+args pair collapses into one entry, so the
          // collapsed response cannot contain duplicate tool_use blocks.
          const callId = (part as any)?.functionCall?.id;
          if (typeof callId === "string" && callId) {
            const at = parts.findIndex(
              (entry: any) => entry?.functionCall?.id === callId,
            );
            if (at >= 0) {
              const prevArgs = JSON.stringify((parts[at] as any).functionCall.args || {});
              const nextArgs = JSON.stringify((part as any).functionCall.args || {});
              if (nextArgs === prevArgs) continue;
            }
          }
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
          ...(grounding.queries.length > 0 || grounding.sources.length > 0
            ? {
                groundingMetadata: {
                  webSearchQueries: grounding.queries,
                  groundingChunks: grounding.sources.map((source) => ({
                    web: { uri: source.url, title: source.title },
                  })),
                },
              }
            : {}),
        },
      ],
      usageMetadata: {
        promptTokenCount: inputTokens,
        candidatesTokenCount: outputTokens,
        cachedContentTokenCount: cacheReadTokens,
      },
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
  model: string,
  options?: StreamResponseOptions,
): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const event of streamSSEResponse(response, model, options)) {
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