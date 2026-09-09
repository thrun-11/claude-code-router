/**
 * Shared Anthropic output builders for provider transformers.
 *
 * Every transformer speaks a different upstream dialect (OpenAI chat,
 * OpenAI Responses, Copilot Responses, Codex Responses, Gemini), but they
 * all converge on the same two Anthropic shapes: a Message object for
 * non-streaming responses and an SSE event sequence for streaming ones.
 * Building those shapes in one place keeps block lifecycles, stop reasons,
 * and usage fields consistent instead of drifting per provider.
 */

export type AnthropicStopReason =
  | "end_turn"
  | "max_tokens"
  | "tool_use"
  | "stop_sequence";

export function sseEvent(event: string, data: unknown): string {
  return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
}

export function messageStartEvent(messageId: string, model: string): string {
  return sseEvent("message_start", {
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
}

export function contentBlockStartEvent(
  index: number,
  block: Record<string, unknown>,
): string {
  return sseEvent("content_block_start", {
    type: "content_block_start",
    index,
    content_block: block,
  });
}

export function textBlockStartEvent(index: number): string {
  return contentBlockStartEvent(index, { type: "text", text: "" });
}

export function thinkingBlockStartEvent(index: number): string {
  return contentBlockStartEvent(index, { type: "thinking", thinking: "" });
}

export function toolUseBlockStartEvent(
  index: number,
  id: string,
  name: string,
): string {
  return contentBlockStartEvent(index, { type: "tool_use", id, name, input: {} });
}

export function textDeltaEvent(index: number, text: string): string {
  return sseEvent("content_block_delta", {
    type: "content_block_delta",
    index,
    delta: { type: "text_delta", text },
  });
}

export function thinkingDeltaEvent(index: number, thinking: string): string {
  return sseEvent("content_block_delta", {
    type: "content_block_delta",
    index,
    delta: { type: "thinking_delta", thinking },
  });
}

export function signatureDeltaEvent(index: number, signature: string): string {
  return sseEvent("content_block_delta", {
    type: "content_block_delta",
    index,
    delta: { type: "signature_delta", signature },
  });
}

export function inputJsonDeltaEvent(index: number, partialJson: string): string {
  return sseEvent("content_block_delta", {
    type: "content_block_delta",
    index,
    delta: { type: "input_json_delta", partial_json: partialJson },
  });
}

export function contentBlockStopEvent(index: number): string {
  return sseEvent("content_block_stop", { type: "content_block_stop", index });
}

export function messageDeltaEvent(
  stopReason: AnthropicStopReason,
  usage: Record<string, number>,
): string {
  return sseEvent("message_delta", {
    type: "message_delta",
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: {
      input_tokens: usage.input_tokens || 0,
      output_tokens: usage.output_tokens || 0,
      ...(usage.cache_read_input_tokens
        ? { cache_read_input_tokens: usage.cache_read_input_tokens }
        : {}),
    },
  });
}

export function messageStopEvent(): string {
  return sseEvent("message_stop", { type: "message_stop" });
}

export function errorEvent(message: string): string {
  return sseEvent("error", {
    type: "error",
    error: { type: "api_error", message },
  });
}

export function buildAnthropicMessage(input: {
  id?: string;
  model?: string;
  content: unknown[];
  stopReason?: AnthropicStopReason;
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
    cache_read_input_tokens?: number;
  };
}): Record<string, unknown> {
  const usage = input.usage || {};
  return {
    id: input.id || `msg_${Date.now()}`,
    type: "message",
    role: "assistant",
    model: input.model || "unknown",
    content:
      input.content.length > 0 ? input.content : [{ type: "text", text: "" }],
    stop_reason: input.stopReason || "end_turn",
    stop_sequence: null,
    usage: {
      input_tokens: usage.input_tokens || 0,
      output_tokens: usage.output_tokens || 0,
      ...(usage.cache_read_input_tokens
        ? { cache_read_input_tokens: usage.cache_read_input_tokens }
        : {}),
    },
  };
}

export type ResponsesContentItem =
  | { type: "text"; text: string }
  | { type: "thinking"; thinking: string; signature: string }
  | { type: "tool_use"; id: string; name: string; input: unknown };

/**
 * Maps OpenAI-style Responses `output` items to Anthropic content blocks.
 *
 * Honest thinking policy: a `thinking` block is emitted only when the
 * upstream item carries visible summary text. Encrypted/opaque reasoning
 * carries nothing the client can display or re-sign, so it is skipped
 * rather than fabricated. Callers that need a placeholder (display-only
 * flows) can opt back in explicitly.
 */
export function responsesOutputToContent(
  output: unknown,
  options?: { includeEmptyThinking?: boolean },
): { content: ResponsesContentItem[]; hasToolUse: boolean } {
  const content: ResponsesContentItem[] = [];
  let hasToolUse = false;

  for (const item of (output as any[]) || []) {
    if (item?.type === "reasoning") {
      const summary = Array.isArray(item.summary)
        ? item.summary
            .map((part: any) =>
              typeof part === "string" ? part : part?.text || "",
            )
            .join("")
        : "";
      if (summary) {
        content.push({
          type: "thinking",
          thinking: summary,
          signature: String(Date.now()),
        });
      } else if (options?.includeEmptyThinking) {
        content.push({
          type: "thinking",
          thinking: "",
          signature: String(Date.now()),
        });
      }
      continue;
    }

    if (item?.type === "message" && Array.isArray(item.content)) {
      for (const part of item.content) {
        if (part?.type === "output_text" && part.text) {
          content.push({ type: "text", text: part.text });
        }
      }
      continue;
    }

    if (item?.type === "function_call") {
      let input: unknown = {};
      try {
        input = JSON.parse(item.arguments || "{}");
      } catch {
        input = {};
      }
      content.push({
        type: "tool_use",
        id: item.call_id || item.id || `toolu_${Date.now()}`,
        name: item.name,
        input,
      });
      hasToolUse = true;
    }
  }

  return { content, hasToolUse };
}

/**
 * Stateful block tracker for Responses→Anthropic stream conversion.
 * Owns Anthropic block indices and maps upstream output items to them,
 * so converters stay focused on protocol parsing instead of bookkeeping.
 */
export class AnthropicBlockTracker {
  private nextIndex = 0;
  private openBlocks: number[] = [];
  private textBlock: number | null = null;
  private thinkingBlock: number | null = null;
  private toolByItemId = new Map<string, number>();
  private toolByOutputIndex = new Map<number, number>();
  private outputIndexBlock = new Map<number, number>();
  hasToolUse = false;

  openText(outputIndex?: number): { index: number; started: boolean } {
    if (this.textBlock !== null) {
      if (typeof outputIndex === "number") {
        this.outputIndexBlock.set(outputIndex, this.textBlock);
      }
      return { index: this.textBlock, started: false };
    }
    const index = this.alloc(outputIndex);
    this.textBlock = index;
    return { index, started: true };
  }

  openThinking(outputIndex?: number): { index: number; started: boolean } {
    if (this.thinkingBlock !== null) {
      if (typeof outputIndex === "number") {
        this.outputIndexBlock.set(outputIndex, this.thinkingBlock);
      }
      return { index: this.thinkingBlock, started: false };
    }
    const index = this.alloc(outputIndex);
    this.thinkingBlock = index;
    return { index, started: true };
  }

  openTool(
    item: { id?: string; call_id?: string; name?: string },
    outputIndex?: number,
  ): number | null {
    if (!item?.name) {
      return this.resolveTool(item?.id || item?.call_id, outputIndex);
    }
    if (item.id) {
      const existing = this.toolByItemId.get(item.id);
      if (existing !== undefined) return existing;
    }
    const index = this.nextIndex++;
    this.openBlocks.push(index);
    if (item.id) this.toolByItemId.set(item.id, index);
    if (typeof outputIndex === "number") {
      this.toolByOutputIndex.set(outputIndex, index);
      this.outputIndexBlock.set(outputIndex, index);
    }
    this.hasToolUse = true;
    return index;
  }

  toolIdFor(item: { id?: string; call_id?: string }): string {
    return item.call_id || item.id || `toolu_${Date.now()}`;
  }

  resolveTool(itemId?: string, outputIndex?: number): number | null {
    if (itemId && this.toolByItemId.has(itemId)) {
      return this.toolByItemId.get(itemId)!;
    }
    if (typeof outputIndex === "number") {
      const mapped = this.toolByOutputIndex.get(outputIndex);
      if (mapped !== undefined && mapped !== this.textBlock && mapped !== this.thinkingBlock) {
        return mapped;
      }
    }
    return null;
  }

  isThinkingBlock(index: number): boolean {
    return index === this.thinkingBlock;
  }

  closeAll(): Array<{ index: number; thinking: boolean }> {
    const thinking = this.thinkingBlock;
    const blocks = this.openBlocks.map((index) => ({
      index,
      thinking: index === thinking,
    }));
    this.openBlocks = [];
    this.textBlock = null;
    this.thinkingBlock = null;
    return blocks;
  }

  private alloc(outputIndex?: number): number {
    const index = this.nextIndex++;
    this.openBlocks.push(index);
    if (typeof outputIndex === "number") {
      this.outputIndexBlock.set(outputIndex, index);
    }
    return index;
  }
}
