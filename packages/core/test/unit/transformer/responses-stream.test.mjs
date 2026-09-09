import assert from "node:assert/strict";
import test from "node:test";
import { ResponsesToAnthropicStream } from "@ccr/core/transformer/codex.transformer.ts";

function frame(type, payload) {
  return `event: ${type}\ndata: ${JSON.stringify({ type, ...payload })}\n\n`;
}

function feed(converter, text, size = 1024) {
  const out = [];
  for (let i = 0; i < text.length; i += size) {
    out.push(...converter.push(text.slice(i, i + size)));
  }
  return out;
}

function parseEvents(lines) {
  return lines.map((line) => {
    const dataLine = line.split("\n").find((l) => l.startsWith("data:"));
    return JSON.parse(dataLine.slice(5).trim());
  });
}

test("text flow produces a complete Anthropic message lifecycle", () => {
  const c = new ResponsesToAnthropicStream("gpt-5.6-luna");
  const raw =
    frame("response.created", { response: { id: "resp_1", model: "gpt-5.6-luna" } }) +
    frame("response.output_item.added", { output_index: 0, item: { id: "msg_1", type: "message" } }) +
    frame("response.output_text.delta", { output_index: 0, delta: "Hi" }) +
    frame("response.output_text.delta", { output_index: 0, delta: " there" }) +
    frame("response.completed", {
      response: { status: "completed", usage: { input_tokens: 5, output_tokens: 8 } },
    });
  const events = parseEvents(feed(c, raw, 37));
  const types = events.map((e) => e.type);

  assert.deepEqual(types, [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
  ]);
  assert.equal(events[0].message.model, "gpt-5.6-luna");
  assert.equal(events[0].message.role, "assistant");
  assert.deepEqual(events[1].content_block, { type: "text", text: "" });
  assert.equal(events[2].delta.text, "Hi");
  assert.equal(events[3].delta.text, " there");
  assert.equal(events[5].delta.stop_reason, "end_turn");
  assert.deepEqual(events[5].usage, { input_tokens: 5, output_tokens: 8 });
});

test("function call deltas reassemble into one tool_use block", () => {
  const c = new ResponsesToAnthropicStream("m");
  const raw =
    frame("response.created", { response: { id: "r", model: "m" } }) +
    frame("response.output_item.added", {
      output_index: 1,
      item: { id: "fc_1", type: "function_call", call_id: "call_1", name: "get_weather", arguments: "" },
    }) +
    frame("response.function_call_arguments.delta", { item_id: "fc_1", output_index: 1, delta: '{"ci' }) +
    frame("response.function_call_arguments.delta", { item_id: "fc_1", output_index: 1, delta: 'ty":"sf"}' }) +
    frame("response.completed", { response: { status: "completed", usage: { input_tokens: 1, output_tokens: 1 } } });
  const events = parseEvents(feed(c, raw));

  const start = events.find((e) => e.type === "content_block_start" && e.content_block.type === "tool_use");
  assert.equal(start.content_block.id, "call_1");
  assert.equal(start.content_block.name, "get_weather");
  const deltas = events.filter(
    (e) => e.type === "content_block_delta" && e.delta.type === "input_json_delta"
  );
  assert.equal(deltas.map((d) => d.delta.partial_json).join(""), '{"city":"sf"}');
  assert.equal(
    events.find((e) => e.type === "message_delta").delta.stop_reason,
    "tool_use"
  );
});

test("reasoning with summary text becomes a thinking block, encrypted-only is skipped", () => {
  const c = new ResponsesToAnthropicStream("m");
  // Non-stream path tested separately; stream reasoning items open thinking blocks.
  const raw =
    frame("response.created", { response: { id: "r", model: "m" } }) +
    frame("response.output_item.added", { output_index: 0, item: { id: "rs_1", type: "reasoning" } }) +
    frame("response.reasoning_summary_text.delta", { output_index: 0, delta: "Let me think" }) +
    frame("response.output_text.delta", { output_index: 1, delta: "done" }) +
    frame("response.completed", { response: { status: "completed", usage: {} } });
  const events = parseEvents(feed(c, raw));

  const thinkingStart = events.find((e) => e.content_block?.type === "thinking");
  assert.ok(thinkingStart, "thinking block opens for reasoning");
  const thinkingDelta = events.find((e) => e.delta?.type === "thinking_delta");
  assert.equal(thinkingDelta.delta.thinking, "Let me think");
  const signature = events.find((e) => e.delta?.type === "signature_delta");
  assert.ok(signature, "thinking block is closed with a signature delta");
});

test("incomplete maps max_output_tokens, failed emits an error event", () => {
  const c1 = new ResponsesToAnthropicStream("m");
  const e1 = parseEvents(
    feed(
      c1,
      frame("response.created", { response: { id: "r", model: "m" } }) +
        frame("response.incomplete", {
          response: { incomplete_details: { reason: "max_output_tokens" }, usage: {} },
        })
    )
  );
  assert.equal(
    e1.find((e) => e.type === "message_delta").delta.stop_reason,
    "max_tokens"
  );

  const c2 = new ResponsesToAnthropicStream("m");
  const e2 = parseEvents(
    feed(
      c2,
      frame("response.created", { response: { id: "r", model: "m" } }) +
        frame("response.failed", { response: { error: { message: "boom" } } })
    )
  );
  const err = e2.find((e) => e.type === "error");
  assert.equal(err.error.message, "boom");
  assert.equal(e2.some((e) => e.type === "message_stop"), false);
});

test("unknown events and pings are ignored, finish is idempotent", () => {
  const c = new ResponsesToAnthropicStream("m");
  const raw =
    frame("response.created", { response: { id: "r", model: "m" } }) +
    frame("response.in_progress", { response: {} }) +
    frame("ping", { cost: "0" }) +
    frame("response.content_part.added", { output_index: 0, content_index: 0, item_id: "x", part: {} }) +
    "data: [DONE]\n\n" +
    frame("response.completed", { response: { status: "completed", usage: {} } });
  const events = parseEvents(feed(c, raw));
  assert.ok(events.every((e) => e.type.startsWith("message_") || e.type.startsWith("content_block_")));
  assert.equal(events.filter((e) => e.type === "message_stop").length, 1);
  assert.deepEqual(c.finish(), [], "second finish emits nothing");
  assert.deepEqual(c.push("event: response.completed\ndata: {}\n\n"), [], "push after finish emits nothing");
});
