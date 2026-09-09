import assert from "node:assert/strict";
import test from "node:test";
import { CopilotTransformer } from "@ccr/core/transformer/copilot.transformer.ts";

// NOTE: copilot streaming currently emits bare data: chunks without a full
// message_start/content_block lifecycle. These tests lock that behavior;
// the emitter migration upgrades it to a complete lifecycle.
test("copilot text delta chunk converts to a message_delta", () => {
  const t = new CopilotTransformer();
  const chunk =
    'data: {"type":"response.output_text.delta","delta":"hello"}\n\n';
  const out = t.convertStreamChunk(chunk);
  assert.ok(out !== null);
  const payload = JSON.parse(out.slice("data: ".length).trim());
  assert.equal(payload.type, "message_delta");
  assert.equal(payload.delta.text, "hello");
});

test("copilot message item added converts to a message_start", () => {
  const t = new CopilotTransformer();
  const chunk =
    'data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1"},"response":{"model":"gpt-5"}}\n\n';
  const out = t.convertStreamChunk(chunk);
  const payload = JSON.parse(out.slice("data: ".length).trim());
  assert.equal(payload.type, "message_start");
  assert.equal(payload.message.id, "msg_1");
});

test("copilot completed converts stop reason and usage", () => {
  const t = new CopilotTransformer();
  const chunk =
    'data: {"response":{"type":"response.completed","incomplete_details":{"reason":"max_output_tokens"}},"usage":{"input_tokens":1}}\n\n';
  const out = t.convertStreamChunk(chunk);
  const payload = JSON.parse(out.slice("data: ".length).trim());
  assert.equal(payload.delta.stop_reason, "max_tokens");
});

test("copilot non-data chunks pass through unchanged", () => {
  const t = new CopilotTransformer();
  assert.equal(t.convertStreamChunk(": ping\n\n"), ": ping\n\n");
  assert.equal(t.convertStreamChunk("\n\n"), "\n\n");
});
