import assert from "node:assert/strict";
import test from "node:test";
import { CodexTransformer } from "@ccr/core/transformer/codex.transformer.ts";
import { OpencodeGoTransformer } from "@ccr/core/transformer/opencodego.transformer.ts";

test("codex Responses payload converts to an Anthropic message", () => {
  const t = new CodexTransformer();
  const out = t.convertResponsesToAnthropic({
    id: "resp_1",
    model: "gpt-5.6-luna",
    status: "completed",
    output: [
      {
        type: "message",
        content: [{ type: "output_text", text: "Hi!" }],
      },
      {
        type: "function_call",
        call_id: "call_1",
        name: "get_weather",
        arguments: '{"city":"sf"}',
      },
    ],
    usage: { input_tokens: 10, output_tokens: 5 },
  });

  assert.equal(out.type, "message");
  assert.equal(out.role, "assistant");
  assert.equal(out.stop_reason, "tool_use");
  assert.deepEqual(out.content, [
    { type: "text", text: "Hi!" },
    { type: "tool_use", id: "call_1", name: "get_weather", input: { city: "sf" } },
  ]);
  assert.deepEqual(out.usage, { input_tokens: 10, output_tokens: 5 });
});

test("codex reasoning with visible summary becomes thinking, encrypted-only is skipped", () => {
  const t = new CodexTransformer();
  const out = t.convertResponsesToAnthropic({
    id: "resp_2",
    model: "m",
    status: "completed",
    output: [
      { type: "reasoning", summary: [{ type: "summary_text", text: "Considering" }] },
      { type: "reasoning", encrypted_content: "opaque-blob" },
      { type: "message", content: [{ type: "output_text", text: "done" }] },
    ],
    usage: {},
  });

  const thinking = out.content.filter((c) => c.type === "thinking");
  assert.equal(thinking.length, 1);
  assert.equal(thinking[0].thinking, "Considering");
  assert.equal(typeof thinking[0].signature, "string");
  assert.ok(out.content.some((c) => c.type === "text" && c.text === "done"));
});

test("codex malformed tool arguments fall back to an empty input object", () => {
  const t = new CodexTransformer();
  const out = t.convertResponsesToAnthropic({
    id: "resp_3",
    model: "m",
    status: "completed",
    output: [{ type: "function_call", call_id: "c", name: "f", arguments: "not-json{{" }],
    usage: {},
  });
  assert.deepEqual(out.content[0].input, {});
  assert.equal(out.stop_reason, "tool_use");
});

test("opencode Responses payload converts the same way (shared contract)", () => {
  const t = new OpencodeGoTransformer();
  const out = t.convertResponsesToAnthropic({
    id: "resp_9",
    model: "muse-spark-1.3-contributor",
    output: [
      { type: "message", content: [{ type: "output_text", text: "OK" }] },
      { type: "function_call", call_id: "call_9", name: "read", arguments: '{"path":"x"}' },
    ],
    usage: { input_tokens: 12, output_tokens: 175, input_tokens_details: { cached_tokens: 3 } },
  });

  assert.equal(out.stop_reason, "tool_use");
  assert.deepEqual(out.content[1], {
    type: "tool_use",
    id: "call_9",
    name: "read",
    input: { path: "x" },
  });
  assert.equal(out.usage.cache_read_input_tokens, 3);
});
