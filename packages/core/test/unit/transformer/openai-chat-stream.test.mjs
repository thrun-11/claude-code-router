import assert from "node:assert/strict";
import test from "node:test";
import { OpencodeGoTransformer } from "@ccr/core/transformer/opencodego.transformer.ts";

function openAIStream(chunks) {
  const text = chunks.map((c) => `data: ${JSON.stringify(c)}\n\n`).join("") + "data: [DONE]\n\n";
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream({
    start(controller) {
      controller.enqueue(bytes);
      controller.close();
    },
  });
}

async function collect(stream) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let out = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    out += decoder.decode(value, { stream: true });
  }
  return out;
}

function events(text) {
  return text
    .split("\n\n")
    .filter((f) => f.includes("data:"))
    .map((f) => JSON.parse(f.split("data:")[1].trim()));
}

test("openai chat stream converts text deltas with lifecycle events", async () => {
  const t = new OpencodeGoTransformer();
  const upstream = openAIStream([
    { model: "deepseek-v4-flash", choices: [{ delta: { content: "OK" }, finish_reason: null }] },
    {
      model: "deepseek-v4-flash",
      choices: [{ delta: {}, finish_reason: "stop" }],
      usage: { prompt_tokens: 9, completion_tokens: 2 },
    },
  ]);
  const converted = t.convertOpenAIStreamToAnthropic(upstream, {});
  const evts = events(await collect(converted));
  const types = evts.map((e) => e.type);

  assert.deepEqual(types, [
    "message_start",
    "content_block_start",
    "content_block_delta",
    "content_block_stop",
    "message_delta",
    "message_stop",
  ]);
  assert.equal(evts[0].message.model, "deepseek-v4-flash");
  assert.equal(evts[2].delta.text, "OK");
  assert.equal(evts[4].delta.stop_reason, "end_turn");
  // Zero cache counters are omitted (valid Anthropic usage needs only in/out).
  assert.deepEqual(evts[4].usage, {
    input_tokens: 9,
    output_tokens: 2,
  });
});

test("openai chat stream converts reasoning deltas to thinking blocks", async () => {
  const t = new OpencodeGoTransformer();
  const upstream = openAIStream([
    { choices: [{ delta: { reasoning_content: "hmm" }, finish_reason: null }] },
    { choices: [{ delta: { content: "hi" }, finish_reason: null }] },
    { choices: [{ delta: {}, finish_reason: "stop" }], usage: {} },
  ]);
  const converted = t.convertOpenAIStreamToAnthropic(upstream, {});
  const evts = events(await collect(converted));

  const thinkingStart = evts.find((e) => e.content_block?.type === "thinking");
  assert.ok(thinkingStart, "thinking block opens");
  const thinkingDelta = evts.find((e) => e.delta?.type === "thinking_delta");
  assert.equal(thinkingDelta.delta.thinking, "hmm");
  assert.ok(evts.some((e) => e.delta?.type === "signature_delta"), "thinking closes with signature");
});

test("openai chat stream converts tool call deltas to tool_use blocks", async () => {
  const t = new OpencodeGoTransformer();
  const upstream = openAIStream([
    {
      choices: [
        {
          delta: {
            tool_calls: [{ index: 0, id: "call_1", function: { name: "get_weather", arguments: '{"a":' } }],
          },
          finish_reason: null,
        },
      ],
    },
    {
      choices: [
        {
          delta: { tool_calls: [{ index: 0, function: { arguments: '1}' } }] },
          finish_reason: null,
        },
      ],
    },
    { choices: [{ delta: {}, finish_reason: "tool_calls" }], usage: {} },
  ]);
  const converted = t.convertOpenAIStreamToAnthropic(upstream, {});
  const evts = events(await collect(converted));

  const toolStart = evts.find((e) => e.content_block?.type === "tool_use");
  assert.equal(toolStart.content_block.name, "get_weather");
  const argDeltas = evts
    .filter((e) => e.delta?.type === "input_json_delta")
    .map((e) => e.delta.partial_json)
    .join("");
  assert.equal(argDeltas, '{"a":1}');
  assert.equal(evts.find((e) => e.type === "message_delta").delta.stop_reason, "tool_use");
});
