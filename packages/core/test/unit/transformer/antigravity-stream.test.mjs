import assert from "node:assert/strict";
import test from "node:test";
import {
  accumulateSSEToResponse,
  streamSSEResponse,
} from "@ccr/core/transformer/antigravity/sse-parser.ts";

function stubResponse(frames) {
  const text = frames.map((f) => `data: ${JSON.stringify(f)}\n\n`).join("");
  const bytes = new TextEncoder().encode(text);
  return new Response(
    new ReadableStream({
      start(controller) {
        controller.enqueue(bytes);
        controller.close();
      },
    }),
    { headers: { "Content-Type": "text/event-stream" } },
  );
}

async function collect(gen) {
  const out = [];
  for await (const e of gen) out.push(e);
  return out;
}

const SIG = "g".repeat(64);

test("thinking signature precedes its block stop at the same index", async () => {
  const events = await collect(
    streamSSEResponse(
      stubResponse([
        { response: { candidates: [{ content: { parts: [{ text: "Hmm", thought: true, thoughtSignature: SIG }] } }] } },
        { response: { candidates: [{ content: { parts: [{ text: "Hi" }] } }] } },
        { response: { candidates: [{ content: { parts: [] }, finishReason: "STOP" }] } },
      ]),
      "gemini-3-flash",
    ),
  );
  const types = events.map((e) => e.type);
  assert.ok(types.includes("message_start"));
  assert.ok(types.includes("message_stop"));

  const thinkingStartIdx = events.findIndex(
    (e) => e.type === "content_block_start" && e.content_block?.type === "thinking",
  );
  const thinkingIdx = events[thinkingStartIdx].index;
  const sigIdx = events.findIndex((e) => e.delta?.type === "signature_delta");
  const stopIdx = events.findIndex(
    (e, i) => i > sigIdx && e.type === "content_block_stop" && e.index === thinkingIdx,
  );
  assert.notEqual(sigIdx, -1, "signature delta emitted");
  assert.equal(events[sigIdx].index, thinkingIdx, "signature targets the thinking block");
  assert.ok(stopIdx > sigIdx, "signature comes before the thinking block stop");
  assert.equal(events[sigIdx].delta.signature, SIG);
});

test("functionCall parts become tool_use with tool_use stop", async () => {
  const events = await collect(
    streamSSEResponse(
      stubResponse([
        {
          response: {
            candidates: [
              {
                content: {
                  parts: [{ functionCall: { id: "fc1", name: "read", args: { path: "x" } } }],
                },
                finishReason: "STOP",
              },
            ],
          },
        },
      ]),
      "gemini-3-flash",
    ),
  );
  const start = events.find((e) => e.content_block?.type === "tool_use");
  assert.equal(start.content_block.name, "read");
  const argDelta = events.find((e) => e.delta?.type === "input_json_delta");
  assert.equal(argDelta.delta.partial_json, JSON.stringify({ path: "x" }));
  assert.equal(
    events.find((e) => e.type === "message_delta").delta.stop_reason,
    "tool_use",
  );
});

test("accumulated non-stream usage attributes input/output/cache tokens", async () => {
  const frame =
    'data: {"response":{"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"cachedContentTokenCount":10}}}\n\n';
  const resp = await accumulateSSEToResponse(new Response(frame), "m");
  const body = await resp.json();
  assert.deepEqual(body.usage, {
    input_tokens: 90,
    output_tokens: 20,
    cache_read_input_tokens: 10,
    cache_creation_input_tokens: 0,
  });
});

test("empty stream yields an error event, unknown parts warn once", async () => {
  const empty = await collect(
    streamSSEResponse(stubResponse([{ response: { candidates: [] } }]), "m"),
  );
  const err = empty.find((e) => e.type === "error");
  assert.ok(err, "no usable events becomes an error, not fabricated text");
  assert.equal(empty.some((e) => e.type === "message_start"), false);

  const warned = [];
  const events = await collect(
    streamSSEResponse(
      stubResponse([
        { response: { candidates: [{ content: { parts: [{ weird: 1 }, { weird: 2 }, { text: "hi" }] } }] } },
      ]),
      "m",
      { onUnknownPart: (s) => warned.push(s) },
    ),
  );
  assert.deepEqual(warned, ["weird"], "unknown shape reported once");
  assert.ok(events.some((e) => e.delta?.text === "hi"), "known parts still convert");
});
