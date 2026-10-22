import assert from "node:assert/strict";
import test from "node:test";
import { convertAnthropicToGoogle } from "@ccr/core/transformer/antigravity/request-converter.ts";
import { convertGoogleToAnthropic } from "@ccr/core/transformer/antigravity/response-converter.ts";
import { streamSSEResponse } from "@ccr/core/transformer/antigravity/sse-parser.ts";

function sseFrames(frames) {
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

const webSearchTool = { type: "web_search_20250305", name: "web_search", max_uses: 8 };
const fnTool = {
  type: "function",
  function: { name: "Get", description: "read", parameters: { type: "object" } },
};

test("gemini request maps web_search to native google_search", () => {
  const body = convertAnthropicToGoogle({
    model: "gemini-3.8-flash",
    messages: [{ role: "user", content: "hi" }],
    tools: [webSearchTool, fnTool],
  });
  const flat = JSON.stringify(body.tools);
  assert.ok(body.tools.some((t) => t.google_search !== undefined), "google_search entry present");
  assert.ok(!flat.includes("web_search_20250305"), "no flattened server tool left");
  const decls = body.tools.find((t) => t.functionDeclarations).functionDeclarations;
  assert.deepEqual(
    decls.map((d) => d.name),
    ["Get"],
  );
});

test("gemini request without web_search has no google_search entry", () => {
  const body = convertAnthropicToGoogle({
    model: "gemini-3.8-flash",
    messages: [{ role: "user", content: "hi" }],
    tools: [fnTool],
  });
  assert.ok(!body.tools.some((t) => t.google_search !== undefined));
  assert.equal(body.tools[0].functionDeclarations.length, 1);
});

test("claude-model request keeps legacy shape (no google_search)", () => {
  const body = convertAnthropicToGoogle({
    model: "claude-sonnet-5",
    messages: [{ role: "user", content: "hi" }],
    tools: [webSearchTool],
  });
  assert.ok(!JSON.stringify(body.tools).includes("google_search"));
});

test("non-stream response maps grounding to citations", () => {
  const res = convertGoogleToAnthropic(
    {
      response: {
        candidates: [
          {
            content: { parts: [{ text: "Spain won." }] },
            finishReason: "STOP",
            groundingMetadata: {
              webSearchQueries: ["Who won?", "Who won?"],
              groundingChunks: [
                { web: { uri: "https://x.example/a", title: "A" } },
                { web: { uri: "https://x.example/a", title: "A" } },
                { web: { uri: "https://x.example/b" } },
              ],
            },
          },
        ],
        usageMetadata: { promptTokenCount: 10, candidatesTokenCount: 5 },
      },
    },
    "gemini-3.8-flash",
  );
  const uses = res.content.filter((b) => b.type === "server_tool_use");
  assert.equal(uses.length, 1);
  assert.deepEqual(uses[0].input, { query: "Who won?" });
  const results = res.content.filter((b) => b.type === "web_search_tool_result");
  assert.equal(results.length, 1);
  assert.equal(results[0].tool_use_id, uses[0].id);
  assert.deepEqual(results[0].content, [
    { type: "web_search_result", title: "A", url: "https://x.example/a" },
    { type: "web_search_result", title: "https://x.example/b", url: "https://x.example/b" },
  ]);
  assert.equal(res.stop_reason, "end_turn");
});

test("streaming flushes grounding citations at end", async () => {
  const events = await collect(
    streamSSEResponse(
      sseFrames([
        { response: { candidates: [{ content: { parts: [{ text: "Hi" }] } }] } },
        {
          response: {
            candidates: [
              {
                content: { parts: [] },
                groundingMetadata: {
                  webSearchQueries: ["Who won?"],
                  groundingChunks: [{ web: { uri: "https://x.example/a", title: "A" } }],
                },
              },
            ],
          },
        },
        { response: { candidates: [{ content: { parts: [] }, finishReason: "STOP" }] } },
      ]),
      "gemini-3.8-flash",
    ),
  );
  const starts = events.filter((e) => e.type === "content_block_start");
  const use = starts.find((b) => b.content_block?.type === "server_tool_use");
  assert.ok(use, "server_tool_use streamed");
  const result = starts.find((b) => b.content_block?.type === "web_search_tool_result");
  assert.ok(result, "citation result streamed");
  assert.equal(result.content_block.tool_use_id, use.content_block.id);
  assert.deepEqual(result.content_block.content, [
    { type: "web_search_result", title: "A", url: "https://x.example/a" },
  ]);
  const stopIdx = events.findIndex((e) => e.type === "message_stop");
  const resIdx = events.findIndex(
    (e) => e.type === "content_block_start" && e.content_block?.type === "web_search_tool_result",
  );
  assert.ok(resIdx !== -1 && resIdx < stopIdx, "citations close before message_stop");
});
