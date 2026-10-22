import assert from "node:assert/strict";
import test from "node:test";
import {
  anthropicMessageToSSE,
  createExecutedSearch,
  executeWebSearchChain,
  fallbackFunctionTool,
  fallbackQueryOf,
  FALLBACK_FN_NAME,
  parseBraveResponse,
  parseTavilyResponse,
  readWebSearchFallbackConfig,
  synthesizeSearchBlocks,
  webSearchFallbackBudget,
} from "@ccr/core/transformer/websearch-fallback.ts";
import { OpencodeGoTransformer } from "@ccr/core/transformer/opencodego.transformer.ts";

const serverTool = { type: "web_search_20250305", name: "web_search", max_uses: 3 };
const fnTool = {
  type: "function",
  function: { name: "Get", description: "read", parameters: { type: "object", properties: {} } },
};

test("config is null without keys, parses engines in order with keys", () => {
  assert.equal(readWebSearchFallbackConfig({}), null);
  assert.equal(readWebSearchFallbackConfig({ CCR_WEBSEARCH_BRAVE_KEY: "" }), null);
  const cfg = readWebSearchFallbackConfig({
    CCR_WEBSEARCH_ENGINES: "tavily,brave,nope",
    CCR_WEBSEARCH_BRAVE_KEY: "b",
    CCR_WEBSEARCH_TAVILY_KEY: "t",
    CCR_WEBSEARCH_MAX_RESULTS: "7",
  });
  assert.deepEqual(
    cfg.engines,
    [
      { name: "tavily", key: "t" },
      { name: "brave", key: "b" },
    ],
  );
  assert.equal(cfg.maxResults, 7);
  assert.equal(cfg.maxTurns, 5);
});

test("budget detects server web_search and clamps max_uses", () => {
  assert.deepEqual(webSearchFallbackBudget([fnTool, serverTool]), { maxUses: 3 });
  assert.deepEqual(webSearchFallbackBudget([{ type: "web_search_20260201", name: "web_search" }]), {
    maxUses: 5,
  });
  assert.deepEqual(webSearchFallbackBudget([{ type: "web_search_x", max_uses: 99 }]), { maxUses: 10 });
  assert.equal(webSearchFallbackBudget([fnTool]), null);
  assert.equal(webSearchFallbackBudget(undefined), null);
});

test("fallback function tool has a query schema", () => {
  const tool = fallbackFunctionTool();
  assert.equal(tool.type, "function");
  assert.equal(tool.function.name, FALLBACK_FN_NAME);
  assert.deepEqual(tool.function.parameters.required, ["query"]);
  assert.equal(
    fallbackQueryOf({ function: { name: FALLBACK_FN_NAME, arguments: '{"query":"  hi  "}' } }),
    "hi",
  );
  assert.equal(fallbackQueryOf({ function: { name: FALLBACK_FN_NAME, arguments: "{}" } }), null);
  assert.equal(fallbackQueryOf({ function: { name: FALLBACK_FN_NAME, arguments: "nope" } }), null);
});

test("engine parsers normalize Brave and Tavily shapes", () => {
  const brave = parseBraveResponse({
    web: {
      results: [
        { title: "A", url: "https://a.example", description: "desc-a" },
        { title: "", url: "https://b.example" },
        { title: "NoUrl" },
      ],
    },
  });
  assert.deepEqual(brave, [
    { title: "A", url: "https://a.example", snippet: "desc-a" },
    { title: "https://b.example", url: "https://b.example", snippet: "" },
  ]);
  const tavily = parseTavilyResponse({
    results: [{ title: "T", url: "https://t.example", content: "x".repeat(9000) }],
  });
  assert.equal(tavily.length, 1);
  assert.ok(tavily[0].snippet.length <= 500);
  assert.throws(() => parseBraveResponse({}), /Brave/);
  assert.throws(() => parseTavilyResponse({ results: "nope" }), /Tavily/);
});

test("chain falls through failed engines and throws when all fail", async () => {
  const realFetch = globalThis.fetch;
  try {
    let calls = 0;
    globalThis.fetch = async (url) => {
      calls++;
      if (String(url).includes("brave")) {
        return new Response("bad", { status: 429 });
      }
      return new Response(
        JSON.stringify({ results: [{ title: "T", url: "https://t.example", content: "c" }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    };
    const seen = [];
    const out = await executeWebSearchChain(
      "q",
      { engines: [{ name: "brave", key: "b" }, { name: "tavily", key: "t" }], maxResults: 3, maxTurns: 5 },
      (eng, msg) => seen.push([eng, msg]),
    );
    assert.equal(out.engine, "tavily");
    assert.equal(out.results[0].url, "https://t.example");
    assert.equal(calls, 2);
    assert.deepEqual(seen.map((s) => s[0]), ["brave"]);
    await assert.rejects(
      executeWebSearchChain("q", { engines: [{ name: "brave", key: "b" }], maxResults: 3, maxTurns: 5 }),
      /all web search engines failed/,
    );
  } finally {
    globalThis.fetch = realFetch;
  }
});

test("synthesized blocks link use id to result", () => {
  const s = createExecutedSearch("q", "brave", [{ title: "A", url: "https://a.example", snippet: "s" }]);
  const blocks = synthesizeSearchBlocks([s]);
  assert.equal(blocks.length, 2);
  assert.equal(blocks[0].type, "server_tool_use");
  assert.equal(blocks[0].name, "web_search");
  assert.deepEqual(blocks[0].input, { query: "q" });
  assert.equal(blocks[1].type, "web_search_tool_result");
  assert.equal(blocks[1].tool_use_id, blocks[0].id);
  assert.deepEqual(blocks[1].content, [{ type: "web_search_result", title: "A", url: "https://a.example" }]);
});

function parseSSE(sse) {
  return sse
    .split("\n")
    .filter((l) => l.startsWith("data:"))
    .map((l) => JSON.parse(l.slice(5).trim()));
}

test("SSE serializer emits citations before stop", () => {
  const s = createExecutedSearch("q", "brave", [{ title: "A", url: "https://a.example", snippet: "s" }]);
  const sse = anthropicMessageToSSE({
    id: "msg_1",
    model: "m",
    content: [...synthesizeSearchBlocks([s]), { type: "text", text: "done" }],
    stop_reason: "end_turn",
    usage: { input_tokens: 10, output_tokens: 5 },
  });
  const events = parseSSE(sse);
  const types = events.map((e) => e.type);
  assert.equal(types[0], "message_start");
  assert.equal(types[types.length - 1], "message_stop");
  const starts = events.filter((e) => e.type === "content_block_start");
  assert.deepEqual(
    starts.map((e) => e.content_block.type),
    ["server_tool_use", "web_search_tool_result", "text"],
  );
  const delta = events.find((e) => e.type === "message_delta");
  assert.equal(delta.delta.stop_reason, "end_turn");
  assert.equal(delta.usage.output_tokens, 5);
});

function anthropicTurn(content, stop = "end_turn") {
  return new Response(
    JSON.stringify({
      id: "msg_t",
      type: "message",
      role: "assistant",
      model: "space-bunny-free",
      content,
      stop_reason: stop,
      usage: { input_tokens: 100, output_tokens: 10 },
    }),
    { status: 200, headers: { "Content-Type": "application/json" } },
  );
}

test("chat loop executes search and returns citations + answer (non-stream)", async () => {
  const realFetch = globalThis.fetch;
  process.env.CCR_WEBSEARCH_BRAVE_KEY = "test-key";
  try {
    globalThis.fetch = async () =>
      new Response(
        JSON.stringify({
          web: { results: [{ title: "A", url: "https://a.example", description: "desc" }] },
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    const t = new OpencodeGoTransformer();
    const turns = [
      anthropicTurn(
        [{ type: "tool_use", id: "call_1", name: FALLBACK_FN_NAME, input: { query: "who won?" } }],
        "tool_use",
      ),
      anthropicTurn([{ type: "text", text: "Spain won." }]),
    ];
    let n = 0;
    const seenBodies = [];
    t.fetchEndpoint = async (body) => {
      seenBodies.push(body);
      return turns[Math.min(n++, turns.length - 1)];
    };
    const res = await t.sendChatWithSearchFallback(
      { model: "space-bunny-free", messages: [{ role: "user", content: "hi" }], tools: [] },
      "https://example.invalid/chat",
      {},
      false,
      { req: {} },
      3,
    );
    assert.equal(res.status, 200);
    assert.ok(res.headers.get("Content-Type").includes("application/json"));
    const msg = await res.json();
    assert.equal(msg.stop_reason, "end_turn");
    const kinds = msg.content.map((b) => b.type);
    assert.deepEqual(kinds, ["server_tool_use", "web_search_tool_result", "text"]);
    assert.equal(msg.content[2].text, "Spain won.");
    assert.equal(msg.usage.input_tokens, 200);
    // Upstream saw the function stand-in, and the second turn carried results.
    assert.ok(seenBodies[0].tools.some((x) => x?.function?.name === FALLBACK_FN_NAME));
    assert.equal(seenBodies[1].messages.length, 3);
    assert.equal(seenBodies[1].messages[2].role, "tool");
  } finally {
    globalThis.fetch = realFetch;
    delete process.env.CCR_WEBSEARCH_BRAVE_KEY;
  }
});

test("chat loop surfaces real tool calls with tool_use stop", async () => {
  process.env.CCR_WEBSEARCH_BRAVE_KEY = "test-key";
  const realFetch = globalThis.fetch;
  try {
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ web: { results: [] } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    const t = new OpencodeGoTransformer();
    t.fetchEndpoint = async () =>
      anthropicTurn(
        [
          { type: "tool_use", id: "call_9", name: "Read", input: { file: "x" } },
        ],
        "tool_use",
      );
    const res = await t.sendChatWithSearchFallback(
      { model: "m", messages: [{ role: "user", content: "hi" }], tools: [] },
      "https://example.invalid/chat",
      {},
      true,
      { req: {} },
      3,
    );
    assert.ok(res.headers.get("Content-Type").includes("text/event-stream"));
    const events = parseSSE(await res.text());
    const starts = events.filter((e) => e.type === "content_block_start");
    assert.deepEqual(
      starts.map((e) => e.content_block.type),
      ["tool_use"],
    );
    assert.equal(starts[0].content_block.name, "Read");
    const delta = events.find((e) => e.type === "message_delta");
    assert.equal(delta.delta.stop_reason, "tool_use");
  } finally {
    globalThis.fetch = realFetch;
    delete process.env.CCR_WEBSEARCH_BRAVE_KEY;
  }
});
