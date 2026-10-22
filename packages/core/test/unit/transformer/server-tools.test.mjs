import assert from "node:assert/strict";
import test from "node:test";
import { AnthropicTransformer } from "@ccr/core/transformer/anthropic.transformer.ts";
import { responsesOutputToContent } from "@ccr/core/transformer/anthropic-emitter.ts";
import {
  isFunctionTool,
  isWebSearchTool,
  mapChatTools,
  mapClaudeTools,
  mapResponsesTools,
  serverToolKind,
  toChatTool,
  toClaudeTool,
  toResponsesTool,
} from "@ccr/core/transformer/server-tools.ts";

const fnTool = {
  type: "function",
  function: { name: "Get", description: "read", parameters: { type: "object" } },
};
const serverTool = { type: "web_search_20250305", name: "web_search", max_uses: 8 };
const legacyFlat = { type: "function", function: { name: "web_search", description: "", parameters: undefined } };

test("isFunctionTool separates function from server tools", () => {
  assert.equal(isFunctionTool(fnTool), true);
  assert.equal(isFunctionTool(serverTool), false);
  assert.equal(isFunctionTool({ name: "x" }), false);
  assert.equal(isFunctionTool(undefined), false);
});

test("serverToolKind reports explicit non-function types", () => {
  assert.equal(serverToolKind(serverTool), "web_search_20250305");
  assert.equal(serverToolKind(fnTool), undefined);
});

test("isWebSearchTool matches native, legacy-flat, and function shapes", () => {
  assert.equal(isWebSearchTool(serverTool), true);
  assert.equal(isWebSearchTool(legacyFlat), true);
  assert.equal(isWebSearchTool(fnTool), false);
  assert.equal(
    isWebSearchTool({ type: "function", function: { name: "Get", parameters: {} } }),
    false,
  );
});

test("toResponsesTool maps functions, translates web_search, skips the rest", () => {
  assert.deepEqual(toResponsesTool(fnTool), {
    type: "function",
    name: "Get",
    description: "read",
    parameters: { type: "object" },
  });
  assert.deepEqual(toResponsesTool(serverTool), { type: "web_search" });
  assert.deepEqual(toResponsesTool(legacyFlat), { type: "web_search" });
  assert.equal(toResponsesTool({ type: "code_execution_20250522" }), null);
  assert.equal(toResponsesTool({ type: "function", function: {} }), null);
});

test("toChatTool passes functions through and drops server tools", () => {
  assert.equal(toChatTool(fnTool), fnTool);
  assert.equal(toChatTool(serverTool), null);
});

test("toClaudeTool maps functions and passes server tools through verbatim", () => {
  assert.deepEqual(toClaudeTool(fnTool), {
    name: "Get",
    description: "read",
    input_schema: { type: "object" },
  });
  const out = toClaudeTool(serverTool);
  assert.equal(out.type, "web_search_20250305");
  assert.equal(out.max_uses, 8);
});

test("producer preserves server tools instead of flattening them", () => {
  const t = new AnthropicTransformer();
  const out = t.convertAnthropicToolsToUnified([
    { type: "web_search_20250305", name: "web_search", max_uses: 8 },
    { name: "Get", description: "read", input_schema: { type: "object" } },
  ]);
  assert.equal(out[0].type, "web_search_20250305");
  assert.equal(out[0].max_uses, 8);
  assert.equal(out[0].function, undefined);
  assert.equal(out[1].type, "function");
  assert.equal(out[1].function.name, "Get");
});

test("shared converter maps searches and citations, dedupes repeats", () => {
  const { content, hasToolUse } = responsesOutputToContent([
    { type: "reasoning", summary: [], encrypted_content: "opaque" },
    {
      type: "message",
      content: [
        {
          type: "output_text",
          text: "Spain won.",
          annotations: [
            { type: "url_citation", title: "T", url: "https://x.example/y" },
            { type: "url_citation", title: "T", url: "https://x.example/y" },
          ],
        },
      ],
    },
    { type: "web_search_call", id: "ws_1", status: "completed", action: { type: "search", query: "Who won?" } },
    { type: "web_search_call", id: "ws_2", status: "completed", action: { type: "open_page" } },
  ]);
  const uses = content.filter((b) => b.type === "server_tool_use");
  assert.equal(uses.length, 1);
  assert.deepEqual(uses[0], {
    type: "server_tool_use",
    id: "ws_1",
    name: "web_search",
    input: { query: "Who won?" },
  });
  const results = content.filter((b) => b.type === "web_search_tool_result");
  assert.equal(results.length, 1);
  assert.equal(results[0].tool_use_id, "ws_1");
  assert.deepEqual(results[0].content, [
    { type: "web_search_result", title: "T", url: "https://x.example/y" },
  ]);
  assert.equal(hasToolUse, false);
  assert.ok(content.some((b) => b.type === "text" && b.text === "Spain won."));
});

test("map helpers filter empties and report skips", () => {
  const skipped = [];
  assert.equal(mapResponsesTools([], (m) => skipped.push(m)), undefined);
  assert.equal(mapResponsesTools(undefined), undefined);
  const out = mapResponsesTools([fnTool, { type: "code_execution_20250522" }], (m) =>
    skipped.push(m),
  );
  assert.equal(out.length, 1);
  assert.equal(skipped.length, 1);
  assert.ok(skipped[0].includes("code_execution_20250522"));
  assert.equal(mapChatTools([serverTool], () => {}), undefined);
});
