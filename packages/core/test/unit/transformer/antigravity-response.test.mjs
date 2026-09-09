import assert from "node:assert/strict";
import test from "node:test";
import { convertGoogleToAnthropic } from "@ccr/core/transformer/antigravity/response-converter.ts";
import {
  cacheThinkingSignature,
  getCachedSignature,
  restoreThinkingSignatures,
} from "@ccr/core/transformer/antigravity/thinking-utils.ts";

const LONG_SIG = "s".repeat(64);

test("text, thinking with real signature, and tool_use convert with tool_use stop", () => {
  const out = convertGoogleToAnthropic(
    {
      response: {
        candidates: [
          {
            content: {
              parts: [
                { text: "Considering", thought: true, thoughtSignature: LONG_SIG },
                { type: "text", text: "Hi" },
                { text: "Hi" },
                { functionCall: { id: "fc1", name: "get_weather", args: { city: "sf" } } },
              ],
            },
            // Gemini reports STOP even when the turn produced tool calls.
            finishReason: "STOP",
            usageMetadata: undefined,
          },
        ],
        usageMetadata: { promptTokenCount: 20, candidatesTokenCount: 8, cachedContentTokenCount: 5 },
      },
    },
    "claude-sonnet-4-6-thinking",
  );

  assert.equal(out.type, "message");
  const thinking = out.content.find((c) => c.type === "thinking");
  assert.equal(thinking.thinking, "Considering");
  assert.equal(thinking.signature, LONG_SIG);
  assert.ok(out.content.some((c) => c.type === "text" && c.text === "Hi"));
  assert.deepEqual(
    out.content.find((c) => c.type === "tool_use"),
    { type: "tool_use", id: "fc1", name: "get_weather", input: { city: "sf" } },
  );
  assert.equal(out.stop_reason, "tool_use", "tools win over STOP");
  assert.deepEqual(out.usage, {
    input_tokens: 15,
    output_tokens: 8,
    cache_read_input_tokens: 5,
    cache_creation_input_tokens: 0,
  });
});

test("MAX_TOKENS maps without tools, empty content falls back to text", () => {
  const out = convertGoogleToAnthropic(
    { candidates: [{ content: { parts: [] }, finishReason: "MAX_TOKENS" }] },
    "gemini-3-flash",
  );
  assert.equal(out.stop_reason, "max_tokens");
  assert.deepEqual(out.content, [{ type: "text", text: "" }]);
});

test("signature cache round-trips by thinking text", () => {
  cacheThinkingSignature(LONG_SIG, "unique thought abc");
  assert.equal(getCachedSignature("unique thought abc"), LONG_SIG);
  assert.equal(getCachedSignature("something else"), null);

  const restored = restoreThinkingSignatures([
    { type: "thinking", thinking: "unique thought abc", signature: "" },
    { type: "text", text: "hi" },
  ]);
  assert.equal(restored[0].signature, LONG_SIG);
  assert.equal(restored[1].type, "text");
});
