import assert from "node:assert/strict";
import test from "node:test";
import { convertAnthropicToGoogle } from "@ccr/core/transformer/antigravity/request-converter.ts";
import { buildCloudCodeRequest } from "@ccr/core/transformer/antigravity/request-builder.ts";

function geminiRequest(overrides = {}) {
  return {
    model: "gemini-3.8-flash-high",
    messages: [{ role: "user", content: "hi" }],
    ...overrides,
  };
}

test("system prefix carries identity with no [ignore] self-negation (gemini)", () => {
  const out = buildCloudCodeRequest(
    { contents: [], generationConfig: {} },
    "proj",
    "tok",
    { model: "gemini-3.8-flash-high" },
  );
  const texts = out.request.systemInstruction.parts.map((p) => p.text);
  assert.ok(texts[0].includes("Antigravity"));
  assert.ok(!texts.some((t) => t.includes("[ignore]")), "no ignore wrapper");
  assert.ok(
    texts.some((t) => t.includes("<USER_REQUEST>")),
    "tag convention explained",
  );
});

test("system prefix stays a single identity line for claude models", () => {
  const out = buildCloudCodeRequest(
    { contents: [], generationConfig: {} },
    "proj",
    "tok",
    { model: "claude-sonnet-4-6" },
  );
  const texts = out.request.systemInstruction.parts.map((p) => p.text);
  assert.equal(texts.length, 1);
  assert.ok(!texts.some((t) => t.includes("[ignore]")));
});

test("gemini user text is framed in USER_REQUEST tags", () => {
  const out = convertAnthropicToGoogle(geminiRequest());
  assert.equal(out.contents.length, 1);
  assert.equal(out.contents[0].role, "user");
  const text = out.contents[0].parts[0].text;
  assert.ok(text.startsWith("<USER_REQUEST>"), "opening tag");
  assert.ok(text.endsWith("</USER_REQUEST>"), "closing tag");
  assert.ok(text.includes("hi"));
});

test("gemini assistant text is never wrapped", () => {
  const out = convertAnthropicToGoogle(
    geminiRequest({
      messages: [{ role: "assistant", content: "working on it" }],
    }),
  );
  assert.equal(out.contents[0].role, "model");
  assert.equal(out.contents[0].parts[0].text, "working on it");
});

test("gemini pure tool-result message becomes a model-role turn", () => {
  const out = convertAnthropicToGoogle(
    geminiRequest({
      messages: [
        { role: "user", content: "write f" },
        {
          role: "assistant",
          content: [
            { type: "tool_use", id: "call_1", name: "Write", input: {} },
          ],
        },
        {
          role: "user",
          content: [{ type: "tool_result", tool_use_id: "call_1", content: "ok" }],
        },
      ],
    }),
  );
  assert.equal(out.contents[0].role, "user");
  assert.equal(out.contents[1].role, "model");
  assert.equal(out.contents[2].role, "model");
  assert.ok(out.contents[2].parts[0].functionResponse);
});

test("gemini mixed text+result user message keeps user role", () => {
  const out = convertAnthropicToGoogle(
    geminiRequest({
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: "note" },
            { type: "tool_result", tool_use_id: "call_1", content: "ok" },
          ],
        },
      ],
    }),
  );
  assert.equal(out.contents[0].role, "user");
});

test("claude path is untouched: no wrap, user-role results", () => {
  const out = convertAnthropicToGoogle({
    model: "claude-sonnet-4-6",
    messages: [
      { role: "user", content: "hi" },
      {
        role: "user",
        content: [{ type: "tool_result", tool_use_id: "call_1", content: "ok" }],
      },
    ],
  });
  assert.equal(out.contents[0].parts[0].text, "hi");
  assert.equal(out.contents[1].role, "user");
  assert.ok(out.contents[1].parts[0].functionResponse);
});
