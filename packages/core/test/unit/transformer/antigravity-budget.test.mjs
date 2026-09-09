import assert from "node:assert/strict";
import test from "node:test";
import { convertAnthropicToGoogle } from "@ccr/core/transformer/antigravity/request-converter.ts";

function request(overrides = {}) {
  return {
    model: "gemini-3.8-flash-medium",
    messages: [{ role: "user", content: "hi" }],
    ...overrides,
  };
}

test("gemini thinking budget leaves headroom under the output cap", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 32000, thinking: { budget_tokens: 32000 } }),
  );
  assert.equal(out.generationConfig.maxOutputTokens, 16384);
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 12288);
});

test("explicit small thinking budgets pass through untouched", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 32000, thinking: { budget_tokens: 10000 } }),
  );
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 10000);
});

test("missing max_tokens leaves budget at the coherent default", () => {
  const out = convertAnthropicToGoogle(request({}));
  assert.equal(out.generationConfig.maxOutputTokens, undefined);
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 24576);
});

test("tiny max_tokens floors thinking at 1024 instead of going negative", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 16, thinking: { budget_tokens: 32000 } }),
  );
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 1024);
});
