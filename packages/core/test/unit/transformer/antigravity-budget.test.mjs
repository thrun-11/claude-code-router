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

test("explicit thinking budget leaves headroom under the output cap", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 32000, thinking: { budget_tokens: 32000 } }),
  );
  assert.equal(out.generationConfig.maxOutputTokens, 32000);
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 24576);
});

test("explicit small thinking budgets pass through untouched", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 32000, thinking: { budget_tokens: 10000 } }),
  );
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 10000);
});

test("missing budget sends dynamic -1 like the official CLI", () => {
  const out = convertAnthropicToGoogle(request({}));
  assert.equal(out.generationConfig.maxOutputTokens, undefined);
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, -1);
});

test("large max_tokens caps at the CLI-parity 65536 ceiling", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 128000, thinking: { budget_tokens: 32000 } }),
  );
  assert.equal(out.generationConfig.maxOutputTokens, 65536);
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 24576);
});

test("tiny max_tokens floors thinking at 1024 instead of going negative", () => {
  const out = convertAnthropicToGoogle(
    request({ max_tokens: 16, thinking: { budget_tokens: 32000 } }),
  );
  assert.equal(out.generationConfig.thinkingConfig.thinkingBudget, 1024);
});
