import assert from "node:assert/strict";
import test from "node:test";
import { CodexTransformer } from "@ccr/core/transformer/codex.transformer.ts";
import {
  isCopilotGPTModel,
  isOpencodeChatCompletionsModel,
  isOpencodeFreeModel,
} from "@ccr/core/transformer/model-capabilities.ts";

test("capability checks route models to the right protocol", () => {
  assert.equal(isOpencodeChatCompletionsModel("deepseek-v4-flash"), true);
  assert.equal(isOpencodeChatCompletionsModel("qwen-2.5-72b"), true);
  assert.equal(isOpencodeChatCompletionsModel("glm-5"), true);
  // Note: "qwen3.5-plus" does NOT carry the "qwen-" prefix, so it takes
  // the default OpenAI-body path (same payload today, documented here).
  assert.equal(isOpencodeChatCompletionsModel("qwen3.5-plus"), false);
  assert.equal(isOpencodeChatCompletionsModel("muse-spark-1.3-contributor"), false);

  assert.equal(isOpencodeFreeModel("deepseek-v4-flash-free"), true);
  assert.equal(isOpencodeFreeModel("big-pickle"), true);
  assert.equal(isOpencodeFreeModel("BIG-PICKLE"), true);
  assert.equal(isOpencodeFreeModel("deepseek-v4-flash"), false);

  assert.equal(isCopilotGPTModel("gpt-5-mini"), true);
  assert.equal(isCopilotGPTModel("o3"), true);
  assert.equal(isCopilotGPTModel("claude-sonnet-4.5"), false);
  assert.equal(isCopilotGPTModel(""), true);
});

test("echoed thinking blocks do not break request translation", () => {
  const t = new CodexTransformer();
  const body = t.transformRequestBody({
    model: "gpt-5.6-luna",
    stream: false,
    messages: [
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: [
          { type: "thinking", thinking: "Considering", signature: "123" },
          { type: "text", text: "Hello" },
        ],
      },
    ],
  });

  const assistantItems = body.input.filter((i) => i.role === "assistant");
  assert.equal(assistantItems.length, 1);
  assert.deepEqual(assistantItems[0].content, [{ type: "output_text", text: "Hello" }]);
});
