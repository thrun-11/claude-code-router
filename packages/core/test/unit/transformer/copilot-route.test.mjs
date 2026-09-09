import assert from "node:assert/strict";
import test from "node:test";
import { selectCopilotEndpoint } from "@ccr/core/services/copilot/model-catalog.ts";
import { CopilotTransformer } from "@ccr/core/transformer/copilot.transformer.ts";

const catalog = [
  { id: "gpt-5.4-mini", supportedEndpoints: ["/responses", "ws:/responses"], policyState: "enabled" },
  { id: "gpt-4o", supportedEndpoints: undefined, policyState: undefined },
  { id: "kimi-k3", supportedEndpoints: ["/chat/completions"], policyState: "enabled" },
  { id: "claude-haiku-4.5", supportedEndpoints: ["/chat/completions", "/v1/messages"], policyState: "enabled" },
];

test("endpoint follows the live catalog, not the name", () => {
  assert.equal(selectCopilotEndpoint("gpt-5.4-mini", catalog), "/v1/responses");
  assert.equal(selectCopilotEndpoint("kimi-k3", catalog), "/chat/completions");
  // Claude-family always uses the Messages API regardless of catalog.
  assert.equal(selectCopilotEndpoint("claude-haiku-4.5", catalog), "/v1/messages");
});

test("missing catalog entries take chat completions, never Messages with non-Anthropic models", () => {
  assert.equal(selectCopilotEndpoint("gpt-4o", catalog), "/chat/completions");
  assert.equal(selectCopilotEndpoint("gpt-4o", null), "/chat/completions");
  assert.equal(selectCopilotEndpoint("claude-opus-9", null), "/v1/messages");
  assert.equal(selectCopilotEndpoint("kimi-k9", null), "/chat/completions");
});

test("chat request builder produces OpenAI chat completions payloads", () => {
  const t = new CopilotTransformer();
  const body = t.buildChatRequest({
    model: "gpt-4o",
    stream: false,
    temperature: 0.5,
    max_tokens: 64,
    messages: [
      { role: "system", content: "sys" },
      { role: "user", content: "hi" },
      {
        role: "assistant",
        content: "calling",
        tool_calls: [{ id: "c1", function: { name: "f", arguments: { a: 1 } } }],
      },
      { role: "tool", tool_call_id: "c1", content: "done" },
    ],
    tools: [{ function: { name: "f", description: "d", parameters: { type: "object" } } }],
  });

  assert.equal(body.model, "gpt-4o");
  assert.deepEqual(body.messages[0], { role: "system", content: "sys" });
  assert.deepEqual(body.messages[1], { role: "user", content: "hi" });
  assert.deepEqual(body.messages[2].tool_calls, [
    { id: "c1", type: "function", function: { name: "f", arguments: '{"a":1}' } },
  ]);
  assert.deepEqual(body.messages[3], { role: "tool", tool_call_id: "c1", content: "done" });
  assert.deepEqual(body.tools, [
    { type: "function", function: { name: "f", description: "d", parameters: { type: "object" } } },
  ]);
  assert.equal(body.temperature, 0.5);
  assert.equal(body.max_tokens, 64);
});
