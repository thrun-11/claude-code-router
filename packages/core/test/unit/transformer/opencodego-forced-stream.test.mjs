import assert from "node:assert/strict";
import test from "node:test";
import {
  OpencodeGoTransformer,
  getOpencodeUserAgent,
  resetOpencodeUserAgentForTests,
} from "@ccr/core/transformer/opencodego.transformer.ts";

const CHAT_URL = "https://opencode.ai/inference/v1/chat/completions";
const RESP_URL = "https://opencode.ai/inference/v1/responses";

function sseResponse(lines) {
  return new Response(lines.join("\n") + "\n", {
    status: 200,
    headers: { "content-type": "text/event-stream" },
  });
}

function withStubFetch(handler) {
  const calls = [];
  const prev = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    calls.push({ url: String(url), init });
    return handler(String(url), init, calls.length);
  };
  return {
    calls,
    restore() {
      globalThis.fetch = prev;
    },
  };
}

test("harness UA defaults to the full captured triple", () => {
  resetOpencodeUserAgentForTests();
  const ua = getOpencodeUserAgent({});
  assert.match(ua, /^opencode\/\d+\.\d+\.\d+ /);
  assert.ok(ua.includes("ai-sdk/provider-utils/"), "ai-sdk token present");
  assert.ok(ua.includes("runtime/bun/"), "runtime token present");
});

test("harness UA differs per endpoint kind (chat 4.0.23, responses 4.0.40)", () => {
  resetOpencodeUserAgentForTests();
  assert.ok(getOpencodeUserAgent({}, "chat").includes("ai-sdk/provider-utils/4.0.23"));
  assert.ok(getOpencodeUserAgent({}, "responses").includes("ai-sdk/provider-utils/4.0.40"));
  assert.ok(getOpencodeUserAgent({}).includes("ai-sdk/provider-utils/4.0.40"), "default is responses");
});

test("harness UA honors env/config overrides", () => {
  resetOpencodeUserAgentForTests();
  process.env["OPENCODE_HARNESS_UA"] = "custom-ua/1.0";
  assert.equal(getOpencodeUserAgent({}), "custom-ua/1.0");
  assert.equal(getOpencodeUserAgent({ opencode_user_agent: "cfg-ua/2.0" }), "cfg-ua/2.0");
  delete process.env["OPENCODE_HARNESS_UA"];
  resetOpencodeUserAgentForTests();
});

test("free model non-stream downstream forces stream:true upstream and collapses SSE", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"id":"c1","model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"content":"Hel"}}]}',
    'data: {"id":"c1","model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"content":"lo"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
    "data: [DONE]",
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  try {
    const res = await t.fetchEndpoint(
      { model: "mimo-v2.5-free", messages: [{ role: "user", content: "hi" }], stream: false },
      CHAT_URL,
      {},
      false,
      {},
    );
    const upstreamCalls = stub.calls.filter((c) => c.url === CHAT_URL);
    assert.equal(upstreamCalls.length, 1);
    const sentBody = JSON.parse(upstreamCalls[0].init.body);
    assert.equal(sentBody.stream, true, "upstream forced to stream");
    assert.deepEqual(sentBody.stream_options, { include_usage: true });
    const sentHeaders = upstreamCalls[0].init.headers;
    assert.ok(sentHeaders["User-Agent"].includes("runtime/bun/"));
    assert.match(sentHeaders["x-opencode-session"], /^ses_/);
    assert.match(sentHeaders["x-opencode-request"], /^msg_/);
    assert.notEqual(sentHeaders["x-opencode-request"], sentHeaders["x-opencode-session"]);
    assert.equal(sentHeaders["x-opencode-project"], "global");

    assert.equal(res.status, 200);
    const data = await res.json();
    assert.equal(data.content[0].text, "Hello");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("free responses model non-stream collapses response.completed", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.created","response":{"id":"r1"}}',
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[{"type":"message","content":[{"type":"output_text","text":"Hi there"}]}],"usage":{"input_tokens":3,"output_tokens":2}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  try {
    const res = await t.fetchEndpoint(
      { model: "muse-spark-1.3-contributor-free", input: [], stream: false },
      RESP_URL,
      {},
      false,
      {},
    );
    const upstreamCalls = stub.calls.filter((c) => c.url === RESP_URL);
    assert.equal(upstreamCalls.length, 1);
    assert.equal(JSON.parse(upstreamCalls[0].init.body).stream, true);
    const data = await res.json();
    assert.equal(data.content[0].text, "Hi there");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("FreeTierError triggers UA refresh and a single retry", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const denied = new Response(
    JSON.stringify({ type: "error", error: { type: "FreeTierError", message: "x" } }),
    { status: 403, headers: { "content-type": "application/json" } },
  );
  const sse = sseResponse([
    'data: {"id":"c1","model":"mimo-v2.5-free","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    const upstreamCalls = stub.calls.filter((c) => c.url === CHAT_URL).length;
    return upstreamCalls === 1 ? denied : sse;
  });
  try {
    const res = await t.fetchEndpoint(
      { model: "mimo-v2.5-free", messages: [], stream: false },
      CHAT_URL,
      {},
      false,
      {},
    );
    const upstreamCalls = stub.calls.filter((c) => c.url === CHAT_URL);
    assert.equal(upstreamCalls.length, 2, "retried once after FreeTierError");
    assert.equal(res.status, 200);
    const data = await res.json();
    assert.equal(data.content[0].text, "ok");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("[1m] gateway suffix is stripped before endpoint selection and upstream body", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[{"type":"message","content":[{"type":"output_text","text":"sfx"}]}],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  try {
    const res = await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free[1m]", messages: [], stream: false },
      {},
      {},
      {},
    );
    const upstreamCalls = stub.calls.filter((c) => c.url === RESP_URL);
    assert.equal(upstreamCalls.length, 1, "routed to zen responses, not go");
    assert.ok(!stub.calls.some((c) => c.url.includes("/go/")), "never hits the go endpoint");
    assert.equal(JSON.parse(upstreamCalls[0].init.body).model, "muse-spark-1.3-contributor-free");
    assert.equal((await res.json()).content[0].text, "sfx");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("free responses sends native tools + inference shape to /inference/v1", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[{"type":"message","content":[{"type":"output_text","text":"hi"}]}],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  const tools = [
    { type: "function", function: { name: "Get", description: "read", parameters: { type: "object" } } },
    { type: "function", function: { name: "Grep", description: "search", parameters: { type: "object" } } },
  ];
  try {
    const res = await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false, tools },
      {},
      {},
      {},
    );
    const call = stub.calls.find((c) => c.url === RESP_URL);
    assert.ok(call, "routed to inference responses");
    const sent = JSON.parse(call.init.body);
    const names = sent.tools.map((x) => x.name);
    assert.ok(names.includes("Get") && names.includes("Grep"), "caller tools preserved");
    for (const fp of ["bash", "edit", "glob", "grep", "read", "skill", "task", "todowrite", "webfetch", "websearch", "write"]) {
      assert.ok(names.includes(fp), `fingerprint ${fp} injected`);
    }
    // Gate validates definitions byte-verbatim (dummy schemas 403 since 2026-10-15).
    const bash = sent.tools.find((x) => x.name === "bash");
    assert.ok(bash.description.length > 1000, "verbatim official description");
    assert.equal(bash.strict, false);
    assert.equal(sent.tool_choice, "auto");
    assert.equal(sent.store, false);
    assert.equal(sent.max_output_tokens, 32000);
    assert.deepEqual(sent.include, ["reasoning.encrypted_content"]);
    assert.match(sent.prompt_cache_key, /^ses_/, "prompt_cache_key = ses id");
    const headers = call.init.headers;
    assert.match(headers["x-opencode-session"], /^ses_/);
    assert.equal(headers["x-opencode-session-id"], headers["x-opencode-session"], "both session headers");
    assert.match(headers["x-opencode-request"], /^msg_/);
    assert.ok(headers["User-Agent"].includes("ai-sdk/provider-utils/4.0.40"), "responses UA suffix");
    assert.equal(res.status, 200);
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("free responses text-protocol tool blocks become tool_use", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[{"type":"message","content":[{"type":"output_text","text":"done ```tool\\n{\\"name\\": \\"Bash\\", \\"arguments\\": {\\"command\\": \\"ls\\"}}```"}]}],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  try {
    const res = await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false },
      {},
      {},
      {},
    );
    const data = await res.json();
    const toolUse = data.content.find((b) => b.type === "tool_use");
    assert.ok(toolUse, "tool_use extracted");
    assert.equal(toolUse.name, "Bash");
    assert.equal(data.stop_reason, "tool_use");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("chat-family free models route to inference chat, spark to inference responses", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const chatSse = sseResponse([
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}',
  ]);
  const respSse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[{"type":"message","content":[{"type":"output_text","text":"ok"}]}],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return url.includes("/responses") ? respSse.clone() : chatSse.clone();
  });
  try {
    await t.sendRequest(
      { model: "space-bunny-free", messages: [{ role: "user", content: "hi" }], stream: false },
      {},
      {},
      {},
    );
    await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false },
      {},
      {},
      {},
    );
    assert.ok(stub.calls.some((c) => c.url === RESP_URL), "spark routed to inference responses");
    assert.ok(!stub.calls.some((c) => c.url.includes("/go/")), "never hits go");
    assert.ok(!stub.calls.some((c) => c.url.includes("/zen/")), "never hits zen for free");
    const chatSent = JSON.parse(stub.calls.find((c) => c.url === CHAT_URL).init.body);
    assert.deepEqual(
      chatSent.tools.map((x) => x.function.name).sort(),
      ["bash", "edit", "glob", "grep", "read", "skill", "task", "todowrite", "webfetch", "websearch", "write"],
    );
    const chatBash = chatSent.tools.find((x) => x.function.name === "bash");
    assert.ok(chatBash.function.description.length > 1000, "verbatim official chat definition");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("ses_/msg_ ids use exact CLI SortableID shape and are unique", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse.clone();
  });
  try {
    await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false },
      {},
      {},
      {},
    );
    await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false },
      {},
      {},
      {},
    );
    const sessions = stub.calls.filter((c) => c.url === RESP_URL).map((c) => c.init.headers["x-opencode-session"]);
    assert.equal(sessions.length, 2);
    for (const ses of sessions) assert.match(ses, /^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$/);
    assert.notEqual(sessions[0], sessions[1], "fresh session per call without caller context");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("paid responses model gets no decoys", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  const tools = [
    { type: "function", function: { name: "Get", description: "read", parameters: {} } },
  ];
  try {
    await t.sendRequest(
      { model: "muse-spark-1.3", messages: [], stream: true, tools },
      {},
      {},
      {},
    );
    const sent = JSON.parse(stub.calls.find((c) => c.url.includes("/responses")).init.body);
    assert.deepEqual(sent.tools.map((x) => x.name), ["Get"]);
    assert.equal(sent.tool_choice, undefined);
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("free responses without caller tools still sends fingerprint set", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  try {
    await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false },
      {},
      {},
      {},
    );
    const sent = JSON.parse(stub.calls.find((c) => c.url === RESP_URL).init.body);
    assert.deepEqual(sent.tools.map((x) => x.name).sort(), ["bash", "edit", "glob", "grep", "read", "skill", "task", "todowrite", "webfetch", "websearch", "write"]);
    assert.equal(sent.tool_choice, "auto");
    assert.match(sent.prompt_cache_key, /^ses_/);
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("free responses keeps function tools incl. web_search mapping", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  // Shape produced by convertAnthropicToolsToUnified for Anthropic server tools.
  const tools = [
    { type: "function", function: { name: "web_search", description: "", parameters: undefined } },
    { type: "function", function: { name: "Get", description: "read", parameters: { type: "object" } } },
  ];
  try {
    await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false, tools },
      {},
      {},
      {},
    );
    const sent = JSON.parse(stub.calls.find((c) => c.url === RESP_URL).init.body);
    const names = sent.tools.map((x) => x.name || x.type);
    assert.ok(names.includes("Get"), "caller tool preserved");
    assert.ok(names.includes("bash"), "fingerprint added");
    assert.equal(sent.tool_choice, "auto");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("injected fingerprint calls are remapped or cloaked, caller calls preserved", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"type":"response.completed","response":{"id":"r1","model":"muse-spark-1.3-contributor-free","output":[{"type":"function_call","call_id":"c1","name":"bash","arguments":"{}"},{"type":"function_call","call_id":"c2","name":"Bash","arguments":"{\\"command\\":\\"ls\\"}"},{"type":"function_call","call_id":"c3","name":"glob","arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":1}}}',
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse;
  });
  const tools = [
    { type: "function", function: { name: "Bash", description: "run", parameters: { type: "object" } } },
  ];
  try {
    const res = await t.sendRequest(
      { model: "muse-spark-1.3-contributor-free", messages: [{ role: "user", content: "hi" }], stream: false, tools },
      {},
      {},
      {},
    );
    const data = await res.json();
    const toolUses = data.content.filter((b) => b.type === "tool_use");
    assert.equal(toolUses.length, 2, "remappable injected call + caller call survive");
    assert.deepEqual(toolUses.map((b) => b.name), ["Bash", "Bash"], "bash remapped to caller spelling, glob cloaked");
    assert.equal(data.stop_reason, "tool_use");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("free chat placeholder call is remapped to caller tool without a retry", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"bash","arguments":"{\\"command\\":\\"ls\\"}"}}]}}]}',
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}',
    "data: [DONE]",
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse.clone();
  });
  try {
    const res = await t.sendRequest(
      {
        model: "space-bunny-free",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
        tools: [{ type: "function", function: { name: "Bash", description: "run", parameters: { type: "object" } } }],
      },
      {},
      {},
      {},
    );
    const data = await res.json();
    const toolUse = data.content.find((b) => b.type === "tool_use");
    assert.ok(toolUse, "tool_use surfaced");
    assert.equal(toolUse.name, "Bash", "remapped to caller spelling");
    assert.equal(toolUse.input.command, "ls");
    assert.equal(data.stop_reason, "tool_use");
    assert.equal(stub.calls.filter((c) => c.url === CHAT_URL).length, 1, "no retry for remappable calls");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("unmappable placeholder call triggers internal retry with tool feedback", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const firstSse = sseResponse([
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,"id":"call_todo","type":"function","function":{"name":"todowrite","arguments":"{\\"todos\\":[]}"}}]}}]}',
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}',
    "data: [DONE]",
  ]);
  const secondSse = sseResponse([
    'data: {"id":"c2","model":"space-bunny-free","choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,"id":"call_bash","type":"function","function":{"name":"bash","arguments":"{\\"command\\":\\"pwd\\"}"}}]}}]}',
    'data: {"id":"c2","model":"space-bunny-free","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}',
    "data: [DONE]",
  ]);
  let upstreamCall = 0;
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    upstreamCall += 1;
    return upstreamCall === 1 ? firstSse.clone() : secondSse.clone();
  });
  try {
    const res = await t.sendRequest(
      {
        model: "space-bunny-free",
        messages: [{ role: "user", content: "start phase 0" }],
        stream: false,
        tools: [{ type: "function", function: { name: "Bash", description: "run", parameters: { type: "object" } } }],
      },
      {},
      {},
      {},
    );
    const data = await res.json();
    const toolUse = data.content.find((b) => b.type === "tool_use");
    assert.ok(toolUse, "retry produced a real tool call");
    assert.equal(toolUse.name, "Bash");
    assert.equal(data.stop_reason, "tool_use");
    const chatCalls = stub.calls.filter((c) => c.url === CHAT_URL);
    assert.equal(chatCalls.length, 2, "exactly one internal retry");
    const retryBody = JSON.parse(chatCalls[1].init.body);
    const feedback = retryBody.messages[retryBody.messages.length - 1];
    const echoedAssistant = retryBody.messages[retryBody.messages.length - 2];
    assert.equal(echoedAssistant.role, "assistant");
    assert.equal(echoedAssistant.tool_calls[0].function.name, "todowrite");
    assert.equal(feedback.role, "tool");
    assert.equal(feedback.tool_call_id, "call_todo");
    assert.match(feedback.content, /todowrite.*not available/);
    assert.match(feedback.content, /Bash/);
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("placeholder retry is bounded and degrades without crashing", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const sse = sseResponse([
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"finish_reason":null,"delta":{"tool_calls":[{"index":0,"id":"call_todo","type":"function","function":{"name":"todowrite","arguments":"{}"}}]}}]}',
    'data: {"id":"c1","model":"space-bunny-free","choices":[{"index":0,"finish_reason":"tool_calls","delta":{}}]}',
    "data: [DONE]",
  ]);
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return sse.clone();
  });
  try {
    const res = await t.sendRequest(
      {
        model: "space-bunny-free",
        messages: [{ role: "user", content: "hi" }],
        stream: false,
        tools: [{ type: "function", function: { name: "Bash", description: "run", parameters: { type: "object" } } }],
      },
      {},
      {},
      {},
    );
    const data = await res.json();
    assert.equal(stub.calls.filter((c) => c.url === CHAT_URL).length, 3, "1 attempt + 2 bounded retries");
    assert.ok(!data.content.some((b) => b.type === "tool_use"), "placeholder calls stay cloaked");
    assert.equal(data.stop_reason, "end_turn");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});

test("paid model non-stream is untouched (no forcing)", async () => {
  resetOpencodeUserAgentForTests();
  const t = new OpencodeGoTransformer();
  const json = new Response(
    JSON.stringify({
      id: "c9",
      model: "grok-4.6",
      choices: [{ message: { role: "assistant", content: "paid" }, finish_reason: "stop" }],
      usage: { prompt_tokens: 1, completion_tokens: 1 },
    }),
    { status: 200, headers: { "content-type": "application/json" } },
  );
  const stub = withStubFetch((url) => {
    if (url.includes("registry.npmjs.org")) return new Response("no", { status: 500 });
    return json;
  });
  try {
    const res = await t.fetchEndpoint(
      { model: "grok-4.6", messages: [], stream: false },
      "https://opencode.ai/zen/go/v1/chat/completions",
      {},
      false,
      {},
    );
    const upstreamCalls = stub.calls.filter((c) => c.url.includes("opencode.ai"));
    assert.equal(JSON.parse(upstreamCalls[0].init.body).stream, false);
    const data = await res.json();
    assert.equal(data.content[0].text, "paid");
  } finally {
    stub.restore();
    resetOpencodeUserAgentForTests();
  }
});
