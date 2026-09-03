const STORE = "~/.claude-code-router/last-model.json";
const AUTO_MODELS = new Set(["claude-sonnet-5", "claude-sonnet-5-20241022", "claude-haiku-4-5", "claude-opus-4-5"]);

if (AUTO_MODELS.has(input.model)) {
  try {
    const data = await api.fs.readJson(STORE);
    const key = input.apiKeyId || "default";
    const last = data[key] || (input.sessionId ? data[input.sessionId] : undefined) || data["default"];
    if (last && !AUTO_MODELS.has(last)) {
      return { model: last };
    }
  } catch {}
  return null;
}

// store non-auto model as current pick
if (input.model && !AUTO_MODELS.has(input.model)) {
  try {
    let data = {};
    try { data = await api.fs.readJson(STORE); } catch {}
    const key = input.apiKeyId || "default";
    data[key] = input.model;
    if (input.sessionId) data[input.sessionId] = input.model;
    data["default"] = input.model;
    await api.fs.writeJson(STORE, data);
  } catch {}
}
return null;
