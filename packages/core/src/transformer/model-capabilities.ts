/**
 * Model-name capability checks for provider transformers.
 *
 * Upstream model registries churn (models appear, disappear, and get
 * renamed — we have watched it happen twice), so every name-based rule
 * lives here in one place with the reason it exists. When a rule stops
 * matching reality, this file is the only place that needs an update.
 * Long term these should become capability probes against live `/models`
 * endpoints; until then, centralized heuristics beat scattered ones.
 */

/**
 * Models served by OpenCode over OpenAI chat-completions rather than the
 * Responses API. Prefix families that only expose `/chat/completions`
 * on the Zen/Go endpoints.
 */
export function isOpencodeChatCompletionsModel(model: string): boolean {
  return (
    model.startsWith("deepseek-") ||
    model.startsWith("kimi-") ||
    model.startsWith("glm-") ||
    model.startsWith("mimo-") ||
    model.startsWith("qwen-") ||
    model.startsWith("minimax-")
  );
}

/**
 * Models billed (or rather, not billed) on the free Zen tier. Free models
 * route to `zen/v1` with the Zen key; everything else routes to `zen/go/v1`
 * with the Go key. `big-pickle` is the free-tier alias that carries no
 * `-free` suffix.
 */
export function isOpencodeFreeModel(model: string): boolean {
  const normalized = model.trim().toLowerCase();
  return normalized === "big-pickle" || normalized.endsWith("-free");
}

/**
 * Models Copilot serves over the Responses API (`/v1/responses`) rather
 * than the Messages API (`/v1/messages`). Covers GPT-family, OpenAI
 * reasoning (`o1`/`o3`/`o4`), and DeepSeek models on Copilot.
 */
export function isCopilotGPTModel(model: string): boolean {
  if (!model) return true;
  const lower = model.toLowerCase();
  return ["gpt-", "o1", "o3", "o4", "deepseek"].some((prefix) =>
    lower.startsWith(prefix),
  );
}

/**
 * Claude-family models on Copilot. These always use the Messages API with
 * Anthropic-format bodies, regardless of what else the catalog lists.
 */
export function isCopilotClaudeModel(model: string): boolean {
  return (model || "").toLowerCase().startsWith("claude");
}
