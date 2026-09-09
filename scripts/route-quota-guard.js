// Quota guard: reroute Antigravity requests when weekly buckets run dry.
//
// Reads the same caches as the statusline bar (no network in the request
// path): agy weekly buckets first (family-aware), then IDE credit cache.
// Below LOW_REMAIN_PCT the request is rewritten to FALLBACK_MODEL.
// Missing or stale telemetry fails OPEN — never block traffic for lack
// of quota data.
//
// Installed to ~/.claude-code-router/route-quota-guard.js and wired as a
// Router.rules script entry (see route-auto-follow.js pattern).
const LOW_REMAIN_PCT = 10;
const FALLBACK_MODEL = "opencode/qwen3.7-plus";
const AGY_CACHE = "/tmp/antigravity-agy-usage";
const AGY_TTL_MS = 5 * 60 * 1000;
const CREDITS_CACHE = "/tmp/antigravity-credits.json";
const CREDITS_TTL_MS = 120 * 1000;

const model = String(input.model || "");
if (!model.toLowerCase().startsWith("antigravity/")) {
  return null;
}

async function freshText(path, ttlMs) {
  try {
    const st = await api.fs.stat(path);
    if (Date.now() - Date.parse(st.modifiedAt) > ttlMs) return null;
    return await api.fs.readText(path);
  } catch {
    return null;
  }
}

// 1) agy weekly buckets (authoritative, family-aware).
try {
  const text = await freshText(AGY_CACHE, AGY_TTL_MS);
  if (text) {
    const wantClaude = model.toLowerCase().includes("claude");
    for (const line of text.split("\n")) {
      const cells = line.split("\t");
      if (cells.length < 3) continue;
      const label = cells[0].toLowerCase();
      const isBucket =
        (wantClaude && label.startsWith("claude")) ||
        (!wantClaude && label.startsWith("gemini"));
      if (!isBucket) continue;
      const pct = parseFloat(cells[2]);
      if (!Number.isFinite(pct)) continue;
      return pct < LOW_REMAIN_PCT ? { model: FALLBACK_MODEL } : null;
    }
  }
} catch {}

// 2) IDE credit cache (prompt credits remaining).
try {
  const text = await freshText(CREDITS_CACHE, CREDITS_TTL_MS);
  if (text) {
    const data = JSON.parse(text);
    if (
      typeof data.available === "number" &&
      typeof data.monthly === "number" &&
      data.monthly > 0
    ) {
      return (data.available / data.monthly) * 100 < LOW_REMAIN_PCT
        ? { model: FALLBACK_MODEL }
        : null;
    }
  }
} catch {}

// 3) No fresh telemetry: allow (fail-open).
return null;
