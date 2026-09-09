import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";

/**
 * Shared reader/writer for the agy weekly-bucket cache consumed by the
 * quota-guard route script and the statusline bar.
 *
 * Freshness is maintained two ways: the statusline refreshes in the
 * background on render, and the gateway triggers a refresh (below) on
 * antigravity traffic so API-driven sessions stay covered too.
 */

export const AGY_USAGE_CACHE_FILE = "/tmp/antigravity-agy-usage";
export const AGY_USAGE_CACHE_TTL_MS = 5 * 60 * 1000;
const AGY_REFRESH_LOCK_FILE = "/tmp/antigravity-agy-usage.lock";
const AGY_REFRESH_LOCK_TTL_MS = 60 * 1000;

export interface AgyBucketLevels {
  geminiRemainingPct: number | null;
  claudeRemainingPct: number | null;
  fetchedAt: number;
}

function parseBucketLines(text: string): Omit<AgyBucketLevels, "fetchedAt"> {
  let geminiRemainingPct: number | null = null;
  let claudeRemainingPct: number | null = null;
  for (const line of text.split("\n")) {
    const cells = line.split("\t").map((c) => c.trim());
    if (cells.length < 3) continue;
    const pct = Number.parseFloat(cells[2].replace("%", ""));
    if (!Number.isFinite(pct)) continue;
    if (/^gemini/i.test(cells[0])) geminiRemainingPct = pct;
    else if (/^claude/i.test(cells[0])) claudeRemainingPct = pct;
  }
  return { geminiRemainingPct, claudeRemainingPct };
}

export function readAgyBucketCache(): AgyBucketLevels | null {
  try {
    const stat = fs.statSync(AGY_USAGE_CACHE_FILE);
    if (Date.now() - stat.mtimeMs > AGY_USAGE_CACHE_TTL_MS) return null;
    const levels = parseBucketLines(fs.readFileSync(AGY_USAGE_CACHE_FILE, "utf8"));
    if (levels.geminiRemainingPct === null && levels.claudeRemainingPct === null) {
      return null;
    }
    return { ...levels, fetchedAt: stat.mtimeMs };
  } catch {
    return null;
  }
}

let resolvedBinary: string | null | undefined;

function resolveAgyBinary(): string | null {
  if (resolvedBinary !== undefined) return resolvedBinary;
  const candidates = [
    process.env.AGY_BIN,
    `${os.homedir()}/.local/bin/agy`,
    "agy",
  ];
  for (const candidate of candidates) {
    if (!candidate) continue;
    try {
      // -p "/usage" doubles as an availability probe further down; here we
      // only need an executable file. PATH lookup covers the bare name.
      const found =
        candidate === "agy" ? findOnPath(candidate) : candidate;
      if (found) {
        fs.accessSync(found, fs.constants.X_OK);
        resolvedBinary = found;
        return found;
      }
    } catch {
      // Not usable; try the next candidate.
    }
  }
  resolvedBinary = null;
  return null;
}

function findOnPath(name: string): string | null {
  const pathEnv = process.env.PATH || "/usr/bin:/bin:/usr/local/bin";
  for (const dir of pathEnv.split(":")) {
    const full = `${dir}/${name}`;
    try {
      fs.accessSync(full, fs.constants.X_OK);
      return full;
    } catch {
      // Keep searching.
    }
  }
  return null;
}

function refreshLockFresh(): boolean {
  try {
    const stat = fs.statSync(AGY_REFRESH_LOCK_FILE);
    return Date.now() - stat.mtimeMs < AGY_REFRESH_LOCK_TTL_MS;
  } catch {
    return false;
  }
}

/**
 * Fire-and-forget refresh of the bucket cache. Never blocks the request
 * path: singleflight-guarded by lockfile, detached child, atomic rename
 * only on valid output.
 */
export function refreshAgyBucketCacheInBackground(logger?: {
  warn?: (...args: unknown[]) => void;
}): void {
  try {
    if (refreshLockFresh()) return;
    fs.writeFileSync(AGY_REFRESH_LOCK_FILE, String(Date.now()));
  } catch {
    return;
  }

  let binary: string | null = null;
  try {
    binary = resolveAgyBinary();
  } catch {
    binary = null;
  }
  if (!binary) return;

  try {
    // Shell wrapper so validation + atomic rename happen in the child;
    // the gateway never waits on it (detached + unref).
    const quoted = (s: string) => `'${s.replace(/'/g, `'\\''`)}'`;
    const tmpFile = `${AGY_USAGE_CACHE_FILE}.tmp`;
    const cmd =
      `${quoted(binary)} -p "/usage" > ${quoted(tmpFile)} 2>/dev/null && ` +
      `grep -q "Weekly Limit Remaining" ${quoted(tmpFile)} 2>/dev/null && ` +
      `mv ${quoted(tmpFile)} ${quoted(AGY_USAGE_CACHE_FILE)}`;
    const child = spawn("sh", ["-c", cmd], { detached: true, stdio: "ignore" });
    child.unref();
    // Best-effort cleanup of the child handle; the detached process
    // continues independently.
    child.on("error", () => undefined);
  } catch (error) {
    logger?.warn?.(`[Antigravity] agy refresh spawn failed: ${error}`);
  }
}

export function parseBucketLinesForTest(text: string): Omit<AgyBucketLevels, "fetchedAt"> {
  return parseBucketLines(text);
}
