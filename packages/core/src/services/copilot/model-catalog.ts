import { promises as fs } from "node:fs";
import { join } from "node:path";
import { getModels } from "./api";
import { isCopilotClaudeModel } from "@ccr/core/transformer/model-capabilities";

export type CopilotEndpointKind = "/v1/responses" | "/v1/messages" | "/chat/completions";

export interface CopilotModelEndpointInfo {
  id: string;
  supportedEndpoints?: string[];
  policyState?: string;
}

const MODELS_CACHE_FILE = join(
  process.env.HOME!,
  ".claude-code-router",
  "copilot-models.json",
);
const MODELS_CACHE_TTL_MS = 60 * 60 * 1000;

let memoryCache: { fetchedAt: number; models: CopilotModelEndpointInfo[] } | null = null;

function normalizeCatalogEntry(entry: any): CopilotModelEndpointInfo | null {
  const id = typeof entry?.id === "string" ? entry.id : entry?.name;
  if (!id) return null;
  const supportedEndpoints = Array.isArray(entry?.supported_endpoints)
    ? entry.supported_endpoints.filter((e: unknown) => typeof e === "string")
    : undefined;
  const policyState =
    typeof entry?.policy?.state === "string" ? entry.policy.state : undefined;
  return { id, supportedEndpoints, policyState };
}

async function readFileCache(): Promise<CopilotModelEndpointInfo[] | null> {
  try {
    const stat = await fs.stat(MODELS_CACHE_FILE);
    if (Date.now() - stat.mtimeMs > MODELS_CACHE_TTL_MS) return null;
    const parsed = JSON.parse(await fs.readFile(MODELS_CACHE_FILE, "utf8"));
    if (!Array.isArray(parsed)) return null;
    const models = parsed
      .map(normalizeCatalogEntry)
      .filter((m): m is CopilotModelEndpointInfo => m !== null);
    return models.length > 0 ? models : null;
  } catch {
    return null;
  }
}

async function writeFileCache(models: CopilotModelEndpointInfo[]): Promise<void> {
  try {
    await fs.mkdir(join(process.env.HOME!, ".claude-code-router"), { recursive: true });
    const tmp = `${MODELS_CACHE_FILE}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(models), "utf-8");
    await fs.rename(tmp, MODELS_CACHE_FILE);
  } catch {
    // Cache is best-effort; live fetch already succeeded.
  }
}

/**
 * Live Copilot model catalog, cached in memory and on disk (1h TTL).
 * Returns null when the catalog cannot be fetched — callers fall back
 * to name heuristics.
 */
export async function loadCopilotModelCatalog(
  copilotToken: string,
  accountType: string,
): Promise<CopilotModelEndpointInfo[] | null> {
  if (memoryCache && Date.now() - memoryCache.fetchedAt < MODELS_CACHE_TTL_MS) {
    return memoryCache.models;
  }
  try {
    const live = await getModels(accountType, copilotToken);
    const models = (live as any[])
      .map(normalizeCatalogEntry)
      .filter((m): m is CopilotModelEndpointInfo => m !== null);
    if (models.length === 0) throw new Error("empty catalog");
    memoryCache = { fetchedAt: Date.now(), models };
    await writeFileCache(models);
    return models;
  } catch {
    const file = await readFileCache();
    if (file) {
      memoryCache = { fetchedAt: Date.now(), models: file };
      return file;
    }
    return null;
  }
}

/**
 * Pure endpoint selection. Claude-family models always use the Messages
 * API; everything else follows the live catalog (`supported_endpoints`)
 * when available. Models unknown to the catalog take chat completions:
 * it is the universal legacy API, while Responses-only models are always
 * new enough to appear in the catalog (refreshed hourly).
 */
export function selectCopilotEndpoint(
  model: string,
  catalog: CopilotModelEndpointInfo[] | null | undefined,
): CopilotEndpointKind {
  if (isCopilotClaudeModel(model)) {
    return "/v1/messages";
  }
  const entry = (catalog || []).find(
    (m) => m.id.toLowerCase() === model.toLowerCase(),
  );
  const endpoints = entry?.supportedEndpoints || [];
  // Catalog paths are unversioned ("/responses"); the gateway path is versioned.
  if (endpoints.some((e) => e.endsWith("/responses"))) {
    return "/v1/responses";
  }
  return "/chat/completions";
}
