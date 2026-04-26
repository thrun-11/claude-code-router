import * as os from "os";
import * as path from "path";

export const IDE_TYPE = {
  UNSPECIFIED: 0,
  JETSKI: 10,
  ANTIGRAVITY: 9,
  PLUGINS: 7,
};

export const PLATFORM = {
  UNSPECIFIED: 0,
  DARWIN_AMD64: 1,
  DARWIN_ARM64: 2,
  LINUX_AMD64: 3,
  LINUX_ARM64: 4,
  WINDOWS_AMD64: 5,
};

export const PLUGIN_TYPE = {
  UNSPECIFIED: 0,
  CLOUD_CODE: 1,
  GEMINI: 2,
};

function getPlatformEnum(): number {
  const platform = os.platform();
  const arch = os.arch();
  if (platform === "darwin") {
    return arch === "arm64" ? PLATFORM.DARWIN_ARM64 : PLATFORM.DARWIN_AMD64;
  } else if (platform === "linux") {
    return arch === "arm64" ? PLATFORM.LINUX_ARM64 : PLATFORM.LINUX_AMD64;
  } else if (platform === "win32") {
    return PLATFORM.WINDOWS_AMD64;
  }
  return PLATFORM.UNSPECIFIED;
}

export const CLIENT_METADATA = {
  ideType: IDE_TYPE.ANTIGRAVITY,
  platform: getPlatformEnum(),
  pluginType: PLUGIN_TYPE.GEMINI,
};

const ANTIGRAVITY_ENDPOINT_DAILY = "https://daily-cloudcode-pa.googleapis.com";
const ANTIGRAVITY_ENDPOINT_PROD = "https://cloudcode-pa.googleapis.com";

export const ANTIGRAVITY_ENDPOINTS = [
  ANTIGRAVITY_ENDPOINT_DAILY,
  ANTIGRAVITY_ENDPOINT_PROD,
];

export const ANTIGRAVITY_HEADERS = {
  "User-Agent": `antigravity/1.23.2 ${os.platform()}/${os.arch()}`,
  "Content-Type": "application/json",
  "X-Client-Name": "antigravity",
  "X-Client-Version": "1.23.2",
  "x-goog-api-client": "gl-node/18.18.2 fire/0.8.6 grpc/1.10.x",
};

export const DEFAULT_PROJECT_ID = "rising-fact-p41fc";
export const DEFAULT_PORT = 8080;

export const GEMINI_MAX_OUTPUT_TOKENS = 16384;
export const MIN_SIGNATURE_LENGTH = 50;
export const GEMINI_SKIP_SIGNATURE = "skip_thought_signature_validator";

export const TOKEN_REFRESH_INTERVAL_MS = 5 * 60 * 1000;
export const DEFAULT_COOLDOWN_MS = 10 * 1000;
export const MAX_RETRIES = 5;

export const MODEL_FALLBACK_MAP: Record<string, string> = {
  "gemini-3.1-pro-high": "claude-opus-4-6-thinking",
  "gemini-3.1-pro-low": "claude-sonnet-4-6",
  "gemini-3-flash": "claude-sonnet-4-6-thinking",
  "claude-opus-4-6-thinking": "gemini-3.1-pro-high",
  "claude-sonnet-4-6-thinking": "gemini-3-flash",
  "claude-sonnet-4-6": "gemini-3-flash",
};

export const OAUTH_CONFIG = {
  clientId:
    "YOUR_GOOGLE_OAUTH_CLIENT_ID.apps.googleusercontent.com",
  clientSecret: "YOUR_GOOGLE_OAUTH_CLIENT_SECRET",
  authUrl: "https://accounts.google.com/o/oauth2/v2/auth",
  tokenUrl: "https://oauth2.googleapis.com/token",
  userInfoUrl: "https://www.googleapis.com/oauth2/v1/userinfo",
  callbackPort: 51121,
  scopes: [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
  ],
};
export const OAUTH_REDIRECT_URI = "http://localhost:51121/oauth-callback";

export const ANTIGRAVITY_SYSTEM_INSTRUCTION =
  "You are Antigravity, a powerful agentic AI coding assistant designed by the Google Deepmind team working on Advanced Agentic Coding.You are pair programming with a USER to solve their coding task.";

export const TEST_MODELS = {
  claude: "claude-sonnet-4-6-thinking",
  gemini: "gemini-3-flash",
};

function getVersion(): string {
  try {
    const packageJson = require("../../../package.json");
    return packageJson.version || "1.0.0";
  } catch {
    return "1.0.0";
  }
}

export function getModelFamily(modelName: string): "claude" | "gemini" | "unknown" {
  const lower = (modelName || "").toLowerCase();
  if (lower.includes("claude")) return "claude";
  if (lower.includes("gemini")) return "gemini";
  return "unknown";
}

export function isThinkingModel(modelName: string): boolean {
  const lower = (modelName || "").toLowerCase();
  if (lower.includes("claude") && lower.includes("thinking")) return true;
  if (lower.includes("gemini")) {
    if (lower.includes("thinking")) return true;
    const versionMatch = lower.match(/gemini-(\d+)/);
    if (versionMatch && parseInt(versionMatch[1], 10) >= 3) return true;
  }
  return false;
}

export function getFallbackModel(model: string): string | null {
  return MODEL_FALLBACK_MAP[model] || null;
}