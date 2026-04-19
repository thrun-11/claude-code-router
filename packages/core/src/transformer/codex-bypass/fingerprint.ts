import { randomUUID } from "crypto";

export interface CodexClientConfig {
  app_version: string;
  build_number: string;
  chromium_version: string;
  platform: string;
  arch: string;
  originator: string;
}

export interface FingerprintConfig {
  user_agent_template: string;
  default_headers: Record<string, string>;
  header_order: string[];
}

const DEFAULT_CLIENT_CONFIG: CodexClientConfig = {
  app_version: "1.0.128",
  build_number: "14567",
  chromium_version: "136",
  platform: "macOS",
  arch: "arm64",
  originator: "codex-cli",
};

const DEFAULT_FINGERPRINT: FingerprintConfig = {
  user_agent_template: "codex/{version} ({platform}; {arch})",
  default_headers: {
    "accept-encoding": "gzip, deflate, br, zstd",
    "accept-language": "en-US,en;q=0.9",
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-origin",
  },
  header_order: [
    "authorization",
    "content-type",
    "user-agent",
    "sec-ch-ua",
    "sec-ch-ua-mobile",
    "sec-ch-ua-platform",
    "originator",
    "x-client-request-id",
    "x-openai-internal-codex-residency",
    "x-codex-turn-state",
    "chatgpt-account-id",
    "accept-language",
    "sec-fetch-dest",
    "sec-fetch-mode",
    "sec-fetch-site",
    "accept-encoding",
  ],
};

export class FingerprintManager {
  private clientConfig: CodexClientConfig;
  private fingerprint: FingerprintConfig;
  private logger?: Console;

  constructor(logger?: Console) {
    this.clientConfig = { ...DEFAULT_CLIENT_CONFIG };
    this.fingerprint = { ...DEFAULT_FINGERPRINT };
    this.logger = logger;
  }

  buildUserAgent(): string {
    return this.fingerprint.user_agent_template
      .replace("{version}", this.clientConfig.app_version)
      .replace("{platform}", this.clientConfig.platform)
      .replace("{arch}", this.clientConfig.arch);
  }

  buildSecChUa(): string {
    return `"Chromium";v="${this.clientConfig.chromium_version}", "Not:A-Brand";v="24"`;
  }

  buildHeaders(authToken: string, accountId?: string): Record<string, string> {
    const headers: Record<string, string> = {};

    headers["authorization"] = `Bearer ${authToken}`;
    headers["user-agent"] = this.buildUserAgent();
    headers["sec-ch-ua"] = this.buildSecChUa();
    headers["sec-ch-ua-mobile"] = "?0";
    headers["sec-ch-ua-platform"] = `"${this.clientConfig.platform}"`;
    headers["originator"] = this.clientConfig.originator;
    headers["x-client-request-id"] = randomUUID();
    headers["x-openai-internal-codex-residency"] = "local";
    headers["x-codex-turn-state"] = "";

    if (accountId) {
      headers["chatgpt-account-id"] = accountId;
    }

    for (const [key, value] of Object.entries(this.fingerprint.default_headers)) {
      if (!(key in headers)) {
        headers[key] = value;
      }
    }

    return this.orderHeaders(headers);
  }

  buildHeadersWithContentType(authToken: string, accountId?: string): Record<string, string> {
    const headers = this.buildHeaders(authToken, accountId);
    headers["content-type"] = "application/json";
    return this.orderHeaders(headers);
  }

  buildAnonymousHeaders(): Record<string, string> {
    const headers: Record<string, string> = {};

    headers["user-agent"] = this.buildUserAgent();
    headers["sec-ch-ua"] = this.buildSecChUa();
    headers["sec-ch-ua-mobile"] = "?0";
    headers["sec-ch-ua-platform"] = `"${this.clientConfig.platform}"`;

    for (const [key, value] of Object.entries(this.fingerprint.default_headers)) {
      if (!(key in headers)) {
        headers[key] = value;
      }
    }

    return this.orderHeaders(headers);
  }

  private orderHeaders(headers: Record<string, string>): Record<string, string> {
    const ordered: Record<string, string> = {};

    for (const key of this.fingerprint.header_order) {
      const lowerKey = key.toLowerCase();
      for (const headerKey of Object.keys(headers)) {
        if (headerKey.toLowerCase() === lowerKey) {
          ordered[key] = headers[headerKey];
          break;
        }
      }
    }

    for (const key of Object.keys(headers)) {
      if (!(key in ordered)) {
        ordered[key] = headers[key];
      }
    }

    return ordered;
  }

  getClientConfig(): CodexClientConfig {
    return { ...this.clientConfig };
  }

  async updateFromCodex(baseUrl: string, httpClient: any): Promise<boolean> {
    try {
      const headers = this.buildAnonymousHeaders();
      const url = `${baseUrl}/codex/usage`;

      const response = await httpClient.get(url, headers, { timeoutSec: 10 });

      if (response.status === 200) {
        const data = JSON.parse(response.body);
        
        if (data.app_version) this.clientConfig.app_version = data.app_version;
        if (data.build_number) this.clientConfig.build_number = data.build_number;
        if (data.chromium_version) this.clientConfig.chromium_version = data.chromium_version;

        this.logger?.info("[Fingerprint] Updated from Codex:", this.clientConfig);
        return true;
      }
    } catch (error) {
      this.logger?.warn("[Fingerprint] Failed to update from Codex:", error);
    }
    return false;
  }

  setClientConfig(config: Partial<CodexClientConfig>): void {
    this.clientConfig = { ...this.clientConfig, ...config };
  }
}