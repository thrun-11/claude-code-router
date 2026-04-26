import * as fs from "fs/promises";
import * as path from "path";
import * as os from "os";
import { OAUTH_CONFIG, OAUTH_REDIRECT_URI } from "./constants";

const AUTH_FILE = path.join(os.homedir(), ".claude-code-router", "antigravity-auth.json");

export interface AuthData {
  email: string;
  access_token: string;
  refresh_token: string;
  expiry: number;
  projectId?: string;
}

interface StoredAuth {
  accounts: AuthData[];
  activeEmail?: string;
}

export class AuthManager {
  private authData: StoredAuth = { accounts: [] };
  private tokenCache: Map<string, { token: string; expiresAt: number }> = new Map();
  private initialized = false;

  constructor() {
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    await this.loadAuth();
    this.initialized = true;
  }

  private async loadAuth(): Promise<void> {
    try {
      const data = await fs.readFile(AUTH_FILE, "utf-8");
      this.authData = JSON.parse(data);
    } catch {
      this.authData = { accounts: [] };
    }
  }

  private async saveAuth(): Promise<void> {
    try {
      await fs.mkdir(path.dirname(AUTH_FILE), { recursive: true });
      await fs.writeFile(AUTH_FILE, JSON.stringify(this.authData, null, 2));
    } catch (err) {
      this.logger?.error(`[AntigravityAuth] Failed to save auth: ${err}`);
    }
  }

  setLogger(logger: any) {
    this.logger = logger;
  }

  async getActiveToken(): Promise<string | null> {
    const activeEmail = this.authData.activeEmail || this.authData.accounts[0]?.email;
    if (!activeEmail) return null;

    const cached = this.tokenCache.get(activeEmail);
    if (cached && cached.expiresAt > Date.now() + 60000) {
      return cached.token;
    }

    const account = this.authData.accounts.find((a) => a.email === activeEmail);
    if (!account) return null;

    if (account.expiry < Date.now() - 60000) {
      const refreshed = await this.refreshToken(account);
      if (refreshed) {
        return refreshed.access_token;
      }
      return null;
    }

    return account.access_token;
  }

  async getActiveAccount(): Promise<AuthData | null> {
    const activeEmail = this.authData.activeEmail || this.authData.accounts[0]?.email;
    return this.authData.accounts.find((a) => a.email === activeEmail) || null;
  }

  async getTokenForAccount(email: string): Promise<string | null> {
    const cached = this.tokenCache.get(email);
    if (cached && cached.expiresAt > Date.now() + 60000) {
      return cached.token;
    }

    const account = this.authData.accounts.find((a) => a.email === email);
    if (!account) return null;

    if (account.expiry < Date.now() - 60000) {
      const refreshed = await this.refreshToken(account);
      if (refreshed) {
        return refreshed.access_token;
      }
      return null;
    }

    return account.access_token;
  }

  private async refreshToken(account: AuthData): Promise<AuthData | null> {
    if (!account.refresh_token) return null;

    try {
      const params = new URLSearchParams({
        client_id: OAUTH_CONFIG.clientId,
        client_secret: OAUTH_CONFIG.clientSecret,
        refresh_token: account.refresh_token,
        grant_type: "refresh_token",
      });

      const response = await fetch(OAUTH_CONFIG.tokenUrl, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: params.toString(),
      });

      if (!response.ok) {
        this.logger?.error("[AntigravityAuth] Token refresh failed");
        return null;
      }

      const data = await response.json();
      account.access_token = data.access_token;
      account.expiry = Date.now() + (data.expires_in || 3600) * 1000;

      this.tokenCache.set(account.email, {
        token: account.access_token,
        expiresAt: account.expiry,
      });

      const idx = this.authData.accounts.findIndex((a) => a.email === account.email);
      if (idx >= 0) {
        this.authData.accounts[idx] = account;
      }

      await this.saveAuth();
      return account;
    } catch (err) {
      this.logger?.error(`[AntigravityAuth] Token refresh error: ${err}`);
      return null;
    }
  }

  async addAccount(authResult: any): Promise<void> {
    const tokenInfo = await this.fetchTokenInfo(authResult.access_token);
    if (!tokenInfo) return;

    const account: AuthData = {
      email: tokenInfo.email || authResult.email || "unknown",
      access_token: authResult.access_token,
      refresh_token: authResult.refresh_token || "",
      expiry: Date.now() + (authResult.expires_in || 3600) * 1000,
      projectId: authResult.projectId || "rising-fact-p41fc",
    };

    const existing = this.authData.accounts.findIndex((a) => a.email === account.email);
    if (existing >= 0) {
      this.authData.accounts[existing] = account;
    } else {
      this.authData.accounts.push(account);
    }

    if (!this.authData.activeEmail) {
      this.authData.activeEmail = account.email;
    }

    this.tokenCache.set(account.email, {
      token: account.access_token,
      expiresAt: account.expiry,
    });

    await this.saveAuth();
  }

  private async fetchTokenInfo(token: string): Promise<any> {
    try {
      const response = await fetch(OAUTH_CONFIG.userInfoUrl, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (response.ok) {
        return await response.json();
      }
    } catch {}
    return null;
  }

  hasAccounts(): boolean {
    return this.authData.accounts.length > 0;
  }

  getAccounts(): AuthData[] {
    return [...this.authData.accounts];
  }

  setActiveAccount(email: string): void {
    if (this.authData.accounts.find((a) => a.email === email)) {
      this.authData.activeEmail = email;
      this.saveAuth();
    }
  }

  getAuthUrl(state?: string): string {
    const params = new URLSearchParams({
      client_id: OAUTH_CONFIG.clientId,
      redirect_uri: OAUTH_REDIRECT_URI,
      response_type: "code",
      scope: OAUTH_CONFIG.scopes.join(" "),
      access_type: "offline",
      prompt: "consent",
      ...(state && { state }),
    });
    return `${OAUTH_CONFIG.authUrl}?${params.toString()}`;
  }

  async exchangeCode(code: string): Promise<any> {
    const params = new URLSearchParams({
      client_id: OAUTH_CONFIG.clientId,
      client_secret: OAUTH_CONFIG.clientSecret,
      code,
      grant_type: "authorization_code",
      redirect_uri: OAUTH_REDIRECT_URI,
    });

    const response = await fetch(OAUTH_CONFIG.tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params.toString(),
    });

    if (!response.ok) {
      const err = await response.text();
      throw new Error(`Token exchange failed: ${err}`);
    }

    return await response.json();
  }

  private logger: any;
}