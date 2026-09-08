import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { randomUUID } from "crypto";

export interface StoredCookie {
  value: string;
  expires: number | null;
}

export interface CookieFile {
  _version: 2;
  accounts: Record<string, Record<string, StoredCookie>>;
}

const CRITICAL_COOKIES = new Set(["cf_clearance", "__cf_bm"]);

export class CookieJar {
  private cookies: Map<string, Record<string, StoredCookie>> = new Map();
  private persistTimer: ReturnType<typeof setTimeout> | null = null;
  private dataDir: string;
  private cookieFile: string;
  private logger?: Console;

  constructor(logger?: Console) {
    this.logger = logger;
    this.dataDir = path.join(os.homedir(), ".claude-code-router");
    this.cookieFile = path.join(this.dataDir, "codex-cookies.json");
    this.load();
  }

  private load(): void {
    try {
      if (fs.existsSync(this.cookieFile)) {
        const data = JSON.parse(fs.readFileSync(this.cookieFile, "utf-8")) as CookieFile;
        if (data._version === 2 && data.accounts) {
          this.cookies = new Map(Object.entries(data.accounts));
          this.logger?.debug("[CookieJar] Loaded cookies for", this.cookies.size, "accounts");
        }
      }
    } catch (error) {
      this.logger?.warn("[CookieJar] Failed to load cookies:", error);
    }
  }

  private schedulePersist(): void {
    if (this.persistTimer) return;
    this.persistTimer = setTimeout(() => {
      this.persistTimer = null;
      this.persistAsync().catch((err) => {
        this.logger?.warn("[CookieJar] Persist failed:", err);
      });
    }, 1000);
  }

  private async persistAsync(): Promise<void> {
    try {
      if (!fs.existsSync(this.dataDir)) {
        fs.mkdirSync(this.dataDir, { recursive: true });
      }

      const data: CookieFile = {
        _version: 2,
        accounts: Object.fromEntries(this.cookies),
      };

      const tmpFile = this.cookieFile + ".tmp";
      fs.writeFileSync(tmpFile, JSON.stringify(data, null, 2), "utf-8");
      fs.renameSync(tmpFile, this.cookieFile);
    } catch (error) {
      this.logger?.warn("[CookieJar] Persist failed:", error);
    }
  }

  set(accountId: string, cookies: string | Record<string, string>): void {
    const existing = this.cookies.get(accountId) || {};

    if (typeof cookies === "string") {
      for (const part of cookies.split(";")) {
        const eq = part.indexOf("=");
        if (eq === -1) continue;
        const name = part.slice(0, eq).trim();
        const value = part.slice(eq + 1).trim();
        if (name) existing[name] = { value, expires: null };
      }
    } else {
      for (const [k, v] of Object.entries(cookies)) {
        existing[k] = { value: v, expires: null };
      }
    }

    this.cookies.set(accountId, existing);
    this.schedulePersist();
  }

  getCookieHeader(accountId: string): string | null {
    const cookies = this.cookies.get(accountId);
    if (!cookies || Object.keys(cookies).length === 0) return null;

    const now = Date.now();
    const pairs: string[] = [];
    for (const [k, c] of Object.entries(cookies)) {
      if (c.expires !== null && c.expires <= now) continue;
      pairs.push(`${k}=${c.value}`);
    }
    return pairs.length > 0 ? pairs.join("; ") : null;
  }

  capture(accountId: string, setCookies: string[]): void {
    if (setCookies.length === 0) return;

    const existing = this.cookies.get(accountId) || {};
    let changed = false;
    let hasCritical = false;

    for (const raw of setCookies) {
      const parts = raw.split(";").map((s) => s.trim());
      const pair = parts[0];
      const eq = pair.indexOf("=");
      if (eq === -1) continue;

      const name = pair.slice(0, eq).trim();
      const value = pair.slice(eq + 1).trim();
      if (!name) continue;

      let expires: number | null = null;
      for (let i = 1; i < parts.length; i++) {
        const attr = parts[i].toLowerCase();
        if (attr.startsWith("max-age=")) {
          const seconds = parseInt(attr.slice(8), 10);
          if (!isNaN(seconds)) {
            expires = seconds <= 0 ? 0 : Date.now() + seconds * 1000;
          }
          break;
        }
        if (attr.startsWith("expires=")) {
          const date = new Date(attr.slice(8));
          if (!isNaN(date.getTime())) {
            expires = date.getTime();
          }
        }
      }

      const prev = existing[name];
      if (!prev || prev.value !== value || prev.expires !== expires) {
        existing[name] = { value, expires };
        changed = true;
        if (CRITICAL_COOKIES.has(name)) hasCritical = true;
      }
    }

    if (changed) {
      this.cookies.set(accountId, existing);
      if (hasCritical) {
        this.persistAsync().catch((err) => {
          this.logger?.warn("[CookieJar] Critical cookie persist failed:", err);
        });
      } else {
        this.schedulePersist();
      }
    }
  }

  get(accountId: string): Record<string, string> | null {
    const cookies = this.cookies.get(accountId);
    if (!cookies) return null;
    const result: Record<string, string> = {};
    for (const [k, c] of Object.entries(cookies)) {
      result[k] = c.value;
    }
    return result;
  }

  clear(accountId: string): void {
    if (this.cookies.delete(accountId)) {
      this.schedulePersist();
    }
  }

  warmup(accountId: string, _baseUrl: string, headers: Record<string, string>, _httpClient: any): string {
    const cookieHeader = this.getCookieHeader(accountId);
    const requestHeaders = { ...headers };
    if (cookieHeader) {
      requestHeaders["Cookie"] = cookieHeader;
    }
    return JSON.stringify(requestHeaders);
  }

  generateRequestId(): string {
    return randomUUID();
  }
}