import * as http from "http";
import * as fs from "fs";
import * as path from "path";
import * as os from "os";
import { exec } from "child_process";
import {
  OAUTH_CONFIG,
  OAUTH_REDIRECT_URI,
} from "@ccr/core/transformer/antigravity/constants";

const AUTH_FILE = path.join(os.homedir(), ".claude-code-router", "antigravity-auth.json");

interface StoredAuth {
  accounts: AuthData[];
  activeEmail?: string;
}

interface AuthData {
  email: string;
  access_token: string;
  refresh_token: string;
  expiry: number;
  projectId?: string;
}

async function loadAuth(): Promise<StoredAuth> {
  try {
    const data = await fs.promises.readFile(AUTH_FILE, "utf-8");
    return JSON.parse(data);
  } catch {
    return { accounts: [] };
  }
}

async function saveAuth(data: StoredAuth): Promise<void> {
  await fs.promises.mkdir(path.dirname(AUTH_FILE), { recursive: true });
  await fs.promises.writeFile(AUTH_FILE, JSON.stringify(data, null, 2));
}

export async function runAuthAntigravity(): Promise<void> {
  console.log("Adding Antigravity account...\n");

  console.log("Opening browser for Google OAuth...\n");

  const authUrl = getAuthUrl();
  console.log(`Auth URL:\n${authUrl}\n`);

  await openBrowser(authUrl);

  console.log("Waiting for OAuth callback...\n");

  const code = await waitForCallback();

  console.log("Exchanging code for tokens...\n");

  const tokens = await exchangeCode(code);

  console.log("Saving account credentials...\n");

  await addAccount(tokens);

  console.log("Account added successfully!");
  console.log(`Credentials saved to: ${AUTH_FILE}`);

  const auth = await loadAuth();
  if (auth.accounts.length > 1) {
    console.log(`\nYou now have ${auth.accounts.length} accounts configured.`);
    console.log(`Active account: ${auth.activeEmail || auth.accounts[0].email}`);
  }
}

function getAuthUrl(): string {
  const params = new URLSearchParams({
    client_id: OAUTH_CONFIG.clientId,
    redirect_uri: OAUTH_REDIRECT_URI,
    response_type: "code",
    scope: OAUTH_CONFIG.scopes.join(" "),
    access_type: "offline",
    prompt: "consent",
  });
  return `${OAUTH_CONFIG.authUrl}?${params.toString()}`;
}

async function openBrowser(url: string): Promise<void> {
  const platform = process.platform;
  let cmd: string;
  if (platform === "win32") {
    cmd = `start "" "${url}"`;
  } else if (platform === "darwin") {
    cmd = `open "${url}"`;
  } else {
    cmd = `xdg-open "${url}"`;
  }
  return new Promise((resolve, reject) => {
    exec(cmd, (err) => {
      if (err) reject(err);
      else resolve();
    });
  });
}

function waitForCallback(): Promise<string> {
  return new Promise((resolve, reject) => {
    let server: http.Server;
    const startServer = (port: number) => {
      server = http.createServer((req, res) => {
        const reqUrl = req.url || "";
        const url = new URL(reqUrl, `http://localhost:${port}`);
        const code = url.searchParams.get("code");
        const error = url.searchParams.get("error");

        if (error) {
          res.writeHead(400, { "Content-Type": "text/html" });
          res.end("<html><body><h1>Authentication Failed</h1><p>You can close this window.</p></body></html>");
          server.close();
          reject(new Error(`OAuth error: ${error}`));
          return;
        }

        if (code) {
          res.writeHead(200, { "Content-Type": "text/html" });
          res.end("<html><body><h1>Authentication Successful!</h1><p>You can close this window and return to the terminal.</p></body></html>");
          server.close();
          resolve(code);
        } else {
          res.writeHead(404);
          res.end();
        }
      });

      server.on("error", (err: NodeJS.ErrnoException) => {
        if (err.code === "EADDRINUSE") {
          server.close();
          const nextPort = OAUTH_CONFIG.callbackFallbackPorts.shift();
          if (nextPort) {
            startServer(nextPort);
          } else {
            reject(new Error("No available ports for OAuth callback"));
          }
        } else {
          reject(err);
        }
      });

      server.listen(port, () => {
        if (port !== OAUTH_CONFIG.callbackPort) {
          console.log(`Port in use, callback server listening on http://localhost:${port}`);
        } else {
          console.log(`OAuth callback server listening on http://localhost:${port}`);
        }
      });
    };

    startServer(OAUTH_CONFIG.callbackPort);
  });
}

async function exchangeCode(code: string): Promise<any> {
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

async function addAccount(authResult: any): Promise<void> {
  const auth: StoredAuth = await loadAuth();

  let email = authResult.email;
  if (!email) {
    try {
      const userInfo = await fetch(OAUTH_CONFIG.userInfoUrl, {
        headers: { Authorization: `Bearer ${authResult.access_token}` },
      });
      if (userInfo.ok) {
        const info = await userInfo.json();
        email = info.email || "unknown";
      }
    } catch {}
  }

  if (!email) email = "unknown";

  const account: AuthData = {
    email,
    access_token: authResult.access_token,
    refresh_token: authResult.refresh_token || "",
    expiry: Date.now() + (authResult.expires_in || 3600) * 1000,
    projectId: authResult.projectId || "rising-fact-p41fc",
  };

  const existing = auth.accounts.findIndex((a) => a.email === account.email);
  if (existing >= 0) {
    auth.accounts[existing] = account;
  } else {
    auth.accounts.push(account);
  }

  if (!auth.activeEmail) {
    auth.activeEmail = account.email;
  }

  await saveAuth(auth);
}

export async function listAccounts(): Promise<void> {
  const auth = await loadAuth();

  if (auth.accounts.length === 0) {
    console.log("No Antigravity accounts configured.");
    console.log("Run `ccr antigravity add` to add your Google account.");
    return;
  }

  console.log(`\nAntigravity accounts (${auth.accounts.length}):\n`);
  for (const account of auth.accounts) {
    const isActive = account.email === auth.activeEmail;
    const expiryDate = new Date(account.expiry).toLocaleString();
    const status = account.expiry > Date.now() ? "Valid" : "Expired";
    console.log(`  ${isActive ? "*" : " "} ${account.email}`);
    console.log(`    Status: ${status} (expires: ${expiryDate})`);
    console.log();
  }
}

export async function removeAccount(email: string): Promise<void> {
  const auth = await loadAuth();
  const idx = auth.accounts.findIndex((a) => a.email === email);

  if (idx < 0) {
    console.log(`Account ${email} not found.`);
    return;
  }

  const removed = auth.accounts.splice(idx, 1)[0];
  if (auth.activeEmail === email) {
    auth.activeEmail = auth.accounts[0]?.email;
  }

  await saveAuth(auth);
  console.log(`Account ${removed.email} removed.`);
}

export async function setActiveAccount(email: string): Promise<void> {
  const auth = await loadAuth();
  if (!auth.accounts.find((a) => a.email === email)) {
    console.log(`Account ${email} not found.`);
    return;
  }

  auth.activeEmail = email;
  await saveAuth(auth);
  console.log(`Active account set to ${email}.`);
}

export async function checkAuthStatus(): Promise<void> {
  const auth = await loadAuth();

  if (auth.accounts.length === 0) {
    console.log("No Antigravity accounts configured.");
    return;
  }

  const account = auth.accounts.find((a) => a.email === (auth.activeEmail || auth.accounts[0].email));
  if (!account) {
    console.log("No active account.");
    return;
  }

  if (account.expiry < Date.now() - 60000) {
    try {
      const refreshed = await refreshToken(account);
      if (refreshed) {
        const authData = await loadAuth();
        const idx = authData.accounts.findIndex((a) => a.email === account.email);
        if (idx >= 0) authData.accounts[idx] = refreshed;
        await saveAuth(authData);
        console.log("Antigravity authentication OK (token refreshed).");
        console.log(`Active account: ${refreshed.email}`);
        return;
      }
    } catch {}
    console.log("Token expired. Run `ccr antigravity add` to re-authenticate.");
    return;
  }

  console.log("Antigravity authentication OK.");
  console.log(`Active account: ${account.email}`);
}

async function refreshToken(account: AuthData): Promise<AuthData | null> {
  if (!account.refresh_token) return null;

  const params = new URLSearchParams({
    client_id: OAUTH_CONFIG.clientId,
    client_secret: OAUTH_CONFIG.clientSecret,
    refresh_token: account.refresh_token,
    grant_type: "refresh_token",
  });

  try {
    const response = await fetch(OAUTH_CONFIG.tokenUrl, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: params.toString(),
    });

    if (!response.ok) return null;

    const data = await response.json();
    account.access_token = data.access_token;
    account.expiry = Date.now() + (data.expires_in || 3600) * 1000;
    return account;
  } catch {
    return null;
  }
}

export async function checkQuota(): Promise<void> {
  console.log("Fetching quota from Antigravity...\n");
  
  const auth = await loadAuth();
  if (auth.accounts.length === 0) {
    console.log("No Antigravity accounts configured.");
    return;
  }

  const account = auth.accounts.find((a) => a.email === (auth.activeEmail || auth.accounts[0].email));
  if (!account) {
    console.log("No active account found.");
    return;
  }

  // Ensure token is valid
  let tokenToUse = account.access_token;
  if (account.expiry < Date.now() - 60000) {
    const refreshed = await refreshToken(account);
    if (!refreshed) {
      console.log("Token expired. Run `ccr antigravity add` to re-authenticate.");
      return;
    }
    const idx = auth.accounts.findIndex((a) => a.email === account.email);
    if (idx >= 0) auth.accounts[idx] = refreshed;
    await saveAuth(auth);
    tokenToUse = refreshed.access_token;
  }

  try {
    const response = await fetch("https://daily-cloudcode-pa.googleapis.com/v1internal:loadCodeAssist", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${tokenToUse}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        mode: "FULL_ELIGIBILITY_CHECK",
        metadata: {
          ideName: "antigravity",
          ideType: "ANTIGRAVITY",
          ideVersion: "1.23.2",
          pluginVersion: "unknown",
          platform: "DARWIN_AMD64",
          updateChannel: "stable",
          pluginType: "GEMINI"
        }
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      console.error(`Failed to fetch quota: HTTP ${response.status}`);
      console.error(errText);
      return;
    }

    const data = await response.json();
    let availableCredits = "Unknown";
    
    // Antigravity returns G1 credits inside paidTier -> availableCredits array
    if (data.paidTier && Array.isArray(data.paidTier.availableCredits)) {
      let total = 0;
      for (const credit of data.paidTier.availableCredits) {
        if (credit.creditAmount) {
          total += Number(credit.creditAmount);
        }
      }
      availableCredits = total.toString();
    } else if (data.currentTier && Array.isArray(data.currentTier.availableCredits)) {
      let total = 0;
      for (const credit of data.currentTier.availableCredits) {
        if (credit.creditAmount) {
          total += Number(credit.creditAmount);
        }
      }
      availableCredits = total.toString();
    }

    const projectId = account.projectId || data.cloudaicompanionProject;
    console.log(`Account: ${account.email}`);
    console.log(`Project: ${projectId || "Unknown"}`);
    console.log(`Tier ID: ${data.paidTier?.id || data.currentTier?.id || "Unknown"}\n`);
    console.log(`Available credits: ${availableCredits}`);

    try {
      const modelsResponse = await fetch("https://daily-cloudcode-pa.googleapis.com/v1internal:fetchAvailableModels", {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${tokenToUse}`,
          "Content-Type": "application/json",
          "User-Agent": "antigravity/1.23.2 DARWIN_AMD64/amd64"
        },
        body: JSON.stringify({})
      });

      if (modelsResponse.ok) {
        const modelsData = await modelsResponse.json();
        if (modelsData.models) {
          const targetModels = [
            "gemini-3.1-pro-high",
            "gemini-3-flash-agent",
            "gemini-3.8-flash-tiered",
            "gemini-3.7-flash-tiered",
            "gemini-3.6-flash-tiered",
            "claude-sonnet-4-6",
            "claude-opus-4-6-thinking"
          ];
          
          console.log("Model Limits:");
          for (const modelId of targetModels) {
            const model = modelsData.models[modelId];
            if (model && model.quotaInfo) {
              const fraction = model.quotaInfo.remainingFraction;
              const reset = model.quotaInfo.resetTime ? new Date(model.quotaInfo.resetTime).toLocaleString() : "Unknown";
              
              let percentage = "Unknown";
              if (fraction !== undefined) {
                percentage = `${Math.round(fraction * 100)}%`;
              } else if (model.quotaInfo.resetTime) {
                percentage = "0%";
              }

              console.log(`  - ${model.displayName || modelId}: ${percentage} remaining (Resets: ${reset})`);
            }
          }
        }
      }
    } catch (e) {
      // Silently ignore
    }

  } catch (error: any) {
    console.error("Error fetching quota:", error.message);
  }
}