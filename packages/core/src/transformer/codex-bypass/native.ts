import * as path from "path";
import * as fs from "fs";

export interface NativeHttpResponse {
  status: number;
  body: string;
  set_cookie_headers?: string[];
  headers?: Record<string, string>;
}

type NativeModule = {
  httpGet: (
    url: string,
    headers: Record<string, string>,
    timeoutSec?: number,
    proxyUrl?: string,
    forceHttp11?: boolean
  ) => NativeHttpResponse;
  httpPost: (
    url: string,
    headers: Record<string, string>,
    body: string,
    timeoutSec?: number,
    proxyUrl?: string,
    forceHttp11?: boolean
  ) => NativeHttpResponse;
  httpPostStream: (
    url: string,
    headers: Record<string, string>,
    body: string,
    callback: (err: any, chunk: { content: string; done: boolean; setCookieHeaders?: string[] }) => void,
    timeoutSec?: number,
    proxyUrl?: string,
    forceHttp11?: boolean
  ) => void;
  clearClientCache?: () => void;
};

let native: NativeModule | null = null;

function getNativePath(): string {
  const root = process.cwd();
  const platform = process.platform;
  const arch = process.arch;
  const fileName = `codex-tls.${platform}-${arch}.node`;
  return path.join(root, "native", fileName);
}

export function loadNative(): NativeModule | null {
  if (native) return native;

  const nativePath = getNativePath();
  if (!fs.existsSync(nativePath)) {
    return null;
  }

  try {
    native = require(nativePath) as NativeModule;
    return native;
  } catch {
    return null;
  }
}

export function hasNative(): boolean {
  return loadNative() !== null;
}

export function nativeGet(
  url: string,
  headers: Record<string, string>,
  timeoutSec?: number,
  proxyUrl?: string,
  forceHttp11?: boolean
): NativeHttpResponse {
  const mod = loadNative();
  if (!mod) {
    throw new Error("Native codex-tls addon not available");
  }
  return mod.httpGet(url, headers, timeoutSec, proxyUrl, forceHttp11);
}

export function nativePost(
  url: string,
  headers: Record<string, string>,
  body: string,
  timeoutSec?: number,
  proxyUrl?: string,
  forceHttp11?: boolean
): NativeHttpResponse {
  const mod = loadNative();
  if (!mod) {
    throw new Error("Native codex-tls addon not available");
  }
  return mod.httpPost(url, headers, body, timeoutSec, proxyUrl, forceHttp11);
}

export function nativePostStream(
  url: string,
  headers: Record<string, string>,
  body: string,
  callback: (chunk: { content: string; done: boolean; set_cookie_headers: string[] }) => void,
  timeoutSec?: number,
  proxyUrl?: string,
  forceHttp11?: boolean
): void {
  const mod = loadNative();
  if (!mod) {
    throw new Error("Native codex-tls addon not available");
  }
  mod.httpPostStream(
    url,
    headers,
    body,
    (err, chunk) => {
      if (err) {
        throw err;
      }
      callback({
        content: chunk?.content || "",
        done: Boolean(chunk?.done),
        set_cookie_headers: chunk?.setCookieHeaders || [],
      });
    },
    timeoutSec,
    proxyUrl,
    forceHttp11
  );
}

export function clearNativeCache(): void {
  const mod = loadNative();
  if (mod?.clearClientCache) {
    mod.clearClientCache();
  }
}
