import * as https from "https";
import * as http from "http";
import { Readable } from "stream";
import type { HttpResponse, HttpClientOptions } from "./types";
import { hasNative, nativeGet, nativePost, nativePostStream, clearNativeCache } from "./native";

function parseUrl(url: string): { protocol: string; hostname: string; port: number; path: string } {
  const parsed = new URL(url);
  return {
    protocol: parsed.protocol,
    hostname: parsed.hostname,
    port: parsed.port ? parseInt(parsed.port, 10) : (parsed.protocol === "https:" ? 443 : 80),
    path: parsed.pathname + parsed.search,
  };
}

function buildHeaders(headers: Record<string, string>): http.OutgoingHttpHeaders {
  const result: http.OutgoingHttpHeaders = {};
  for (const [key, value] of Object.entries(headers)) {
    result[key] = value;
  }
  return result;
}

function extractSetCookie(headers: http.IncomingHttpHeaders): string[] {
  const setCookie = headers["set-cookie"];
  if (!setCookie) return [];
  return Array.isArray(setCookie) ? setCookie : [setCookie];
}

function headersToMap(headers: http.IncomingHttpHeaders): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    if (value) {
      result[key] = Array.isArray(value) ? value.join(", ") : value;
    }
  }
  return result;
}

function makeRequest(
  isStream: boolean,
  method: string,
  url: string,
  headers: Record<string, string>,
  body: string | undefined,
  options: HttpClientOptions = {},
  onChunk?: (chunk: string, setCookies: string[]) => void
): Promise<HttpResponse> {
  return new Promise((resolve, reject) => {
    const parsed = parseUrl(url);
    const isHttps = parsed.protocol === "https:";
    const lib = isHttps ? https : http;

    const requestOptions: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port,
      path: parsed.path,
      method,
      headers: buildHeaders(headers),
      timeout: options.timeoutSec ? options.timeoutSec * 1000 : 30000,
    };

    const req = lib.request(requestOptions, (res) => {
      const setCookies = extractSetCookie(res.headers);
      const responseHeaders = headersToMap(res.headers);

      if (isStream && onChunk) {
        res.on("data", (chunk: Buffer) => {
          onChunk(chunk.toString(), []);
        });

        res.on("end", () => {
          onChunk("", setCookies);
          resolve({
            status: res.statusCode || 0,
            body: "",
            set_cookie_headers: setCookies,
            headers: responseHeaders,
          });
        });

        res.on("error", reject);
      } else {
        let data = "";
        res.on("data", (chunk: Buffer) => {
          data += chunk.toString();
        });

        res.on("end", () => {
          resolve({
            status: res.statusCode || 0,
            body: data,
            set_cookie_headers: setCookies,
            headers: responseHeaders,
          });
        });

        res.on("error", reject);
      }
    });

    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timeout"));
    });

    if (body) {
      req.write(body);
    }
    req.end();
  });
}

export async function httpGet(
  url: string,
  headers: Record<string, string>,
  options: HttpClientOptions = {}
): Promise<HttpResponse> {
  if (hasNative()) {
    return nativeGet(url, headers, options.timeoutSec, options.proxyUrl, options.forceHttp11);
  }
  return makeRequest(false, "GET", url, headers, undefined, options);
}

export async function httpPost(
  url: string,
  headers: Record<string, string>,
  body: string,
  options: HttpClientOptions = {}
): Promise<HttpResponse> {
  if (hasNative()) {
    return nativePost(url, headers, body, options.timeoutSec, options.proxyUrl, options.forceHttp11);
  }
  return makeRequest(false, "POST", url, headers, body, options);
}

export async function httpPostStream(
  url: string,
  headers: Record<string, string>,
  body: string,
  onChunk: (chunk: string, setCookies: string[]) => void,
  options: HttpClientOptions = {}
): Promise<void> {
  if (hasNative()) {
    return new Promise((resolve, reject) => {
      try {
        nativePostStream(
          url,
          headers,
          body,
          (chunk) => {
            onChunk(chunk.content, chunk.done ? (chunk.set_cookie_headers || []) : []);
            if (chunk.done) {
              resolve();
            }
          },
          options.timeoutSec,
          options.proxyUrl,
          options.forceHttp11,
        );
      } catch (error) {
        reject(error);
      }
    });
  }
  return makeRequest(true, "POST", url, headers, body, options, onChunk) as unknown as Promise<void>;
}

export async function httpPostStreamResponse(
  url: string,
  headers: Record<string, string>,
  body: string,
  options: HttpClientOptions = {}
): Promise<{ response: Response; setCookieHeaders: string[] }> {
  return new Promise((resolve, reject) => {
    const parsed = parseUrl(url);
    const isHttps = parsed.protocol === "https:";
    const lib = isHttps ? https : http;

    const requestOptions: http.RequestOptions = {
      hostname: parsed.hostname,
      port: parsed.port,
      path: parsed.path,
      method: "POST",
      headers: buildHeaders(headers),
      timeout: options.timeoutSec ? options.timeoutSec * 1000 : 30000,
    };

    const req = lib.request(requestOptions, (res) => {
      const setCookieHeaders = extractSetCookie(res.headers);
      const responseHeaders = headersToMap(res.headers);
      const nodeReadable = res as unknown as Readable;
      const webStream = Readable.toWeb(nodeReadable) as ReadableStream<Uint8Array>;
      const response = new Response(webStream, {
        status: res.statusCode || 0,
        headers: responseHeaders,
      });
      resolve({ response, setCookieHeaders });
    });

    req.on("error", reject);
    req.on("timeout", () => {
      req.destroy();
      reject(new Error("Request timeout"));
    });

    req.write(body);
    req.end();
  });
}

export async function clearCache(): Promise<void> {
  clearNativeCache();
}
