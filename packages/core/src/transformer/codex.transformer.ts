import { Transformer, TransformerContext } from "@ccr/core/types/transformer";
import { LLMProvider, UnifiedChatRequest, UnifiedMessage } from "@ccr/core/types/llm";
import * as os from "os";
import * as path from "path";
import * as fs from "fs/promises";
import { CookieJar } from "./codex-bypass/cookie-jar";
import { FingerprintManager } from "./codex-bypass/fingerprint";
import { httpGet, httpPost, httpPostStream } from "./codex-bypass/http-client";
import type { HttpResponse } from "./codex-bypass/types";
import { hasNative } from "./codex-bypass/native";

const AUTH_FILE = path.join(os.homedir(), ".codex", "auth.json");
const DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api/codex";
const DEFAULT_INSTRUCTIONS = "You are Codex, a coding assistant.";
// Throttle for the usage warmup: rate-limit data changes slowly and the
// request adds latency, so refresh at most once per minute per instance.
const CODEX_WARMUP_THROTTLE_MS = 60 * 1000;
// Statusline quota bar cache (scripts/statusline-opencode.sh reads this).
// Written by the gateway because script curl cannot pass the Cloudflare check.
const CODEX_USAGE_CACHE_FILE = "/tmp/codex-usage.json";

interface CodexAuth {
  auth_mode: string;
  OPENAI_API_KEY: string | null;
  tokens: {
    id_token: string;
    access_token: string;
    refresh_token: string;
    account_id: string;
  };
  last_refresh: string;
}

/**
 * Converts OpenAI-style Responses SSE frames to Anthropic Messages SSE.
 * Frames are split on blank lines so truncated JSON across TCP chunks is
 * buffered, not dropped. Unknown event types are ignored; only Anthropic
 * events are emitted. Shared with the opencode-go transformer, whose
 * Responses endpoints speak the same protocol.
 */
export class ResponsesToAnthropicStream {
  private messageId: string;
  private model: string;
  private started = false;
  private finished = false;
  private buffer = "";
  private nextIndex = 0;
  private openBlocks: number[] = [];
  private textBlock: number | null = null;
  private thinkingBlock: number | null = null;
  private toolByItemId = new Map<string, number>();
  private toolByOutputIndex = new Map<number, number>();
  private hasToolUse = false;
  private usage: any = null;

  constructor(fallbackModel: string) {
    this.messageId = `msg_${Date.now()}_${Math.floor(Math.random() * 1e6)}`;
    this.model = fallbackModel || "unknown";
  }

  push(chunk: string): string[] {
    if (this.finished) return [];
    this.buffer += chunk;
    const out: string[] = [];
    let boundary: number;
    while ((boundary = this.buffer.indexOf("\n\n")) >= 0) {
      const frame = this.buffer.slice(0, boundary);
      this.buffer = this.buffer.slice(boundary + 2);
      out.push(...this.handleFrame(frame));
      if (this.finished) break;
    }
    return out;
  }

  finish(stopReason = "end_turn"): string[] {
    if (this.finished) return [];
    return this.finishStream(stopReason);
  }

  private handleFrame(frame: string): string[] {
    const dataLines: string[] = [];
    for (const line of frame.split("\n")) {
      const trimmed = line.trimStart();
      if (trimmed.startsWith("data:")) {
        dataLines.push(trimmed.slice(5).trim());
      }
    }
    const raw = dataLines.join("\n").trim();
    if (!raw || raw === "[DONE]") return [];
    try {
      return this.handleEvent(JSON.parse(raw));
    } catch {
      return [];
    }
  }

  private handleEvent(event: any): string[] {
    const out: string[] = [];
    const type = event?.type;

    if (type === "response.created") {
      if (event.response?.model) this.model = event.response.model;
      if (event.response?.id) this.messageId = `msg_${event.response.id}`;
      out.push(...this.ensureStarted());
      return out;
    }

    if (type === "response.output_item.added") {
      out.push(...this.ensureStarted());
      this.openItemBlock(event.item, event.output_index, out);
      return out;
    }

    if (type === "response.output_item.done") {
      out.push(...this.ensureStarted());
      const item = event.item || {};
      if (item.type === "function_call") {
        const index = this.ensureToolBlock(item, event.output_index, out);
        // If the full arguments arrived without deltas, emit them so the
        // tool_use block is not left with an empty input.
        if (index !== null && item.arguments) {
          out.push(
            this.sse("content_block_delta", {
              type: "content_block_delta",
              index,
              delta: { type: "input_json_delta", partial_json: item.arguments },
            })
          );
        }
      } else if (item.type === "reasoning") {
        this.ensureThinkingBlock(out, event.output_index);
      } else {
        this.ensureTextBlock(out);
      }
      return out;
    }

    if (type === "response.output_text.delta") {
      out.push(...this.ensureStarted());
      this.ensureTextBlock(out);
      if (this.textBlock !== null && event.delta) {
        out.push(
          this.sse("content_block_delta", {
            type: "content_block_delta",
            index: this.textBlock,
            delta: { type: "text_delta", text: event.delta },
          })
        );
      }
      return out;
    }

    if (type === "response.function_call_arguments.delta") {
      out.push(...this.ensureStarted());
      const index = this.resolveToolBlock(event.item_id, event.output_index);
      if (index !== null && event.delta) {
        out.push(
          this.sse("content_block_delta", {
            type: "content_block_delta",
            index,
            delta: { type: "input_json_delta", partial_json: event.delta },
          })
        );
      }
      return out;
    }

    // Reasoning summary deltas (e.g. response.reasoning_summary_text.delta).
    if (
      typeof type === "string" &&
      type.includes("reasoning") &&
      typeof event.delta === "string" &&
      event.delta
    ) {
      out.push(...this.ensureStarted());
      this.ensureThinkingBlock(out, event.output_index);
      if (this.thinkingBlock !== null) {
        out.push(
          this.sse("content_block_delta", {
            type: "content_block_delta",
            index: this.thinkingBlock,
            delta: { type: "thinking_delta", thinking: event.delta },
          })
        );
      }
      return out;
    }

    if (type === "response.completed") {
      this.usage = event.response?.usage || this.usage;
      return this.finishStream(this.hasToolUse ? "tool_use" : "end_turn");
    }

    if (type === "response.incomplete") {
      this.usage = event.response?.usage || this.usage;
      const reason = event.response?.incomplete_details?.reason;
      return this.finishStream(reason === "max_output_tokens" ? "max_tokens" : "end_turn");
    }

    if (type === "response.failed") {
      const message =
        event.response?.error?.message || "Codex request failed upstream.";
      out.push(
        this.sse("error", {
          type: "error",
          error: { type: "api_error", message },
        })
      );
      this.finished = true;
      return out;
    }

    return out;
  }

  private openItemBlock(item: any, outputIndex: number | undefined, out: string[]): void {
    if (!item || typeof item !== "object") {
      this.ensureTextBlock(out);
      return;
    }
    if (item.type === "function_call") {
      this.ensureToolBlock(item, outputIndex, out);
      return;
    }
    if (item.type === "reasoning") {
      this.ensureThinkingBlock(out, outputIndex);
      return;
    }
    this.ensureTextBlock(out);
  }

  private ensureStarted(): string[] {
    if (this.started) return [];
    this.started = true;
    return [
      this.sse("message_start", {
        type: "message_start",
        message: {
          id: this.messageId,
          type: "message",
          role: "assistant",
          content: [],
          model: this.model,
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      }),
    ];
  }

  private ensureTextBlock(out: string[]): void {
    if (this.textBlock !== null) return;
    const index = this.nextIndex++;
    this.textBlock = index;
    this.openBlocks.push(index);
    out.push(
      this.sse("content_block_start", {
        type: "content_block_start",
        index,
        content_block: { type: "text", text: "" },
      })
    );
  }

  private ensureThinkingBlock(out: string[], outputIndex?: number): void {
    if (this.thinkingBlock !== null) {
      if (typeof outputIndex === "number") {
        this.toolByOutputIndex.set(outputIndex, this.thinkingBlock);
      }
      return;
    }
    const index = this.nextIndex++;
    this.thinkingBlock = index;
    this.openBlocks.push(index);
    if (typeof outputIndex === "number") {
      this.toolByOutputIndex.set(outputIndex, index);
    }
    out.push(
      this.sse("content_block_start", {
        type: "content_block_start",
        index,
        content_block: { type: "thinking", thinking: "" },
      })
    );
  }

  private ensureToolBlock(item: any, outputIndex: number | undefined, out: string[]): number | null {
    const name = item?.name;
    if (!name) return this.resolveToolBlock(item?.id || item?.call_id, outputIndex);
    const id = item.call_id || item.id || `toolu_${Date.now()}`;
    if (item.id) {
      const existing = this.toolByItemId.get(item.id);
      if (existing !== undefined) return existing;
    }
    const index = this.nextIndex++;
    this.openBlocks.push(index);
    if (item.id) this.toolByItemId.set(item.id, index);
    if (typeof outputIndex === "number") this.toolByOutputIndex.set(outputIndex, index);
    this.hasToolUse = true;
    out.push(
      this.sse("content_block_start", {
        type: "content_block_start",
        index,
        content_block: { type: "tool_use", id, name, input: {} },
      })
    );
    return index;
  }

  private resolveToolBlock(itemId?: string, outputIndex?: number): number | null {
    if (itemId && this.toolByItemId.has(itemId)) {
      return this.toolByItemId.get(itemId)!;
    }
    if (typeof outputIndex === "number" && this.toolByOutputIndex.has(outputIndex)) {
      const mapped = this.toolByOutputIndex.get(outputIndex)!;
      // Only return tool_use blocks here; text/thinking blocks are tracked
      // separately and must not receive input_json deltas.
      if (mapped !== this.textBlock && mapped !== this.thinkingBlock) {
        return mapped;
      }
      return null;
    }
    return null;
  }

  private finishStream(stopReason: string): string[] {
    if (this.finished) return [];
    this.finished = true;
    const out: string[] = [];
    out.push(...this.ensureStarted());
    for (const index of this.openBlocks) {
      if (index === this.thinkingBlock) {
        out.push(
          this.sse("content_block_delta", {
            type: "content_block_delta",
            index,
            delta: { type: "signature_delta", signature: String(Date.now()) },
          })
        );
      }
      out.push(
        this.sse("content_block_stop", { type: "content_block_stop", index })
      );
    }
    this.openBlocks = [];
    this.textBlock = null;
    this.thinkingBlock = null;
    const usage = this.usage || {};
    const cachedTokens = usage.input_tokens_details?.cached_tokens || 0;
    out.push(
      this.sse("message_delta", {
        type: "message_delta",
        delta: { stop_reason: stopReason, stop_sequence: null },
        usage: {
          input_tokens: usage.input_tokens || 0,
          output_tokens: usage.output_tokens || 0,
          ...(cachedTokens ? { cache_read_input_tokens: cachedTokens } : {}),
        },
      })
    );
    out.push(this.sse("message_stop", { type: "message_stop" }));
    return out;
  }

  private sse(event: string, data: any): string {
    return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  }
}

export class CodexTransformer implements Transformer {
  name = "codex";
  logger?: any;
  private codexAuth?: CodexAuth;
  private cookieJar?: CookieJar;
  private fingerprintManager?: FingerprintManager;
  private baseUrl: string = DEFAULT_CODEX_BASE_URL;
  private lastWarmupAt = 0;

  constructor() {
    this.cookieJar = new CookieJar(console);
    this.fingerprintManager = new FingerprintManager(console);
    if (hasNative()) {
      console.info("[CODEX] Native TLS addon loaded successfully");
    } else {
      console.warn("[CODEX] Native TLS addon not found, using Node.js HTTP fallback");
    }
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    provider: LLMProvider,
    _context: TransformerContext
  ): Promise<{
    body: UnifiedChatRequest;
    config: {
      headers: Record<string, string>;
      url?: URL;
      [key: string]: any;
    };
  }> {
    this.logger?.debug({ model: request.model }, "[CODEX] transformRequestIn called");

    if (!this.codexAuth) {
      await this.loadAuth();
    }

    if (!this.codexAuth) {
      throw new Error(
        "No Codex credentials found. Please authenticate with Codex CLI first."
      );
    }

    if (!this.isTokenValid()) {
      throw new Error(
        "Codex token expired. Please run `codex login` to refresh."
      );
    }

    this.baseUrl = provider.baseUrl || DEFAULT_CODEX_BASE_URL;
    const accountId = this.codexAuth.tokens.account_id;
    const accessToken = this.codexAuth.tokens.access_token;

    await this.warmupIfNeeded(accountId);

    const transformedBody = this.transformRequestBody(request);
    const headers = this.fingerprintManager!.buildHeadersWithContentType(accessToken, accountId);
    headers["x-codex-internal"] = "true";
    
    const cookieHeader = this.cookieJar!.getCookieHeader(accountId);
    if (cookieHeader) {
      headers["cookie"] = cookieHeader;
    }

    return {
      body: transformedBody,
      config: {
        url: new URL(`${this.baseUrl}/responses`),
        codexAccountId: accountId,
        headers: headers,
      },
    };
  }

  async sendRequest(
    requestBody: UnifiedChatRequest,
    config: any,
    provider: LLMProvider,
    _context: TransformerContext
  ): Promise<Response> {
    const url = config.url || new URL(provider.baseUrl || this.baseUrl);
    const headers = config.headers || {};
    const accountId = config.codexAccountId || this.codexAuth?.tokens.account_id;

    if (requestBody.stream !== true) {
      return this.sendNonStreamingRequest(
        typeof url === "string" ? url : url.toString(),
        headers,
        requestBody,
        accountId
      );
    }

    const encoder = new TextEncoder();
    // Convert Codex Responses SSE to Anthropic SSE. The gateway returns our
    // response bytes directly to Claude Code, so raw Codex events would yield
    // zero parseable stream events client-side.
    const converter = new ResponsesToAnthropicStream(requestBody.model || "unknown");
    const stream = new ReadableStream<Uint8Array>({
      start: (controller) => {
        httpPostStream(
          typeof url === "string" ? url : url.toString(),
          headers,
          JSON.stringify(requestBody),
          (chunk, setCookies) => {
            try {
              if (chunk) {
                for (const line of converter.push(chunk)) {
                  controller.enqueue(encoder.encode(line));
                }
              }

              if (setCookies?.length && accountId) {
                this.cookieJar?.capture(accountId, setCookies);
              }

              if (!chunk && setCookies) {
                for (const line of converter.finish()) {
                  controller.enqueue(encoder.encode(line));
                }
                controller.close();
              }
            } catch (error) {
              try {
                controller.error(error);
              } catch {
                // Controller already closed.
              }
            }
          },
          { timeoutSec: 60 * 5 }
        ).catch((error) => {
          try {
            for (const line of converter.finish()) {
              controller.enqueue(encoder.encode(line));
            }
          } catch {
            // Ignore conversion errors on the failure path.
          }
          try {
            controller.error(error);
          } catch {
            // Controller already closed.
          }
        });
      },
      cancel: () => {
        // no-op
      },
    });

    return new Response(stream, {
      status: 200,
      headers: {
        "content-type": "text/event-stream",
      },
    });
  }

  private async sendNonStreamingRequest(
    url: string,
    headers: Record<string, string>,
    requestBody: UnifiedChatRequest,
    accountId?: string
  ): Promise<Response> {
    const streamBody: any = {
      ...(requestBody as any),
      stream: true,
    };

    const response = await httpPost(
      url,
      headers,
      JSON.stringify(streamBody),
      { timeoutSec: 60 * 5 }
    );

    if (response.set_cookie_headers?.length && accountId) {
      this.cookieJar?.capture(accountId, response.set_cookie_headers);
    }

    if (response.status < 200 || response.status >= 300) {
      return new Response(response.body, {
        status: response.status,
        headers: {
          "content-type": response.headers?.["content-type"] || "application/json",
        },
      });
    }

    const normalized = this.convertSseToResponsesPayload(response.body, requestBody.model);
    // The gateway returns our bytes directly to Claude Code, so convert the
    // Codex Responses payload to an Anthropic message (same pattern as the
    // opencode-go transformer). Raw Codex JSON is "JSON but not a Message".
    const anthropic = this.convertResponsesToAnthropic(normalized);
    return new Response(JSON.stringify(anthropic), {
      status: 200,
      headers: {
        "content-type": "application/json",
      },
    });
  }

  private convertResponsesToAnthropic(data: any): any {
    const content: any[] = [];
    let hasToolUse = false;

    for (const item of data.output || []) {
      if (item?.type === "reasoning") {
        const summary = Array.isArray(item.summary)
          ? item.summary
              .map((part: any) =>
                typeof part === "string" ? part : part?.text || ""
              )
              .join("")
          : "";
        // Encrypted-only reasoning carries no visible thought; skip it rather
        // than fabricating content. Visible summaries become thinking blocks.
        if (summary) {
          content.push({
            type: "thinking",
            thinking: summary,
            signature: String(Date.now()),
          });
        }
        continue;
      }

      if (item?.type === "message" && Array.isArray(item.content)) {
        for (const part of item.content) {
          if (part?.type === "output_text" && part.text) {
            content.push({ type: "text", text: part.text });
          }
        }
        continue;
      }

      if (item?.type === "function_call") {
        let input: any = {};
        try {
          input = JSON.parse(item.arguments || "{}");
        } catch {
          input = {};
        }
        content.push({
          type: "tool_use",
          id: item.call_id || item.id || `toolu_${Date.now()}`,
          name: item.name,
          input,
        });
        hasToolUse = true;
      }
    }

    const usage = data.usage || {};
    return {
      id: data.id || `msg_${Date.now()}`,
      type: "message",
      role: "assistant",
      model: data.model || "unknown",
      content: content.length > 0 ? content : [{ type: "text", text: "" }],
      stop_reason:
        data.status === "incomplete"
          ? "max_tokens"
          : hasToolUse
            ? "tool_use"
            : "end_turn",
      stop_sequence: null,
      usage: {
        input_tokens: usage.input_tokens || 0,
        output_tokens: usage.output_tokens || 0,
      },
    };
  }

  private convertSseToResponsesPayload(ssePayload: string, fallbackModel: string): any {
    try {
      const json = JSON.parse(ssePayload);
      if (json && typeof json === "object" && Array.isArray(json.output)) {
        return json;
      }
    } catch {
      // ignore
    }

    const outputItems = new Map<number, any>();
    const functionCallByItemId = new Map<string, any>();
    let latestResponse: any = null;

    const lines = ssePayload.split(/\r?\n/);
    for (const rawLine of lines) {
      const line = rawLine.trim();
      if (!line.startsWith("data: ")) {
        continue;
      }

      const dataStr = line.slice(6).trim();
      if (!dataStr || dataStr === "[DONE]") {
        continue;
      }

      let event: any;
      try {
        event = JSON.parse(dataStr);
      } catch {
        continue;
      }

      if (event?.response) {
        latestResponse = { ...latestResponse, ...event.response };
      }

      if (event.type === "response.output_item.added" && typeof event.output_index === "number") {
        outputItems.set(event.output_index, { ...(event.item || {}) });
        if (event.item?.id) {
          functionCallByItemId.set(event.item.id, outputItems.get(event.output_index));
        }
      }

      if (event.type === "response.output_item.done" && typeof event.output_index === "number") {
        outputItems.set(event.output_index, { ...(event.item || {}) });
        if (event.item?.id) {
          functionCallByItemId.set(event.item.id, outputItems.get(event.output_index));
        }
      }

      if (event.type === "response.output_text.delta" && typeof event.output_index === "number") {
        const existing = outputItems.get(event.output_index) || {
          type: "message",
          content: [{ type: "output_text", text: "" }],
        };

        if (!Array.isArray(existing.content) || existing.content.length === 0) {
          existing.content = [{ type: "output_text", text: "" }];
        }

        const textItem =
          existing.content.find((item: any) => item?.type === "output_text") ||
          existing.content[0];
        textItem.type = "output_text";
        textItem.text = (textItem.text || "") + (event.delta || "");
        outputItems.set(event.output_index, existing);
      }

      if (event.type === "response.function_call_arguments.delta") {
        const byItem = event.item_id ? functionCallByItemId.get(event.item_id) : null;
        if (byItem) {
          byItem.arguments = (byItem.arguments || "") + (event.delta || "");
        } else if (typeof event.output_index === "number") {
          const existing = outputItems.get(event.output_index);
          if (existing && existing.type === "function_call") {
            existing.arguments = (existing.arguments || "") + (event.delta || "");
            outputItems.set(event.output_index, existing);
          }
        }
      }
    }

    const output = Array.from(outputItems.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([, item]) => item)
      .filter(Boolean);

    return {
      id: latestResponse?.id || `resp_${Date.now()}`,
      object: "response",
      model: latestResponse?.model || fallbackModel,
      status: latestResponse?.status || "completed",
      output,
      usage: latestResponse?.usage,
    };
  }

  private async warmupIfNeeded(accountId: string): Promise<void> {
    const now = Date.now();
    if (now - this.lastWarmupAt < CODEX_WARMUP_THROTTLE_MS) {
      return;
    }
    this.lastWarmupAt = now;

    try {
      const headers = this.fingerprintManager!.buildAnonymousHeaders();
      const accessToken = this.codexAuth?.tokens?.access_token;
      if (accessToken) {
        headers["authorization"] = `Bearer ${accessToken}`;
      }
      if (accountId) {
        headers["chatgpt-account-id"] = accountId;
      }
      // The native client does not decompress gzip; ask for identity so the
      // JSON usage body parses.
      delete headers["accept-encoding"];
      const cookieHeader = this.cookieJar!.getCookieHeader(accountId);
      if (cookieHeader) {
        headers["cookie"] = cookieHeader;
      }

      const url = `${this.baseUrl}/usage`;
      const response: HttpResponse = await httpGet(url, headers, { timeoutSec: 15 });

      if (response.set_cookie_headers?.length) {
        this.cookieJar!.capture(accountId, response.set_cookie_headers);
      }

      if (response.status === 200) {
        try {
          const data = JSON.parse(response.body);
          if (data.app_version) {
            this.fingerprintManager!.setClientConfig({
              app_version: data.app_version,
              build_number: data.build_number || "12345",
              chromium_version: data.chromium_version || "136",
            });
            this.logger?.info("[CODEX] Fingerprint updated:", data.app_version);
          }
          await this.writeUsageCache(data);
        } catch (e) {
          // Ignore parse errors
        }
      }
    } catch (error) {
      this.logger?.warn("[CODEX] Warmup request failed:", error);
    }
  }

  /**
   * Persists Codex rate-limit usage for the statusline quota bar
   * (scripts/statusline-opencode.sh reads this; plain curl from a script
   * cannot pass the Cloudflare check, but the gateway native client can).
   */
  private async writeUsageCache(data: any): Promise<void> {
    try {
      const window =
        data?.rate_limit?.primary_window || data?.rate_limit?.secondary_window;
      if (!window || typeof window.used_percent !== "number") {
        return;
      }
      const cache = {
        fetched_at: Date.now(),
        limit_reached: data?.rate_limit?.limit_reached === true,
        plan_type: data?.plan_type || null,
        reset_after_seconds:
          typeof window.reset_after_seconds === "number"
            ? window.reset_after_seconds
            : null,
        used_percent: window.used_percent,
      };
      const tmpFile = `${CODEX_USAGE_CACHE_FILE}.tmp`;
      await fs.writeFile(tmpFile, JSON.stringify(cache), "utf-8");
      await fs.rename(tmpFile, CODEX_USAGE_CACHE_FILE);
    } catch (error) {
      this.logger?.warn("[CODEX] Failed to write usage cache:", error);
    }
  }

  private transformRequestBody(request: UnifiedChatRequest): UnifiedChatRequest {
    const systemMessages = request.messages.filter((msg) => msg.role === "system");
    const conversationalMessages = request.messages.filter((msg) => msg.role !== "system");

    const instructions = this.buildInstructions(systemMessages);
    const input = conversationalMessages.flatMap((msg) => this.transformMessageToInput(msg));

    const transformed: any = {
      model: request.model,
      instructions,
      input,
      stream: request.stream,
      store: false,
      text: {
        format: {
          type: "text",
        },
      },
      parallel_tool_calls: true,
    };

    transformed.reasoning = {
      effort: request.reasoning?.effort || "xhigh",
    };

    if (Array.isArray(request.tools) && request.tools.length > 0) {
      transformed.tools = request.tools.map((tool) => ({
        type: "function",
        name: tool.function.name,
        description: tool.function.description,
        parameters: tool.function.parameters,
      }));
    }

    if (request.tool_choice !== undefined) {
      if (typeof request.tool_choice === "string") {
        transformed.tool_choice = request.tool_choice;
      } else if (
        request.tool_choice &&
        typeof request.tool_choice === "object" &&
        request.tool_choice.type === "function"
      ) {
        transformed.tool_choice = {
          type: "function",
          name: request.tool_choice.function.name,
        };
      }
    }

    return transformed as UnifiedChatRequest;
  }

  private buildInstructions(systemMessages: UnifiedMessage[]): string {
    if (systemMessages.length === 0) {
      return DEFAULT_INSTRUCTIONS;
    }

    const instructions = systemMessages
      .map((message) => {
        if (typeof message.content === "string") {
          return message.content;
        }

        if (Array.isArray(message.content)) {
          return message.content
            .map((part: any) => (part?.type === "text" ? part.text || "" : ""))
            .join("\n");
        }

        return "";
      })
      .filter((text) => text.length > 0)
      .join("\n\n");

    return instructions.length > 0 ? instructions : DEFAULT_INSTRUCTIONS;
  }

  private transformMessageToInput(message: UnifiedMessage): any[] {
    if (message.role === "tool") {
      if (!message.tool_call_id) {
        return [];
      }

      return [
        {
          type: "function_call_output",
          call_id: message.tool_call_id,
          output: this.getStringContent(message.content),
        },
      ];
    }

    if (message.role === "assistant") {
      const items: any[] = [];

      const text = this.extractText(message.content);
      if (text !== null) {
        items.push({
          role: "assistant",
          content: [{ type: "output_text", text }],
        });
      }

      if (Array.isArray(message.tool_calls)) {
        for (const toolCall of message.tool_calls) {
          if (!toolCall?.function?.name) {
            continue;
          }

          items.push({
            type: "function_call",
            call_id: toolCall.id,
            name: toolCall.function.name,
            arguments: this.normalizeToolArguments(toolCall.function.arguments),
          });
        }
      }

      return items;
    }

    if (message.role === "user") {
      const content = this.transformUserContent(message.content);
      if (content.length === 0) {
        return [];
      }

      return [
        {
          role: "user",
          content,
        },
      ];
    }

    return [];
  }

  private transformUserContent(content: UnifiedMessage["content"]): any[] {
    if (typeof content === "string") {
      return [{ type: "input_text", text: content }];
    }

    if (!Array.isArray(content)) {
      return [];
    }

    const parts: any[] = [];
    for (const part of content as any[]) {
      if (part?.type === "text") {
        parts.push({ type: "input_text", text: part.text || "" });
        continue;
      }

      if (part?.type === "image_url" && part.image_url?.url) {
        parts.push({
          type: "input_image",
          image_url: part.image_url.url,
        });
      }
    }

    return parts;
  }

  private extractText(content: UnifiedMessage["content"]): string | null {
    if (typeof content === "string") {
      return content;
    }

    if (!Array.isArray(content)) {
      return null;
    }

    const text = content
      .map((part: any) => (part?.type === "text" ? part.text || "" : ""))
      .join("");

    return text.length > 0 ? text : null;
  }

  private getStringContent(content: UnifiedMessage["content"]): string {
    if (typeof content === "string") {
      return content;
    }

    if (!Array.isArray(content)) {
      return "";
    }

    const text = content
      .map((part: any) => (part?.type === "text" ? part.text || "" : ""))
      .join("");

    if (text.length > 0) {
      return text;
    }

    try {
      return JSON.stringify(content);
    } catch {
      return "";
    }
  }

  private normalizeToolArguments(argumentsValue: any): string {
    if (typeof argumentsValue === "string") {
      return argumentsValue;
    }

    if (argumentsValue === undefined) {
      return "{}";
    }

    try {
      return JSON.stringify(argumentsValue);
    } catch {
      return "{}";
    }
  }

  private isTokenValid(): boolean {
    if (!this.codexAuth?.tokens?.access_token) {
      return false;
    }
    return true;
  }

  private async loadAuth(): Promise<void> {
    try {
      const data = await fs.readFile(AUTH_FILE, "utf-8");
      this.codexAuth = JSON.parse(data);
      this.logger?.debug("[CODEX] Auth loaded from ~/.codex/auth.json");
    } catch (error) {
      this.logger?.error("[CODEX] No auth file found at " + AUTH_FILE);
    }
  }

  async captureCookies(response: HttpResponse): Promise<void> {
    if (!this.codexAuth || !this.cookieJar) return;
    
    if (response.set_cookie_headers?.length) {
      this.cookieJar.capture(this.codexAuth.tokens.account_id, response.set_cookie_headers);
    }
  }
}
