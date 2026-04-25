import { Transformer, TransformerContext } from "@/types/transformer";
import { LLMProvider, UnifiedChatRequest, UnifiedMessage } from "@/types/llm";
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

export class CodexTransformer implements Transformer {
  name = "codex";
  logger?: any;
  private codexAuth?: CodexAuth;
  private cookieJar?: CookieJar;
  private fingerprintManager?: FingerprintManager;
  private baseUrl: string = DEFAULT_CODEX_BASE_URL;

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
    context: TransformerContext
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
    context: TransformerContext
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
    const stream = new ReadableStream<Uint8Array>({
      start: (controller) => {
        httpPostStream(
          typeof url === "string" ? url : url.toString(),
          headers,
          JSON.stringify(requestBody),
          (chunk, setCookies) => {
            if (chunk) {
              controller.enqueue(encoder.encode(chunk));
            }

            if (setCookies?.length && accountId) {
              this.cookieJar?.capture(accountId, setCookies);
            }

            if (!chunk && setCookies) {
              controller.close();
            }
          },
          { timeoutSec: 60 * 5 }
        ).catch((error) => {
          controller.error(error);
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
    return new Response(JSON.stringify(normalized), {
      status: 200,
      headers: {
        "content-type": "application/json",
      },
    });
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
    try {
      const headers = this.fingerprintManager!.buildAnonymousHeaders();
      const cookieHeader = this.cookieJar!.getCookieHeader(accountId);
      if (cookieHeader) {
        headers["cookie"] = cookieHeader;
      }

      const url = `${this.baseUrl}/codex/usage`;
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
        } catch (e) {
          // Ignore parse errors
        }
      }
    } catch (error) {
      this.logger?.warn("[CODEX] Warmup request failed:", error);
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
