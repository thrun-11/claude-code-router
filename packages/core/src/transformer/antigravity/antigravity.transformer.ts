import { Transformer, TransformerContext } from "@ccr/core/types/transformer";
import { LLMProvider, UnifiedChatRequest } from "@ccr/core/types/llm";
import { convertAnthropicToGoogle } from "./request-converter";
import { resolveTieredModel } from "./constants";
import { CloudCodeClient } from "./cloudcode-client";
import { AuthManager } from "./auth-manager";
import { sseToResponse } from "./sse-parser";

export class AntigravityTransformer implements Transformer {
  name = "antigravity";

  private cloudCodeClient: CloudCodeClient;
  private authManager: AuthManager;
  private _logger: any;

  constructor() {
    this.authManager = new AuthManager();
    this.cloudCodeClient = new CloudCodeClient(this.authManager);
  }

  set logger(logger: any) {
    this._logger = logger;
    this.authManager.setLogger(logger);
    this.cloudCodeClient.setLogger(logger);
  }

  get logger(): any {
    return this._logger;
  }

  async transformRequestIn(
    request: UnifiedChatRequest,
    _provider: LLMProvider,
    _context: TransformerContext
  ): Promise<{
    body: UnifiedChatRequest;
    config: {
      headers?: Record<string, string>;
      url?: URL;
      [key: string]: any;
    };
  }> {
    this._logger?.debug({ model: request.model }, "[Antigravity] transformRequestIn");

    await this.authManager.initialize();

    if (!this.authManager.hasAccounts()) {
      throw new Error(
        "No Antigravity credentials. Run `ccr antigravity add` to add your Google account."
      );
    }

    const token = await this.authManager.getActiveToken();
    if (!token) {
      throw new Error(
        "Antigravity token expired or invalid. Run `ccr antigravity add` to re-authenticate."
      );
    }

    // Resolve tiered Flash aliases (e.g. gemini-3.8-flash-high -> gemini-3.8-flash-tiered + ThinkingLevel HIGH)
    const resolved = resolveTieredModel(request.model || "");
    const normalizedRequest =
      resolved.backendModel !== request.model ? { ...request, model: resolved.backendModel } : request;

    const googleRequest = convertAnthropicToGoogle(normalizedRequest);
    if (resolved.thinkingLevel) {
      googleRequest.generationConfig = googleRequest.generationConfig || {};
      googleRequest.generationConfig.thinkingConfig = {
        ...(googleRequest.generationConfig.thinkingConfig || {}),
        thinkingLevel: resolved.thinkingLevel,
      };
    }

    return {
      body: {
        ...normalizedRequest,
        googleRequest,
      } as any,
      config: {
        url: new URL("https://daily-cloudcode-pa.googleapis.com"),
        headers: {
          Authorization: `Bearer ${token}`,
          "x-ccr-transformed": "antigravity",
        },
      },
    };
  }

  async sendRequest(
    requestBody: UnifiedChatRequest,
    _config: any,
    _provider: LLMProvider,
    _context: TransformerContext
  ): Promise<Response> {
    await this.authManager.initialize();
    const response = await this.cloudCodeClient.sendMessage(requestBody);
    return response;
  }

  async transformResponseIn(
    response: Response,
    context?: TransformerContext
  ): Promise<Response> {
    const isStream = response.headers
      .get("Content-Type")
      ?.includes("text/event-stream");

    const body = context?.req?.body as UnifiedChatRequest | undefined;
    const model = body?.model || "claude-sonnet-4-6-thinking";

    if (isStream) {
      if (!response.body) {
        throw new Error("Stream response body is null");
      }
      return sseToResponse(response, model);
    }

    return response;
  }
}