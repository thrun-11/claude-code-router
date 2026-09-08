import { UnifiedChatRequest } from "@ccr/core/types/llm";
import {
  ANTIGRAVITY_ENDPOINTS,
  isThinkingModel,
  getFallbackModel,
} from "./constants";
import { buildCloudCodeRequest, buildHeaders } from "./request-builder";
import { streamSSEResponse, accumulateSSEToResponse } from "./sse-parser";
import { convertGoogleToAnthropic } from "./response-converter";
import { AuthManager } from "./auth-manager";
import { GoogleRequest } from "./request-converter";

const MAX_RETRIES = 5;
const DEFAULT_COOLDOWN_MS = 10000;

export class CloudCodeClient {
  private authManager: AuthManager;
  private logger: any;

  constructor(authManager: AuthManager) {
    this.authManager = authManager;
  }

  setLogger(logger: any) {
    this.logger = logger;
  }

  async sendMessage(
    request: UnifiedChatRequest,
    _options?: { fallbackEnabled?: boolean }
  ): Promise<Response> {
    const token = await this.authManager.getActiveToken();
    if (!token) {
      throw new Error(
        "No Antigravity credentials. Run `ccr antigravity add` to add your account."
      );
    }

    const account = await this.authManager.getActiveAccount();
    if (!account) {
      throw new Error("No active account found.");
    }

    const googleRequest = (request as any).googleRequest as GoogleRequest;

    if (request.stream) {
      return this.sendMessageStream(request, token, account, googleRequest);
    }

    return this.sendNonStreaming(request, token, account, googleRequest);
  }

  private async sendNonStreaming(
    request: UnifiedChatRequest,
    token: string,
    account: any,
    googleRequest: GoogleRequest
  ): Promise<Response> {
    const model = (request as any).model || "claude-sonnet-4-6-thinking";
    const isThinking = isThinkingModel(model);

    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      for (const endpoint of ANTIGRAVITY_ENDPOINTS) {
        try {
          const url = isThinking
            ? `${endpoint}/v1internal:streamGenerateContent?alt=sse`
            : `${endpoint}/v1internal:generateContent`;

          const payload = buildCloudCodeRequest(googleRequest, account.projectId || "rising-fact-p41fc", token, {
            model,
            temperature: (request as any).temperature,
            topP: (request as any).top_p,
            topK: (request as any).top_k,
          });

          const sessionId = payload.request.sessionId;
          const headers = buildHeaders(token, model, sessionId);

          const response = await fetch(url, {
            method: "POST",
            headers,
            body: JSON.stringify(payload),
          });

          if (!response.ok) {
            const errorText = await response.text();
            this.logger?.warn(`[CloudCode] ${response.status} at ${endpoint}: ${errorText.substring(0, 200)}`);

            if (response.status === 401) {
              this.authManager.getTokenForAccount(account.email);
              continue;
            }

            if (response.status === 429) {
              await sleep(DEFAULT_COOLDOWN_MS);
              continue;
            }

            if (response.status >= 500) {
              await sleep(1000);
              continue;
            }

            return new Response(JSON.stringify({ error: errorText }), {
              status: response.status,
              headers: { "content-type": "application/json" },
            });
          }

          if (isThinking) {
            const resultResponse = await accumulateSSEToResponse(response, model);
            const resultText = await resultResponse.text();
            return new Response(resultText, {
              status: 200,
              headers: { "content-type": "application/json" },
            });
          }

          const data = await response.json();
          const anthropic = convertGoogleToAnthropic(data, model);
          return new Response(JSON.stringify(anthropic), {
            status: 200,
            headers: { "content-type": "application/json" },
          });
        } catch (err) {
          this.logger?.warn(`[CloudCode] Error at ${endpoint}: ${err}`);
          continue;
        }
      }
    }

    const fallback = getFallbackModel(model);
    if (fallback) {
      const fallbackRequest = { ...request, model: fallback };
      return this.sendMessage(fallbackRequest, { fallbackEnabled: false });
    }

    throw new Error("Max retries exceeded");
  }

  private async sendMessageStream(
    request: UnifiedChatRequest,
    token: string,
    account: any,
    googleRequest: GoogleRequest
  ): Promise<Response> {
    const model = (request as any).model || "claude-sonnet-4-6-thinking";
    const logger = this.logger;

    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
          for (const endpoint of ANTIGRAVITY_ENDPOINTS) {
            try {
              const url = `${endpoint}/v1internal:streamGenerateContent?alt=sse`;
              const payload = buildCloudCodeRequest(googleRequest, account.projectId || "rising-fact-p41fc", token, {
                model,
                temperature: (request as any).temperature,
                topP: (request as any).top_p,
                topK: (request as any).top_k,
              });

              const sessionId = payload.request.sessionId;
              const headers = buildHeaders(token, model, sessionId);

              const response = await fetch(url, {
                method: "POST",
                headers,
                body: JSON.stringify(payload),
              });

              if (!response.ok) {
                const errorText = await response.text();
                logger?.warn(`[CloudCode] Stream ${response.status} at ${endpoint}`);

                if (response.status === 401) {
                  continue;
                }

                if (response.status === 429) {
                  // If it's a 429, we should only retry a couple of times, but for now we'll respect the loop.
                  // However, if we hit max retries, we need to send an explicit error event.
                  await sleep(DEFAULT_COOLDOWN_MS);
                  continue;
                }

                if (response.status >= 500) {
                  await sleep(1000);
                  continue;
                }

                // For other non-retryable errors (e.g. 400), send an error event
                const errorEvent = `event: error\ndata: ${JSON.stringify({
                  type: "error",
                  error: {
                    type: "api_error",
                    message: `Antigravity API Error: ${response.status} - ${errorText.substring(0, 200)}`
                  }
                })}\n\n`;
                controller.enqueue(encoder.encode(errorEvent));
                controller.close();
                return;
              }

              const eventGenerator = streamSSEResponse(response, model);
              for await (const event of eventGenerator) {
                const sseLine = `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`;
                controller.enqueue(encoder.encode(sseLine));
              }

              controller.close();
              return;
            } catch (err) {
              logger?.warn(`[CloudCode] Stream error: ${err}`);
              continue;
            }
          }
        }

        // If we exhausted all retries (e.g., persistent 429s), we must tell Claude Code!
        const timeoutError = `event: error\ndata: ${JSON.stringify({
          type: "error",
          error: {
            type: "rate_limit_error",
            message: "Antigravity API Quota Exceeded or Max Retries Reached (429). Please check your G1 Credits at https://antigravity.google/g1-credits"
          }
        })}\n\n`;
        controller.enqueue(encoder.encode(timeoutError));
        controller.close();
      },
    });

    return new Response(stream, {
      status: 200,
      headers: {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      },
    });
  }
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}