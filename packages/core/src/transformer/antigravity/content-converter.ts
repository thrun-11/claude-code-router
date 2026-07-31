import { UnifiedMessage } from "@ccr/core/types/llm";

export function convertRole(role: string): string {
  switch (role) {
    case "user":
      return "user";
    case "assistant":
      return "model";
    case "model":
      return "model";
    case "system":
      return "user";
    case "tool":
      return "user";
    default:
      return "user";
  }
}

export function convertContentToParts(
  content: any,
  isClaudeModel: boolean,
  isGeminiModel: boolean
): any[] {
  if (!content) return [];

if (typeof content === "string") {
    return content.trim() ? [{ text: content }] : [];
  }

  if (!Array.isArray(content)) {
    const text = String(content);
    return text.trim() ? [{ text }] : [];
  }

  const parts: any[] = [];

  for (const item of content) {
    if (!item) continue;

    if (item.type === "text") {
      if (typeof item.text === "string" && item.text.trim()) {
        parts.push({ text: item.text });
      }
      continue;
    }

    if (item.type === "image_url" || item.type === "image" || item.type === "document") {
      let data: string;
      let mimeType: string;

      if (item.image_url?.url) {
        const url = item.image_url.url;
        if (url.startsWith("data:")) {
          const match = url.match(/^data:([^;]+);base64,(.+)$/);
          if (match) {
            mimeType = match[1];
            data = match[2];
          } else {
            continue;
          }
        } else {
          continue;
        }
      } else if (item.source?.data) {
        data = item.source.data;
        mimeType = item.source.media_type || (item.type === "document" ? "application/pdf" : "image/png");
      } else {
        continue;
      }

      parts.push({
        inlineData: {
          mimeType,
          data,
        },
      });
      continue;
    }

    if (item.type === "tool_use") {
      const toolId = item.id || `call_${Date.now()}`;
      const toolName = item.name || item.function?.name || "";
      const rawToolArgs =
        item.input !== undefined ? item.input : item.function?.arguments;
      const toolArgs = normalizeToolArgs(rawToolArgs);

      parts.push({
        functionCall: {
          id: toolId,
          name: toolName,
          args: toolArgs,
        },
      });
      continue;
    }

    if (item.type === "tool_result") {
      const toolUseId =
        item.tool_use_id ||
        item.id ||
        item.tool_call_id ||
        `call_${Date.now()}_${Math.random().toString(36).slice(2)}`;
      const toolName =
        item.name || item.tool_name || item.function?.name || toolUseId || "tool_result";

      let responseContent = item.content;
      if (typeof responseContent === "string") {
        responseContent = { result: responseContent };
      } else if (Array.isArray(responseContent)) {
        const texts = responseContent
          .filter((c: any) => c?.type === "text")
          .map((c: any) => c.text)
          .join("\n");
        responseContent = { result: texts || responseContent };
      } else if (typeof responseContent === "object" && responseContent !== null) {
        responseContent = { result: responseContent };
      }

      const functionResponse: any = {
        name: toolName,
        response: responseContent,
      };

      if (isClaudeModel && toolUseId) {
        functionResponse.id = toolUseId;
      }

      parts.push({ functionResponse });
      continue;
    }

    if (item.type === "thinking") {
      parts.push({
        text: item.thinking || item.content || "",
        thought: true,
        ...(item.signature ? { thoughtSignature: item.signature } : {}),
      });
      continue;
    }
  }

  if (parts.length === 0) return [];

  return parts;
}

function normalizeToolArgs(value: any): any {
  if (value === undefined || value === null) {
    return {};
  }

  if (typeof value === "string") {
    const trimmed = value.trim();
    if (!trimmed) return {};
    try {
      return JSON.parse(trimmed);
    } catch {
      return { input: value };
    }
  }

  if (typeof value === "object") {
    return value;
  }

  return { input: value };
}

function normalizeToolResultContent(content: any): any {
  if (content === undefined || content === null) {
    return { text: "" };
  }

  if (typeof content === "string") {
    return { text: content };
  }

  if (Array.isArray(content)) {
    const textContent = content
      .filter((item) => item?.type === "text" && typeof item.text === "string")
      .map((item) => item.text)
      .join("\n");

    if (textContent) {
      return { text: textContent };
    }

    return { text: JSON.stringify(content) };
  }

  if (typeof content === "object") {
    return content;
  }

  return { text: String(content) };
}

function extractToolResultText(content: any): string {
  if (content === undefined || content === null) {
    return "";
  }

  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    const textContent = content
      .filter((item) => item?.type === "text" && typeof item.text === "string")
      .map((item) => item.text)
      .join("\n");

    if (textContent) {
      return textContent;
    }

    return JSON.stringify(content);
  }

  if (typeof content === "object") {
    return JSON.stringify(content);
  }

  return String(content);
}
