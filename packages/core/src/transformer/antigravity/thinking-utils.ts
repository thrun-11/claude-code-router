import { MIN_SIGNATURE_LENGTH } from "./constants";

const signatureCache = new Map<string, { sig: string; ts: number }>();
const GEMINI_SIGNATURE_CACHE_TTL_MS = 2 * 60 * 60 * 1000;

export function cacheThinkingSignature(sig: string, _modelFamily: string): void {
  signatureCache.set(sig, { sig, ts: Date.now() });
}

export function getCachedSignature(key: string): string | null {
  const entry = signatureCache.get(key);
  if (entry && Date.now() - entry.ts < GEMINI_SIGNATURE_CACHE_TTL_MS) {
    return entry.sig;
  }
  signatureCache.delete(key);
  return null;
}

export function hasGeminiHistory(messages: any[]): boolean {
  return messages.some((msg) => {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      return msg.content.some(
        (c: any) => c.type === "thinking" || c.thought === true
      );
    }
    return false;
  });
}

export function hasUnsignedThinkingBlocks(messages: any[]): boolean {
  return messages.some((msg) => {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      return msg.content.some((c: any) => {
        if (c.type === "thinking") {
          return !c.signature || c.signature.length < MIN_SIGNATURE_LENGTH;
        }
        return false;
      });
    }
    return false;
  });
}

export function needsThinkingRecovery(messages: any[]): boolean {
  if (!Array.isArray(messages)) return false;

  let foundToolResult = false;
  let foundTrailingThinking = false;
  let foundUnsignedThinking = false;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];
    if (msg.role === "tool") {
      foundToolResult = true;
    }

    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const c of msg.content) {
        if (c.type === "tool_use") {
          if (foundTrailingThinking) return true;
        }
        if (c.type === "thinking") {
          foundTrailingThinking = foundToolResult;
          if (!c.signature || c.signature.length < MIN_SIGNATURE_LENGTH) {
            foundUnsignedThinking = true;
          }
        }
      }
    }
  }

  return foundUnsignedThinking || foundTrailingThinking;
}

export function closeToolLoopForThinking(messages: any[], _modelType: string): any[] {
  return messages.map((msg) => {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      const filtered = msg.content.filter((c: any) => {
        if (c.type === "thinking") {
          return c.signature && c.signature.length >= MIN_SIGNATURE_LENGTH;
        }
        return true;
      });
      return { ...msg, content: filtered };
    }
    return msg;
  });
}

export function filterUnsignedThinkingBlocks(contents: any[]): any[] {
  return contents.map((content) => {
    if (content.role === "model" && Array.isArray(content.parts)) {
      const filtered = content.parts.filter((p: any) => {
        if (p.thought === true) {
          return (
            p.thoughtSignature && p.thoughtSignature.length >= MIN_SIGNATURE_LENGTH
          );
        }
        return true;
      });
      if (filtered.length === 0) {
        return { ...content, parts: [{ text: "." }] };
      }
      return { ...content, parts: filtered };
    }
    return content;
  });
}

export function reorderAssistantContent(parts: any[]): any[] {
  const thinking: any[] = [];
  const text: any[] = [];
  const toolUse: any[] = [];
  const other: any[] = [];

  for (const p of parts) {
    if (p.thought === true) thinking.push(p);
    else if (p.text !== undefined) text.push(p);
    else if (p.functionCall) toolUse.push(p);
    else other.push(p);
  }

  return [...thinking, ...text, ...toolUse, ...other];
}

export function restoreThinkingSignatures(parts: any[]): any[] {
  return parts.map((p: any) => {
    if (p.type === "thinking" && (!p.signature || p.signature.length < MIN_SIGNATURE_LENGTH)) {
      const cached = getCachedSignature(p.thinking || p.content || "");
      if (cached) {
        return { ...p, signature: cached };
      }
    }
    return p;
  });
}

export function removeTrailingThinkingBlocks(parts: any[]): any[] {
  if (!Array.isArray(parts)) return parts;
  const lastIndex = parts.length - 1;
  for (let i = lastIndex; i >= 0; i--) {
    const p = parts[i];
    if (p.type === "thinking" || p.thought === true) {
      if (i === lastIndex) {
        continue;
      }
      break;
    }
    break;
  }
  return parts;
}

export function clampGeminiThinkingBudget(
  modelName: string,
  budget?: number
): number {
  const maxBudget = 24576;
  const defaultBudget = 32000;
  if (budget !== undefined) {
    return Math.min(budget, maxBudget);
  }
  return defaultBudget;
}