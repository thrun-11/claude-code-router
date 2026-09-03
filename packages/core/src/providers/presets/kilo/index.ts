import { defaultProviderAccountConfig, type ProviderPreset } from "@ccr/core/providers/presets/types";

export const kiloProviderPreset: ProviderPreset = {
  account: defaultProviderAccountConfig,
  aliases: ["kilo", "kilo-ai", "kilo-code"],
  defaultModels: [
    "kilo-auto/free",
    "thinkingmachines/inkling:free",
    "minimax/minimax-m2.7:free",
    "minimax/minimax-m3:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "stepfun/step-3.7-flash:free",
    "poolside/laguna-s-2.1:free"
  ],
  endpoints: [
    {
      baseUrl: "https://api.kilo.ai/api/gateway",
      protocols: ["openai_chat_completions"]
    }
  ],
  id: "kilo",
  name: "Kilo AI Gateway",
  websiteUrl: "https://kilocode.ai/"
};
