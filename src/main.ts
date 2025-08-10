// This is just intended to execute Claude Code while setting up a proxy for tokens.

import { createAnthropic } from "@ai-sdk/anthropic";
import { createAzure } from "@ai-sdk/azure";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { createXai } from "@ai-sdk/xai";
import { spawn } from "child_process";
import {
  createAnthropicProxy,
  type CreateAnthropicProxyOptions,
} from "./anthropic-proxy";

const rawArgs = process.argv.slice(2);
let reasoningEffort: string | undefined;
const filteredArgs: string[] = [];
for (let i = 0; i < rawArgs.length; i++) {
  const arg = rawArgs[i]!;
  if (arg === "--reasoning-effort" || arg === "-e") {
    const val = rawArgs[i + 1];
    if (!val) {
      console.error("Missing value for --reasoning-effort");
      process.exit(1);
    }
    reasoningEffort = val;
    i++;
    continue;
  }
  const m = arg.match(/^--reasoning-effort=(.+)$/);
  if (m) {
    reasoningEffort = m[1]!;
    continue;
  }
  filteredArgs.push(arg);
}
if (reasoningEffort) {
  const allowed = new Set(["minimal", "low", "medium", "high"]);
  if (!allowed.has(reasoningEffort)) {
    console.error("Invalid reasoning effort. Use minimal|low|medium|high.");
    process.exit(1);
  }
}

// providers are supported providers to proxy requests by name.
// Model names are split when requested by `/`. The provider
// name is the first part, and the rest is the model name.
const providers: CreateAnthropicProxyOptions["providers"] = {
  openai: createOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_API_URL,
    fetch: (async (url, init) => {
      if (init?.body && typeof init.body === "string") {
        const body = JSON.parse(init.body);
        const maxTokens = body.max_tokens;
        delete body["max_tokens"];
        if (typeof maxTokens !== "undefined") body.max_completion_tokens = maxTokens;
        if (reasoningEffort) body.reasoning = { effort: reasoningEffort };
        init.body = JSON.stringify(body);
      }
      return globalThis.fetch(url, init);
    }) as typeof fetch,
  }),
  azure: createAzure({
    apiKey: process.env.AZURE_API_KEY,
    baseURL: process.env.AZURE_API_URL,
  }),
  google: createGoogleGenerativeAI({
    apiKey: process.env.GOOGLE_API_KEY,
    baseURL: process.env.GOOGLE_API_URL,
  }),
  xai: createXai({
    apiKey: process.env.XAI_API_KEY,
    baseURL: process.env.XAI_API_URL,
  }),
};

// We exclude this by default, because the Claude Code
// API key is not supported by Anthropic endpoints.
if (process.env.ANTHROPIC_API_KEY) {
  providers.anthropic = createAnthropic({
    apiKey: process.env.ANTHROPIC_API_KEY,
    baseURL: process.env.ANTHROPIC_API_URL,
  });
}

const proxyURL = createAnthropicProxy({
  providers,
});

if (process.env.PROXY_ONLY === "true") {
  console.log("Proxy only mode: "+proxyURL);
} else {
  const claudeArgs = filteredArgs;
  const proc = spawn("claude", claudeArgs, {
    env: {
      ...process.env,
      ANTHROPIC_BASE_URL: proxyURL,
    },
    stdio: "inherit",
  });
  proc.on("exit", (code) => {
    if (claudeArgs[0] === "-h" || claudeArgs[0] === "--help") {
      console.log("\nCustom Models:")
      console.log("  --model <provider>/<model>      e.g. openai/o3");
      console.log("  --reasoning-effort, -e <minimal|low|medium|high>");
    }

    process.exit(code);
  });
}

