// This is just intended to execute Claude Code while setting up a proxy for tokens.

import { createAnthropic } from "@ai-sdk/anthropic";
import { createAzure } from "@ai-sdk/azure";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { createOpenAI } from "@ai-sdk/openai";
import { createXai } from "@ai-sdk/xai";
import { spawn } from "child_process";
import { randomUUID } from "crypto";
import {
  createAnthropicProxy,
  type CreateAnthropicProxyOptions,
} from "./anthropic-proxy";

// providers are supported providers to proxy requests by name.
// Model names are split when requested by `/`. The provider
// name is the first part, and the rest is the model name.
const providers: CreateAnthropicProxyOptions["providers"] = {
  openai: createOpenAI({
    apiKey: process.env.OPENAI_API_KEY,
    baseURL: process.env.OPENAI_API_URL,
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

// Authentication token for the proxy. If the user already provided an
// ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY, reuse it so Claude sends it as
// Authorization: Bearer / X-Api-Key. Otherwise generate an ephemeral token and
// inject it only into the spawned Claude process.
const proxyAuthToken =
  process.env.ANTHROPIC_AUTH_TOKEN ??
  process.env.ANTHROPIC_API_KEY ??
  randomUUID();

const proxyAuthHeaderName = process.env.ANYCLAUDE_AUTH_HEADER ?? "X-AnyClaude-Token";

// Allow overriding host/port via environment variables for flexibility. If
// port is not provided, we bind to an ephemeral random port for parallel safety.
const proxyHost = process.env.ANYCLAUDE_HOST || "127.0.0.1";
const proxyPort = process.env.ANYCLAUDE_PORT
  ? Number(process.env.ANYCLAUDE_PORT)
  : undefined;

(async () => {
  const proxyURL = await createAnthropicProxy({
    providers,
    host: proxyHost,
    port: proxyPort,
    authToken: proxyAuthToken,
    authHeaderName: proxyAuthHeaderName,
  });

  if (process.env.PROXY_ONLY === "true") {
    console.log("Proxy only mode: " + proxyURL);
    return;
  }

  const claudeArgs = process.argv.slice(2);
  const proc = spawn("claude", claudeArgs, {
    env: {
      ...process.env,
      ANTHROPIC_BASE_URL: proxyURL,
      // Ensure the Claude CLI includes our custom header via ANTHROPIC_CUSTOM_HEADERS
      ANTHROPIC_CUSTOM_HEADERS: (() => {
        const existing = process.env.ANTHROPIC_CUSTOM_HEADERS;
        const line = `${proxyAuthHeaderName}: ${proxyAuthToken}`;
        return existing ? `${existing}\n${line}` : line;
      })(),
    },
    stdio: "inherit",
  });
  proc.on("exit", (code) => {
    if (claudeArgs[0] === "-h" || claudeArgs[0] === "--help") {
      console.log("\nCustom Models:");
      console.log("  --model <provider>/<model>      e.g. openai/gpt-5-mini");
    }
    process.exit(code ?? 0);
  });
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
