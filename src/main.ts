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
import yargsParser from "yargs-parser";

const FLAGS = {
  reasoningEffort: {
    long: "reasoning-effort",
    short: "e",
    values: ["minimal", "low", "medium", "high"] as const,
  },
  serviceTier: {
    long: "service-tier",
    short: "t",
    values: ["flex", "priority"] as const,
  },
} as const;

function parseAnyclaudeFlags(rawArgs: string[]) {
  const parsed = yargsParser(rawArgs, {
    configuration: {
      "unknown-options-as-args": false,
      "halt-at-non-option": false,
      "camel-case-expansion": false,
      "dot-notation": false,
    },
  });
  const reasoningEffort = (parsed[FLAGS.reasoningEffort.long] ??
    parsed[FLAGS.reasoningEffort.short]) as string | undefined;
  const serviceTier = (parsed[FLAGS.serviceTier.long] ??
    parsed[FLAGS.serviceTier.short]) as string | undefined;
  const specs = Object.values(FLAGS);
  const filteredArgs: string[] = [];
  let helpRequested = false;
  let i = 0;
  let passthrough = false;
  while (i < rawArgs.length) {
    const arg = rawArgs[i]!;
    if (passthrough) {
      filteredArgs.push(arg);
      i++;
      continue;
    }
    if (arg === "--") {
      passthrough = true;
      filteredArgs.push(arg);
      i++;
      continue;
    }
    if (arg === "-h" || arg === "--help") helpRequested = true;
    let matched = false;
    for (const spec of specs) {
      const long = `--${spec.long}`;
      const short = `-${spec.short}`;
      if (arg === long || arg === short) {
        i += 2;
        matched = true;
        break;
      }
      if (arg.startsWith(long + "=") || arg.startsWith(short + "=")) {
        i += 1;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    filteredArgs.push(arg);
    i++;
  }
  return { reasoningEffort, serviceTier, filteredArgs, helpRequested };
}

const rawArgs = process.argv.slice(2);
const { reasoningEffort, serviceTier, filteredArgs, helpRequested } =
  parseAnyclaudeFlags(rawArgs);

for (const [key, spec] of Object.entries(FLAGS) as Array<
  [keyof typeof FLAGS, (typeof FLAGS)[keyof typeof FLAGS]]
>) {
  const val = (key === "reasoningEffort" ? reasoningEffort : serviceTier) as
    | string
    | undefined;
  if (val) {
    const allowed = new Set(spec.values as readonly string[]);
    if (!allowed.has(val as any)) {
      console.error(`Invalid ${spec.long}. Use ${spec.values.join("|")}.`);
      process.exit(1);
    }
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
        if (typeof maxTokens !== "undefined")
          body.max_completion_tokens = maxTokens;
        if (reasoningEffort) body.reasoning = { effort: reasoningEffort };
        if (serviceTier) body.service_tier = serviceTier;
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

const params = [
  `proxy=${proxyURL}`,
  ...(
    Object.entries({ reasoningEffort, serviceTier }) as Array<
      [keyof typeof FLAGS, string | undefined]
    >
  ).map(([k, v]) => (v ? `${FLAGS[k].long}=${v}` : undefined)),
]
  .filter(Boolean)
  .join(" ");
console.log(`[anyclaude] ${params}`);

if (process.env.PROXY_ONLY === "true") {
  console.log("Proxy only mode: " + proxyURL);
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
    if (helpRequested) {
      console.log("\nanyclaude flags:");
      console.log("  --model <provider>/<model>      e.g. openai/gpt-5");
      for (const spec of Object.values(FLAGS)) {
        const vals = spec.values.join("|");
        console.log(`  --${spec.long}, -${spec.short} <${vals}>`);
      }
    }

    process.exit(code);
  });
}
