# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

anyclaude is a proxy wrapper for Claude Code that enables using alternative LLM providers (OpenAI, Google, xAI, Azure) through the Anthropic API format. It intercepts Anthropic API calls and translates them to/from the Vercel AI SDK format for the specified provider.

## Architecture

The proxy works by:

1. Spawning a local HTTP server that mimics the Anthropic API
2. Intercepting `/v1/messages` requests containing `<provider>/<model>` format
3. Converting Anthropic message format to AI SDK format
4. Routing to the appropriate provider (OpenAI, Google, xAI, Azure)
5. Converting responses back to Anthropic format
6. Setting `ANTHROPIC_BASE_URL` to point Claude Code at the proxy

Key components:

- `src/main.ts`: Entry point that sets up providers and spawns Claude with proxy
- `src/anthropic-proxy.ts`: HTTP server that handles request/response translation
- `src/convert-anthropic-messages.ts`: Bidirectional message format conversion
- `src/convert-to-anthropic-stream.ts`: Stream response conversion
- `src/json-schema.ts`: Schema adaptation for different providers

## Development Commands

```bash
# Install dependencies
bun install

# Build the project (creates dist/main.js with shebang)
bun run build

# The build command:
# 1. Compiles TypeScript to CommonJS for Node.js compatibility
# 2. Adds Node shebang for CLI execution
```

## Testing

Test the proxy manually:

```bash
# Run in proxy-only mode to get the URL
PROXY_ONLY=true bun run src/main.ts

# Test with a provider
OPENAI_API_KEY=your-key bun run src/main.ts --model openai/gpt-5-mini
```

## Environment Variables

Required for each provider:

- `OPENAI_API_KEY` + optional `OPENAI_API_URL` for OpenAI/OpenRouter
- `GOOGLE_API_KEY` + optional `GOOGLE_API_URL` for Google
- `XAI_API_KEY` + optional `XAI_API_URL` for xAI
- `AZURE_API_KEY` + optional `AZURE_API_URL` for Azure
- `ANTHROPIC_API_KEY` + optional `ANTHROPIC_API_URL` for Anthropic passthrough

Special modes:

- `PROXY_ONLY=true`: Run proxy server without spawning Claude Code
- `ANYCLAUDE_DEBUG=1|2`: Enable debug logging (1=basic, 2=verbose)

- OpenAI's gpt-5 was released in August 2025