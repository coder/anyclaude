# anyclaude

![NPM Version](https://img.shields.io/npm/v/anyclaude)

Use Claude Code with OpenAI, Google, xAI, and other providers.

- Extremely simple setup - just a basic command wrapper
- Uses the AI SDK for simple support of new providers
- Works with Claude Code GitHub Actions

<img src="./demo.png" width="65%">

## Get Started

```sh
# Use your favorite package manager (bun, pnpm, and npm are supported)
$ pnpm install -g anyclaude

# anyclaude is a wrapper for the Claude CLI
# `openai/`, `google/`, `xai/`, and `anthropic/` are supported
$ anyclaude --model openai/gpt-5-mini
```

Switch models in the Claude UI with `/model openai/gpt-5-mini`.

## FAQ

### What providers are supported?

See [the providers](./src/main.ts#L17) for the implementation.

- `GOOGLE_API_KEY` supports `google/*` models.
- `OPENAI_API_KEY` supports `openai/*` models.
- `XAI_API_KEY` supports `xai/*` models.

Set a custom OpenAI endpoint with `OPENAI_API_URL` to use OpenRouter

### How does this work?

Claude Code has added support for customizing the Anthropic endpoint with `ANTHROPIC_BASE_URL`.

anyclaude spawns a simple HTTP server that translates between Anthropic's format and the [AI SDK](https://github.com/vercel/ai) format, enabling support for any [AI SDK](https://github.com/vercel/ai) provider (e.g., Google, OpenAI, etc.)

## Do other models work better in Claude Code?

Not really, but it's fun to experiment with them.

`ANTHROPIC_MODEL` and `ANTHROPIC_SMALL_MODEL` are supported with the `<provider>/` syntax.
