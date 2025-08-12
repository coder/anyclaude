# Repository Guidelines

## Project Structure & Module Organization

- `src/`: TypeScript sources. Key files: `main.ts` (CLI entry), `anthropic-proxy.ts` (HTTP proxy), `convert-*.ts` (format converters), `detect-mimetype.ts`, `json-schema.ts`.
- `dist/`: Bundled CLI entry `main.js` (created by build).
- `package.json`, `bun.lock`: Bun-based build and deps; `bin.anyclaude` points to `dist/main.js`.
- `flake.nix`, `.envrc`: Nix/direnv developer shell and formatters.
- Assets/config: `README.md`, `CLAUDE.md`, `tsconfig.json`, `demo.png`.

## Build, Test, and Development Commands

- Install: `bun install`.
- Build: `bun run build` (outputs `dist/main.js` with Node shebang).
- Run CLI (after build): `./dist/main.js --model openai/gpt-5-mini`.
- Dev run (no build):
  - `PROXY_ONLY=true bun run src/main.ts` (prints proxy URL)
  - `OPENAI_API_KEY=... bun run src/main.ts --model openai/gpt-5-mini`
- Nix shell: `direnv allow` (or `nix develop`); format Nix/shell files with `nix fmt`.

## Coding Style & Naming Conventions

- Language: TypeScript (ESNext, strict mode enabled).
- Indentation: 2 spaces; keep lines reasonable; use explicit imports (`verbatimModuleSyntax`).
- Files: kebab-case `.ts` (e.g., `convert-to-anthropic-stream.ts`).
- Names: `camelCase` for variables/functions; `PascalCase` for types/classes.
- Exports: prefer named exports; keep modules single‑purpose and small.

## Testing Guidelines

- No test runner is configured yet. If adding tests:
  - Place under `src/**/*.test.ts` or `src/__tests__/`.
  - Prioritize pure units (converters, schema, MIME detection). Avoid live provider calls by default; gate with env vars.
  - Add a `test` script in `package.json` and document how to run it in `README.md`.

## Commit & Pull Request Guidelines

- Commits: imperative, concise subjects (e.g., "Add Nix development environment and Claude guidance file"). Include rationale in the body when helpful.
- PRs: clear description, linked issues, commands used to verify (with relevant env vars), and expected behavior. Avoid committing secrets; scrub logs.
- Keep scope small; update `README.md`/`CLAUDE.md` when behavior or env vars change.

## Security & Configuration Tips

- Never commit API keys. Use `direnv` for local secrets and keep `.envrc` minimal.
- Provider envs: `OPENAI_*`, `GOOGLE_*`, `XAI_*`, `AZURE_*`, optional `ANTHROPIC_*`. Use `PROXY_ONLY=true` to inspect the proxy without launching Claude.
