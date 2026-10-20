# Repository guidance for coding agents

## Product goal

Build `llmrouter`, a small, readable Python 3.11+ library that routes application LLM calls across configurable free and low-cost providers. Follow `llmrouter-phase1-spec.md` and the milestone sequence in `IMPLEMENTATION_PLAN.md`.

## Scope and architecture

- Phase 1 is a local library using SQLite and a CLI. Do not build HTTP hosting or D1 storage unless explicitly asked to start Phase 2.
- Keep persistence behind `src/llmrouter/store/protocol.py`; SQL belongs under `src/llmrouter/store/`.
- Keep providers behind a shared adapter contract. Prefer plain `httpx` to provider SDKs.
- Use strict Pydantic v2 models and reject unknown YAML fields.
- Secrets are environment variable names in config; secret values must never be stored or logged.
- Never invent provider model names, prices, rate limits, or data policies. Example values must be marked as placeholders.
- Store UTC timestamps as integer epoch seconds.

## Implementation workflow

- Work milestone by milestone. Update `IMPLEMENTATION_PLAN.md` when a milestone starts or finishes.
- Before changing the public API or persisted schema, explain the impact and ask the user.
- Prefer focused, typed modules with comments explaining non-obvious design choices.
- Add offline tests using fake adapters for core logic. Keep real-provider integration tests opt-in.
- Run relevant tests and report exactly what was run and the result.
- Keep local databases, `.env`, and generated artifacts out of version control.

## Local setup

Install in editable mode with `python -m pip install -e .`. The Phase 1 database is local SQLite. The supplied D1 binding is reserved for a later adapter and must not be treated as permission to access or modify a remote database.
