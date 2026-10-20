# LLM Router implementation plan

This project is a local-first Python library for routing calls to free and low-cost AI models. Phase 1 uses SQLite and a small CLI. The supplied Cloudflare D1 binding is a Phase 2 deployment target; it does not change Phase 1 storage.

## Decisions already reflected in the spec

- Each attempt reserves worst-case budget inside a transaction. Success settles it; token-free failure releases it; expired reservations are swept after crashes. Free models reserve zero.
- Daily request and token limits share the `day` usage window and use separate counters. `rpd` checks requests; `tpd` checks input plus output tokens.
- Quota windows are currently fixed and UTC-aligned. Quota counters are atomic across SQLite clients; per-model concurrency semaphores coordinate Router instances within one Python process.
- The router keeps persistence behind a `Store` interface. Phase 1 uses SQLite; Phase 2 may supply a strongly consistent D1-backed implementation.
- Keep provider limits, prices, and model IDs as verified configuration, never invented defaults.

## Milestones

1. **Foundation** — package metadata, strict config validation, SQLite migrations/store, examples, and setup docs. Implemented.
2. **Execution core** — typed provider errors, normalized request/response types, scripted fake adapter, retry and fallback executor, and deterministic offline tests. Implemented.
3. **Limits and selection** — atomic quota reservations, concurrency controls, cooldowns, candidate selection, and status data. Implemented.
4. **OpenAI-compatible adapter** — first real HTTP adapter and a local end-to-end call. Implemented; 13 offline tests pass.
5. **Structured output** — JSON-mode prompting, parsing, Pydantic/JSON Schema validation, correction retries, and tier escalation. Implemented; offline coverage includes schema and Pydantic validation, correction, fallback mode, and escalation.
6. **Cache and budgets** — stable cache keys, cache controls, transactional budget reservations/settlement/recovery, per-call cost ceilings, and monthly reporting. Implemented; 23 offline tests pass.
7. **Observability and CLI** — structured redacted logs and status/report/models/cache/test-model/replay commands. Implemented; 27 offline tests pass.
8. **Additional adapters** — Anthropic, Gemini, and Cloudflare using focused `httpx` calls. Implemented; mocked REST tests cover request mapping, usage, and error classification.
9. **Async and batch** — async calls, bounded concurrency, ordered per-item outcomes, progress, and cache-based resumability. Implemented; Router async calls offload the shared sync pipeline to a worker pool, and provider adapters expose native `acomplete()` methods for direct async use.
10. **Hardening and docs** — implemented with concurrent hard-cap coverage, crash-expiry recovery, cache-resume coverage, seeded failure injection, report checks, an async batch example, endpoint/secret handling hardening, owner-only local artifacts, privacy-aware cache invalidation, and retry deadline/backoff behavior. Full suite: 50 offline tests.

## Phase 1 acceptance review

- Validated `call_json`, YAML-only routing changes, 429 fallback/cooldown, private-data filtering, and report metrics have offline coverage.
- Fifty-thread quota and hard-budget checks verify atomic enforcement; reopening SQLite expires abandoned budget holds and preserves settled reporting.
- Re-running a successful batch verifies cache hits and no additional provider calls.
- Setup, smoke-call, structured-output, and async batch instructions are in the README. Provider checks remain opt-in.
- Known boundaries: `settings.max_wait_s` is reserved but not used to wait for quota capacity; the router returns `NoModelAvailable` when no slot is available. Router async methods offload the existing sync pipeline to a per-Router worker pool; adapter `acomplete()` methods are also available for direct async requests. Embeddings and provider batch endpoints remain outside Phase 1.

## Phase 2: D1 target

The intended Cloudflare binding supplied for later deployment is:

```json
{
  "d1_databases": [{
    "binding": "DB",
    "database_name": "llmrouter",
    "database_id": "b1ab8346-1516-42c6-a96c-68bbecb25bb0"
  }]
}
```

This is a deployment note only. No Wrangler configuration or remote database access is added during Phase 1. Before Phase 2, implement a D1 `Store` that preserves transaction and single-writer/strong-consistency requirements for quota and budget reservations.

## Working agreement

- Work one milestone at a time and keep the code readable and typed.
- Do not change the public API or persisted data model without asking first.
- Keep credentials out of config, the database, and logs.
- Use offline fake-provider tests for core routing behavior; real-provider checks require explicit credentials and should be opt-in.
- Update this file as milestone status changes.
