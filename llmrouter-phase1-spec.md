# LLM Router: Phase 1 Build Specification

*A generic, reusable model-routing library. Hand this document to a coding assistant as the full brief.*

## 1. Purpose

Build a Python library, `llmrouter`, that is the **single entry point for every LLM call** in a product or set of products. Application code never talks to a provider SDK directly. It calls the router with a *task name* and the router decides which model to use, enforces rate limits and budgets, retries and falls back on failure, validates the output, caches results, and logs everything.

The main goals:

- **Cost control:** use free tiers and cheap models first, escalate to expensive ones only when needed, and never exceed a monthly budget.
- **Resilience:** one provider failing or rate-limiting must not break a pipeline.
- **Swappability:** changing a model or provider is a config change, not a code change.
- **Observability:** every call is recorded so quality, cost and reliability can be compared across models.
- **Learnability:** the code is small, readable and well commented, because the owner wants to understand and modify it.

## 2. Scope

**Phase 1 (this document):** an importable Python library with a local SQLite store and a small CLI. Used by batch scripts, agents and cron jobs.

**Out of scope for Phase 1** (but the design must not block them):

- Running as a hosted HTTP service (Phase 2, possibly FastAPI on Cloudflare Python Workers or a container).
- A web UI.
- Fine-tuning, model hosting, or training.
- Multi-user auth and billing.

## 3. Design principles

1. **Config over code.** Models, limits, prices, tiers and task routes live in a YAML file that is loaded into SQLite. Adding a model means editing config.
2. **Secrets never in the database or config.** Config holds the *name* of an environment variable (`api_key_env`), never the key.
3. **Small, typed, tested.** Python 3.11+, type hints throughout, `pydantic` for models and validation, `pytest` for tests.
4. **Fail loudly on bugs, quietly on transient errors.** Misconfiguration raises clear errors. Rate limits and outages are handled automatically.
5. **Provider-agnostic interface.** One internal request/response format. Each provider is an adapter behind a common interface.
6. **No hardcoded prices or limits.** Provider limits and prices change often. Ship the config as an example the owner must verify and fill in. Do not invent numbers.
7. **Deterministic where possible.** Same inputs plus same config yields the same routing decision, so behaviour is testable.
8. **Storage abstraction.** All persistence goes through a `Store` interface with a SQLite implementation, so Phase 2 can swap in D1, Postgres or Durable Objects.

## 4. Architecture

```
application code
      |
      v
  llmrouter.call(task, messages, ...)
      |
      +-- Config loader / registry   (models, tasks, tiers)
      +-- Cache                      (hit -> return immediately)
      +-- Budget guard               (refuse if cap exceeded)
      +-- Router / selector          (pick candidates for the task)
      +-- Limiter                    (rpm/rps/rpd/tpm/concurrency per model)
      +-- Executor                   (call adapter, retry, fall back)
      +-- Provider adapters          (OpenAI-compatible, Anthropic, Gemini, ...)
      +-- Validator                  (JSON schema / custom checks)
      +-- Logger                     (call_log, usage, cooldowns)
      |
      v
  Store (SQLite)
```

## 5. Data model

All tables are created by a migration on first run. Use integer UTC epoch seconds or ISO-8601 UTC strings consistently (pick one and document it).

### 5.1 `models` (configuration, loaded from YAML)

| column | type | notes |
| --- | --- | --- |
| id | text PK | stable slug, e.g. `gemini-flash-free` |
| provider | text | adapter name, e.g. `openai_compat`, `anthropic`, `gemini`, `cloudflare` |
| model\_name | text | the provider's model identifier |
| base\_url | text null | for OpenAI-compatible endpoints |
| api\_key\_env | text | **name** of the env var holding the key |
| tier | text | `T0`..`T3` or any tier names defined in config |
| priority | int | lower is tried first within a tier |
| enabled | bool |  |
| rpm, rps, rpd | int null | request limits; null = unlimited/unknown |
| tpm, tpd | int null | token limits |
| max\_concurrency | int | default 4 |
| context\_window | int | max input+output tokens |
| max\_output\_tokens | int |  |
| supports\_json\_mode | bool |  |
| supports\_tools | bool |  |
| supports\_vision | bool |  |
| input\_price\_per\_mtok | real | USD, 0 for free tiers |
| output\_price\_per\_mtok | real | USD |
| shadow\_input\_price\_per\_mtok, shadow\_output\_price\_per\_mtok | real null | what it *would* cost on a paid plan, for savings reports |
| is\_free\_tier | bool |  |
| data\_policy | text | e.g. `may_train`, `no_train`, `unknown`; used to block sensitive data from free tiers |
| notes | text |  |

### 5.2 `task_routes` (configuration)

| column | type | notes |
| --- | --- | --- |
| task | text PK | e.g. `extract_claims`, `judge`, `draft_reply` |
| tier\_order | json | ordered list of tiers to try, e.g. `["T1","T2","T3"]` |
| requires | json | capability requirements, e.g. `{"json_mode": true, "min_context": 32000}` |
| default\_temperature | real |  |
| max\_attempts | int | total attempts across models |
| budget\_group | text | which budget this task is charged to |
| sensitivity | text | `public` or `private`; private tasks must not use models with `data_policy = may_train` |
| cache\_ttl\_seconds | int null | null = no expiry, 0 = never cache |

### 5.3 `usage_windows` (live counters)

Primary key: `(model_id, window_type, window_start)` where `window_type` is one of `second`, `minute`, `day`. Columns: `requests`, `input_tokens`, `output_tokens`, `cost_usd`. Old windows are pruned periodically.

**Which counters each limit uses.** Daily request limits (`rpd`) and daily token limits (`tpd`) both use the `day` window, with separate counters: `requests` is checked against `rpd`, and `input_tokens + output_tokens` is checked against `tpd`. Per-minute token limits (`tpm`) use the `minute` window the same way. Reservations (section 8) increment these counters before a call and reconcile them to actual usage afterwards.

### 5.4 `cooldowns`

`model_id` PK, `cooldown_until`, `reason`, `consecutive_failures`. Set when a model returns 429 or repeated 5xx. The selector skips models that are cooling down.

### 5.5 `call_log` (one row per attempt)

`id`, `ts`, `request_id` (shared by all attempts for one logical call), `task`, `caller` (free-text tag from the application), `model_id`, `attempt_no`, `status` (`ok`, `rate_limited`, `server_error`, `timeout`, `client_error`, `invalid_output`, `budget_blocked`, `cache_hit`), `http_status`, `latency_ms`, `input_tokens`, `output_tokens`, `cost_usd`, `shadow_cost_usd`, `error_message` (truncated), `prompt_hash`. Optionally store full prompt and response in a separate table `call_payloads`, off by default, with a retention setting.

### 5.6 `cache`

`key` PK (hash), `task`, `model_family`, `response_json`, `created_at`, `expires_at`, `hits`.

### 5.7 `budgets`

`budget_group` PK, `monthly_cap_usd`, `soft_warn_pct` (default 80), `hard_stop` (bool), `month` (YYYY-MM), `spent_usd`. Spend is computed from `call_log` or maintained incrementally; the spec requires it to be correct after a crash.

### 5.8 `budget_reservations`

Columns: `id`, `request_id`, `budget_group`, `reserved_usd`, `status` (`active`, `settled`, `released`, `expired`), `created_at`, `expires_at`. This table is what keeps a hard cap correct under concurrency and crashes.

- **Reserve before each attempt.** In one `BEGIN IMMEDIATE` transaction, compute `spent_usd + sum(active reservations) + worst-case cost` for the attempt. If that exceeds the cap and `hard_stop` is set, raise `BudgetExceeded`. Otherwise insert an `active` reservation.
- **Worst-case cost** = estimated input tokens times the input price, plus `max_output_tokens` times the output price. Free-tier models reserve zero.
- **Settle or release.** On success, mark the reservation `settled` and record the actual cost. On a failure that used no tokens, mark it `released`.
- **Crash safety.** Every reservation has an `expires_at` (call timeout plus a margin). Expired `active` reservations are swept to `expired` at start-up and before each budget check, so a crash can never lock budget permanently. After a crash, spend is recomputed from settled `call_log` rows.

## 6. Public API

```python
from llmrouter import Router

router = Router.from_config("config/router.yaml", db_path="data/router.db")

# Plain text
result = router.call(
    task="summarize",
    messages=[{"role": "user", "content": "..."}],
    caller="weekly_report",
)
result.text; result.model_id; result.cost_usd; result.cached; result.attempts

# Structured output validated against a pydantic model or JSON schema
result = router.call_json(
    task="extract_claims",
    messages=[...],
    schema=ClaimsModel,           # pydantic class or JSON schema dict
    system="...",
)
result.data                       # parsed, validated object

# Async and batch
await router.acall(...)
results = await router.batch(items, task="judge", concurrency=8, on_progress=print)

# Embeddings (separate method, same routing/limits/logging)
vectors = router.embed(task="embed", texts=[...])
```

Call options: `temperature`, `max_output_tokens`, `force_tier`, `force_model`, `use_cache` (default true), `sensitivity` override, `metadata` dict, `timeout_s`, `max_cost_usd` (per-call ceiling).

Errors are typed: `BudgetExceeded`, `NoModelAvailable`, `OutputValidationError`, `ConfigError`, `AllAttemptsFailed`. Each carries the attempt history so callers can decide what to do.

## 7. Routing algorithm

For each call:

1. **Resolve the task route.** Unknown task raises `ConfigError`.
2. **Check the cache.** On hit, log `cache_hit`, return it. No budget or limit is consumed.
3. **Check the budget.** If the group's hard cap is exceeded, raise `BudgetExceeded`. If past the soft threshold, emit a warning log.
4. **Build the candidate list.** For each tier in `tier_order`, take models where `enabled`, capabilities satisfy `requires`, the estimated input fits `context_window`, sensitivity rules pass, and the model is not cooling down. Sort by `priority`, then by remaining headroom (most unused quota first).
5. **Reserve capacity.** For the first candidate, atomically check and reserve against rpm/rps/rpd/tpm/tpd and concurrency. If it has no headroom, move to the next candidate. If *no* candidate has headroom but some will within `max_wait_s` (default 20), sleep until the earliest slot, then retry selection. Otherwise raise `NoModelAvailable`.
6. **Execute** via the adapter (section 9). On success, validate (section 10).
7. **On failure,** classify the error (section 9), update cooldowns and logs, release or adjust the reservation, and continue to the next candidate until `max_attempts` is reached.
8. **On success,** write the cache entry, update actual token usage and cost, update the budget, log, and return.

The budget check in step 3 is a reservation (section 5.8), not a plain read. It is taken before each attempt and settled or released afterwards, so parallel calls cannot jointly exceed a hard cap.

Escalation by quality: if output validation fails twice on one tier, the next attempt moves to the next tier even if the earlier tier has headroom.

## 8. Rate limiting

- Support `rps`, `rpm`, `rpd`, `tpm`, `tpd` and `max_concurrency` per model. Null means not enforced.
- Use **fixed windows aligned to the provider's window** where known, otherwise a sliding window. Make the strategy a per-model option.
- **Token estimation before the call:** use a tokenizer if available, otherwise `len(text)/4`. Reserve `estimated_input + max_output_tokens`, then **reconcile** with the actual usage returned by the provider.
- Reservation must be **atomic** (single SQLite transaction with `BEGIN IMMEDIATE`) so concurrent threads or processes cannot over-commit.
- Honor a provider's `Retry-After` and rate-limit headers when present, and set the cooldown accordingly.
- Safety margin: enforce limits at a configurable fraction (default 90%) of the stated value.
- Provide `router.status()` that returns, for each model, current usage versus limits and any active cooldown.

## 9. Execution, retries and fallback

**Error classification**

| condition | class | action |
| --- | --- | --- |
| 429 | `rate_limited` | set cooldown (use `Retry-After` or exponential), try next model |
| 5xx, connection reset | `server_error` | retry same model once with backoff, then next model; after N consecutive failures, cooldown |
| timeout | `timeout` | next model; count toward cooldown |
| 400 (bad request), context too long | `client_error` | do not retry the same request on the same model; if context-length, try a model with a larger window; otherwise raise |
| 401, 403 | `auth_error` | disable the model for the session, log loudly, try next |
| content filtered / safety block | `blocked` | try next model once, else raise |
| output fails validation | `invalid_output` | see section 10 |

**Backoff:** exponential with full jitter, base 1s, factor 2, cap 60s, configurable. A hard per-call `timeout_s` (default 60) applies to every attempt, and an overall deadline applies to the whole logical call.

**Idempotency:** retries of the same logical call share a `request_id`. Never double-charge budget for a failed attempt that returned no tokens.

## 10. Output validation

- `call_json` asks the model for JSON (using native JSON mode or schema-constrained output when `supports_json_mode` is true, otherwise via prompt instructions), strips code fences, parses, and validates against the supplied pydantic model or JSON schema.
- On failure: first retry the **same model** once with the validation error appended as a correction message. If it fails again, move to the **next model/tier**.
- Support **custom validators**: a callable `(parsed) -> list[str]` of problems, for rules the schema cannot express.
- Log every validation failure with the reason. Aggregate this per model, since it is a key signal of which models are good enough for a task.

## 11. Caching

- Key: hash of `(task, normalized messages, system prompt, schema hash, temperature, max_output_tokens, model_family)`. Include the **prompt template version** if the caller supplies one.
- Only cache successful, validated responses. Respect `cache_ttl_seconds` from the task route and a per-call override.
- Cache hits cost nothing and are logged.
- Provide `router.cache_clear(task=None)` and a `--no-cache` CLI flag.
- Temperature greater than 0 should still be cacheable (the owner usually wants repeatability for batch jobs), but make this a per-task setting.

## 12. Budget guard

- Cost per call = `input_tokens * input_price + output_tokens * output_price` (per million tokens). Free-tier models record `cost_usd = 0` and also `shadow_cost_usd` using the shadow prices.
- Budgets are per `budget_group` per calendar month, with a soft warning threshold and an optional hard stop.
- A per-call `max_cost_usd` ceiling uses the *estimated* cost and refuses to start a call that could exceed it.
- `router.report(month=None)` returns spend by group, task, and model, plus **savings from free tiers** (shadow cost minus actual cost).

## 13. Provider adapters

Define one abstract interface:

```python
class ProviderAdapter(Protocol):
    name: str
    def complete(self, req: NormalizedRequest, model: ModelConfig) -> NormalizedResponse: ...
    async def acomplete(self, req, model) -> NormalizedResponse: ...
    def embed(self, texts, model) -> EmbedResponse: ...
    def classify_error(self, exc_or_response) -> ErrorClass: ...
```

`NormalizedResponse` includes `text`, `input_tokens`, `output_tokens`, `finish_reason`, `raw_headers`, and `raw` (kept only for debug logging).

Ship adapters for:

1. **`openai_compat`**: any OpenAI-style `/chat/completions` endpoint with a configurable `base_url`. This one adapter covers OpenAI, many free-tier hosts, OpenRouter, NVIDIA NIM, DeepSeek and others. Build it first.
2. **`anthropic`**: the Messages API, including system prompts and prompt-caching hints.
3. **`gemini`**: Google's generative API.
4. **`cloudflare`**: Workers AI via its REST API or OpenAI-compatible endpoint.

Use plain `httpx` rather than heavy provider SDKs where possible, so behaviour (timeouts, headers, retries) stays under the router's control. Adding a new provider must require only a new adapter file plus config.

## 14. Configuration

`config/router.yaml` contains `models`, `tiers`, `task_routes`, `budgets` and global settings (`default_timeout_s`, `limit_safety_fraction`, `log_payloads`, `cache_enabled`). On start, validate the file (pydantic), then upsert into SQLite. Unknown fields are errors. Provide:

- `config/router.example.yaml` with **placeholder values and comments**, and a clear note that limits, prices and model names must be checked against each provider's current documentation before use.
- `.env.example` listing the environment variable names. Load with `python-dotenv` only in development.

## 15. Concurrency

- Provide both sync and async APIs. The async path uses `asyncio` with a semaphore per model (`max_concurrency`).
- `router.batch()` processes a list of items with bounded concurrency, preserves input order in the output, never aborts the whole batch because one item failed (returns per-item success or error), supports resumability (skip items whose cache key is already complete), and reports progress.
- Optional, behind a flag: use provider **batch APIs** (typically discounted) for large, non-urgent jobs. Design the interface for it but implementing it can wait until after the core is solid.

## 16. Observability and CLI

A `llmrouter` CLI (use `typer`) with commands:

- `status`: live usage versus limits, cooldowns.
- `report [--month]`: spend, savings, calls, success rate, mean latency, validation-failure rate, grouped by model/task.
- `test-model <model_id>`: send a tiny prompt to verify the key and endpoint work.
- `models`: list configured models and whether they are enabled and healthy.
- `cache clear|stats`.
- `replay <request_id>`: show the attempt history for one logical call.

Use structured logging (JSON lines) to a file in addition to the database. Never log API keys or full prompts unless `log_payloads` is enabled.

## 17. Security and privacy

- Keys only from environment variables or a secrets manager. Redact them from all logs and exceptions.
- Enforce the `sensitivity` / `data_policy` rule: a `private` task can never be routed to a model whose policy is `may_train` or `unknown`. Fail closed.
- Truncate stored error messages and strip anything resembling a key.
- Document that if the router is exposed over HTTP in Phase 2, it must require authentication.

## 18. Testing

- A **fake provider adapter** that can be scripted to return success, 429 with `Retry-After`, 500, timeouts, malformed JSON and slow responses, so every branch of the retry and fallback logic is covered without network access.
- Unit tests for: window arithmetic, atomic reservation under concurrency (many threads, assert limits never exceeded), cooldown expiry, budget cutoffs, cache keys, config validation, error classification per adapter, sensitivity blocking.
- Property-style test: with N models and random failure injection, a call either returns a valid result or raises a typed error, never hangs, never exceeds limits.
- A small integration test per real adapter, skipped unless the relevant key is present.
- Target: all core logic covered, CI via `pytest`.

## 19. Project layout

```
llmrouter/
  pyproject.toml
  README.md
  config/router.example.yaml
  .env.example
  src/llmrouter/
    __init__.py
    router.py        # Router class, public API
    config.py        # pydantic models, YAML loading
    store/           # Store protocol, sqlite.py, migrations
    selector.py      # candidate selection
    limiter.py       # reservations and windows
    executor.py      # retry, fallback, error handling
    validation.py
    cache.py
    budget.py
    adapters/        # base.py, openai_compat.py, anthropic.py, gemini.py, cloudflare.py, fake.py
    cli.py
    errors.py
  tests/
  examples/          # runnable examples for each public method
```

Tech choices: Python 3.11+, `httpx`, `pydantic` v2, `typer`, `pyyaml`, `pytest`, `pytest-asyncio`. SQLite in WAL mode. Keep dependencies minimal.

## 20. Phase 2 readiness (design constraints only)

Do not build these now, but do not make them hard:

- All persistence behind the `Store` interface, with no SQL outside `store/`.
- Core logic has no global state and no dependency on the filesystem other than through the config loader and the store.
- The `Router` can be wrapped by a thin HTTP layer (for example FastAPI) exposing `POST /v1/call` and `GET /v1/status` with token auth.
- Rate-limit counters will need a single-writer or strongly consistent store when running as a multi-instance service, so keep counter logic isolated in `limiter.py`.

## 21. Acceptance criteria

Phase 1 is done when:

1. A script can call `router.call_json(task=...)` and receive a validated object, with no provider SDK imported by the script.
2. Changing which model serves a task requires editing only YAML.
3. With a model scripted to return 429, calls transparently fall back to the next model and the failing model enters cooldown, visible in `status`.
4. Under 50 concurrent calls across several threads, no model's configured limits are exceeded.
5. Setting a small monthly cap causes `BudgetExceeded` once reached, and the report shows correct spend after a restart.
6. Re-running an identical batch costs zero (all cache hits) and completes quickly.
7. A `private` task is never sent to a `may_train` or `unknown` model.
8. `report` shows per-model success rate, latency, validation failures, cost and free-tier savings.
9. Tests pass, the README explains setup in under ten minutes, and each module has comments explaining *why*, not just *what*.

Also required: with a small hard cap and 50 concurrent calls, total settled spend never exceeds the cap, and a simulated crash mid-call leaves no permanently locked budget.

## 22. Build order

1. Project skeleton, config models, SQLite store and migrations.
2. Fake adapter plus the executor (retry, backoff, error classes). Tests first.
3. Limiter with atomic reservation, then selector and cooldowns.
4. `openai_compat` adapter and the first real end-to-end call.
5. Validation and `call_json`, with correction retry and tier escalation.
6. Cache, then budget guard.
7. Logging, `status` and `report` CLI.
8. Anthropic, Gemini and Cloudflare adapters.
9. Async API and `batch()`.
10. Hardening: concurrency tests, failure injection, documentation, examples.

## 23. Instructions to the building assistant

- Work milestone by milestone. After each, show what was built, how to run its tests, and what comes next. Do not generate the whole project in one pass.
- Ask before making any decision that changes the data model or public API. Otherwise proceed with sensible defaults and state them.
- **Do not invent provider limits, prices or model names.** Use clearly marked placeholders and tell me what to verify in each provider's documentation.
- Prefer simple, readable code over clever abstractions. Comment the reasoning behind non-obvious choices, because I want to learn from the code.
- Flag any place where the spec is ambiguous or where you think a different design would be better, and explain the trade-off briefly.
