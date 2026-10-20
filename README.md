# llmrouter

A local-first Python library for routing LLM calls across configurable providers, with budgets, limits, caching, and call history.

The project is being built in milestones. It provides strict YAML configuration, SQLite storage, rate limiting and model selection, provider adapters, synchronous and asynchronous call APIs, structured output, caching, budgets, and a CLI.

## Local setup

Use Python 3.11 or newer and install the project in editable mode:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[dev]'
cp config/router.example.yaml config/router.yaml
cp .env.example .env
```

Review every model identifier, price, and quota in `config/router.yaml` against the provider's current documentation before use. The example intentionally contains placeholders, not provider claims. API keys belong in environment variables named by `api_key_env`; they do not belong in YAML or SQLite.

The SQLite database is created on first store access. Timestamps in the database use integer UTC epoch seconds. SQLite uses WAL mode to support concurrent local processes.

The SQLite database and JSONL log are local sensitive artifacts and are created
owner-readable only. Protect the containing directory and use
`cache_ttl_seconds: 0` for tasks whose responses must not be retained locally.
Provider `base_url` values must use HTTPS; HTTP is accepted only for loopback
test servers, and URL credentials/query strings are rejected. Built-in HTTP
clients do not use ambient proxy environment variables when sending API keys.

## Test this stage

Run the offline test suite. It uses a mock HTTP transport, so it does not contact a provider or use API credits:

```sh
python -m pytest -q
```

The suite includes quota and hard-budget concurrency checks, expired-reservation recovery after reopening the database, deterministic failure injection, and batch cache-resume coverage.

For a live smoke call, edit `config/router.yaml`: replace the placeholder model ID and base URL, verify its limits/prices/data policy, and set `enabled: true`. Put that provider's key in `.env` under the variable named by `api_key_env`. Then run:

```sh
python examples/test_client.py --task example_task --message "Reply with the word OK."
```

The live smoke call sends a real request and may use provider quota or incur a charge. The library itself does not load `.env`; only this development client does.

## Provider adapters

Provider names `openai_compat`, `anthropic`, `gemini`, and `cloudflare` are registered by default. `anthropic` uses the Messages API; `gemini` uses `generateContent`; and `cloudflare` uses the Workers AI account REST endpoint. For Cloudflare's OpenAI-compatible endpoint, configure `provider: openai_compat` and use its account `/ai/v1` base URL.

For Anthropic and Gemini, `base_url` may be omitted to use their standard API roots. Set `api_key_env` to the environment-variable name holding the key. Gemini model names are passed as configured, with an optional leading `models/`. For the direct Cloudflare adapter, set `base_url` to `https://api.cloudflare.com/client/v4/accounts/<account_id>` and use the Workers AI model name such as `@cf/meta/...`; `api_key_env` should name a Cloudflare API token with Workers AI permission.

Each adapter maps normalized text messages, temperature, output-token limits, usage metadata, and provider errors. Provider-specific multimodal content and tool calling remain outside these adapters.

Each provider attempt is written to `call_log` in SQLite and to `router.jsonl` beside the database (by default, `data/router.jsonl`). Logs include task/model, attempt status, latency, usage, cost, and a prompt hash; prompt contents and API keys are omitted.

When the library is exposed over HTTP in a later phase, the HTTP layer must require token authentication.

Private routes fail closed unless the selected model is explicitly configured
with `data_policy: no_train`. The sample live configuration intentionally uses
`unknown` policies, so it is suitable only for public data until each provider's
current terms have been verified. Do not treat the local SQLite cache as
encrypted storage; filesystem permissions and operational retention remain part
of the deployment responsibility.

## Phase 1 boundaries

Quota and budget reservations are persisted atomically in SQLite. Per-model concurrency slots coordinate Router instances in the same Python process. `settings.max_wait_s` is currently reserved for a later wait-and-retry behavior; when all eligible capacity is used, the current call raises `NoModelAvailable`. The async Router methods run the common sync execution pipeline in a per-Router worker pool, so provider requests do not block the caller's event loop. Embeddings, provider-specific batch endpoints, and hosted HTTP service support are outside Phase 1.

## CLI

Install the editable package to add the `llmrouter` command, or run it as a module. Global `--config` and `--db` options go before the command:

```sh
python -m llmrouter --config config/router.yaml --db data/router.db status
python -m llmrouter --config config/router.yaml report --month 2026-10
python -m llmrouter --config config/router.yaml models
python -m llmrouter --config config/router.yaml call example_task --message "Summarize this text" --no-cache
python -m llmrouter --config config/router.yaml cache stats
python -m llmrouter --config config/router.yaml cache clear --task example_task
python -m llmrouter --config config/router.yaml replay REQUEST_ID
python -m llmrouter --config config/router.yaml test-model MODEL_ID --task example_task
```

`test-model` sends a real small request and can consume provider quota or budget. It loads `.env` for this command only, tests one enabled model, bypasses cache, and still observes the selected task's budget and that model's rate limits. `models` reports health as `unknown` until a provider attempt has been recorded; it does not make background health-check requests.

`call` and `test-model` output `cost_usd` for compatibility and `estimated_cost_usd` as its explicit label. Both are router-calculated from provider-reported token usage and the configured per-million-token prices; they are not provider-reported invoice amounts. `comparison_cost_usd` and `savings_usd` use the model's optional `shadow_input_price_per_mtok` and `shadow_output_price_per_mtok` rates, and are `null` unless both are configured. Cache hits report zero estimated cost and no comparison or savings estimate.

## Async and batch calls

`Router.acall()` and `Router.acall_json()` provide asynchronous APIs. The router runs its synchronous routing and provider pipeline in worker threads, keeping the calling event loop responsive while retaining the same retries, quota, cache, and budget reservation behavior. Provider adapters also expose `acomplete()` for direct async use.

`Router.batch()` runs independent message lists with a concurrency limit, returns one `BatchResult` per input in the original order, and keeps item failures in that result instead of stopping the rest. An optional callback receives `(completed, total, item_result)`. Successful calls use the normal response cache, so rerunning a batch with caching enabled reuses completed items.

```python
import asyncio
from llmrouter import Router

async def main():
    router = Router.from_config("config/router.yaml")
    try:
        results = await router.batch(
            [
                [{"role": "user", "content": "Summarize item one."}],
                [{"role": "user", "content": "Summarize item two."}],
            ],
            task="example_task",
            concurrency=4,
            on_progress=lambda done, total, item: print(f"{done}/{total}"),
        )
        for item in results:
            if item.ok:
                print(item.result.text)
            else:
                print(f"item {item.index} failed: {item.error}")
    finally:
        router.close()

asyncio.run(main())
```

Each batch input can also be a mapping with a `messages` key and optional per-call `options`, for example `{"messages": [...], "options": {"force_model": "cheap-model"}}`. Batch progress callbacks are synchronous from the event loop; async callbacks are also accepted. Callback exceptions are ignored so a reporting failure does not discard completed work.

Run the live batch example after configuring an enabled model and provider key:

```sh
python examples/async_batch.py --task example_task --message "Summarize item one" --message "Summarize item two" --concurrency 2
```

This command sends real provider requests and may consume quota or incur charges. Re-running identical items uses the response cache unless caching is disabled or the route TTL expires.

## Structured output

`Router.call_json()` accepts either a Pydantic model class or a JSON Schema dictionary. It returns the validated value in `result.data` (a Pydantic instance for a model class, or a Python value for a JSON Schema). Models marked `supports_json_mode: true` receive OpenAI-compatible JSON mode; other models receive the schema in the system instructions.

If the first response is invalid, the router sends one correction request to the same model with the validation issues. After a second invalid response, it escalates to the next tier. An `OutputValidationError` includes the issues and attempt history if all eligible candidates fail validation. An optional `validator` callable can apply extra rules and return a list of problem strings.

```python
from pydantic import BaseModel
from llmrouter import Router

class Claim(BaseModel):
    text: str
    confidence: float

router = Router.from_config("config/router.yaml")
result = router.call_json(
    task="extract_claims",
    messages=[{"role": "user", "content": "Extract the main claim."}],
    schema=Claim,
)
print(result.data.text, result.data.confidence)
```

## Cache and budgets

Successful text calls and validated `call_json()` responses are cached by task, normalized messages, schema, temperature, output limit, sensitivity, and configured model family. `cache_ttl_seconds: null` keeps an entry until cleared; `0` disables caching for that task. A call can override this with `cache_ttl_seconds=...` or bypass lookup and writes with `use_cache=False`. Custom validators bypass caching because their callable logic cannot be represented safely in a stable key.

```python
router.cache_clear(task="extract_claims")  # Omit task to clear the whole cache.
report = router.report()                    # Current UTC month; pass "YYYY-MM" for another month.

result = router.call(
    "summarize", messages, max_cost_usd=0.01, use_cache=False,
)
```

Each provider attempt reserves its estimated worst-case cost in SQLite before the request. A hard monthly budget blocks attempts that would exceed the cap; successful and billable validation-failure attempts settle the reservation against provider usage, while failures without usage release it. Reservations expire after the request timeout plus a one-minute margin. `router.report()` returns spend by budget group, task, and model, plus savings against configured shadow prices.
