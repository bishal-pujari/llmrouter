CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY, provider TEXT NOT NULL, model_name TEXT NOT NULL,
    base_url TEXT, api_key_env TEXT NOT NULL, tier TEXT NOT NULL, priority INTEGER NOT NULL,
    enabled INTEGER NOT NULL, rpm INTEGER, rps INTEGER, rpd INTEGER, tpm INTEGER, tpd INTEGER,
    max_concurrency INTEGER NOT NULL, context_window INTEGER NOT NULL, max_output_tokens INTEGER NOT NULL,
    supports_json_mode INTEGER NOT NULL, supports_tools INTEGER NOT NULL, supports_vision INTEGER NOT NULL,
    input_price_per_mtok REAL NOT NULL, output_price_per_mtok REAL NOT NULL,
    shadow_input_price_per_mtok REAL, shadow_output_price_per_mtok REAL,
    is_free_tier INTEGER NOT NULL, data_policy TEXT NOT NULL, notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS task_routes (
    task TEXT PRIMARY KEY, tier_order TEXT NOT NULL, requires TEXT NOT NULL,
    default_temperature REAL NOT NULL, max_attempts INTEGER NOT NULL,
    budget_group TEXT NOT NULL, sensitivity TEXT NOT NULL, cache_ttl_seconds INTEGER
);

CREATE TABLE IF NOT EXISTS usage_windows (
    model_id TEXT NOT NULL, window_type TEXT NOT NULL, window_start INTEGER NOT NULL,
    requests INTEGER NOT NULL DEFAULT 0, input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (model_id, window_type, window_start)
);

CREATE TABLE IF NOT EXISTS cooldowns (
    model_id TEXT PRIMARY KEY, cooldown_until INTEGER NOT NULL, reason TEXT NOT NULL,
    consecutive_failures INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS call_log (
    id TEXT PRIMARY KEY, ts INTEGER NOT NULL, request_id TEXT NOT NULL, task TEXT NOT NULL,
    caller TEXT, model_id TEXT, attempt_no INTEGER NOT NULL, status TEXT NOT NULL,
    http_status INTEGER, latency_ms INTEGER, input_tokens INTEGER, output_tokens INTEGER,
    cost_usd REAL NOT NULL DEFAULT 0, shadow_cost_usd REAL NOT NULL DEFAULT 0,
    error_message TEXT, prompt_hash TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS call_log_request_idx ON call_log(request_id, attempt_no);
CREATE INDEX IF NOT EXISTS call_log_ts_idx ON call_log(ts);

CREATE TABLE IF NOT EXISTS call_payloads (
    call_id TEXT PRIMARY KEY, prompt_json TEXT NOT NULL, response_json TEXT,
    created_at INTEGER NOT NULL, expires_at INTEGER
);

CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY, task TEXT NOT NULL, model_family TEXT NOT NULL,
    response_json TEXT NOT NULL, created_at INTEGER NOT NULL, expires_at INTEGER,
    hits INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS budgets (
    budget_group TEXT PRIMARY KEY, monthly_cap_usd REAL NOT NULL,
    soft_warn_pct REAL NOT NULL DEFAULT 80, hard_stop INTEGER NOT NULL,
    month TEXT NOT NULL, spent_usd REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS budget_reservations (
    id TEXT PRIMARY KEY, request_id TEXT NOT NULL, budget_group TEXT NOT NULL,
    reserved_usd REAL NOT NULL, status TEXT NOT NULL CHECK(status IN ('active','settled','released','expired')),
    created_at INTEGER NOT NULL, expires_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS budget_reservations_active_idx
    ON budget_reservations(budget_group, status, expires_at);

CREATE TABLE IF NOT EXISTS global_settings (
    key TEXT PRIMARY KEY, value_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tiers (
    name TEXT PRIMARY KEY, ordinal INTEGER NOT NULL
);
