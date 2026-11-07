"""Cloudflare D1 implementation of the async Store boundary.

The ``db`` object is the D1 binding exposed as ``self.env.DB`` by Python
Workers. Query values are always bound parameters; ``exec`` is used only for
the static, packaged schema migration.
"""

from __future__ import annotations

import calendar
import json
import time
from importlib.resources import files
from typing import Any

from llmrouter.config import RouterConfig
from llmrouter.errors import BudgetExceeded
from llmrouter.store.protocol import CallLogEntry


def _plain(value: Any) -> Any:
    """Convert a Pyodide JavaScript proxy into ordinary Python containers."""
    converter = getattr(value, "to_py", None)
    if callable(converter):
        value = converter()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain(item) for item in value]
    return value


def _rows(result: Any) -> list[dict[str, Any]]:
    result = _plain(result) or {}
    raw = result.get("results", []) if isinstance(result, dict) else []
    return [dict(row) for row in raw]


def _changes(result: Any) -> int:
    result = _plain(result) or {}
    meta = result.get("meta", {}) if isinstance(result, dict) else {}
    return int(meta.get("changes", 0) or 0)


class D1Store:
    """Async D1 store with the same logical schema as the local SQLite store."""

    db_path = "d1"

    def __init__(self, database: Any) -> None:
        self.database = database

    async def migrate(self) -> None:
        migration = files("llmrouter.store").joinpath("migrations/001_initial.sql")
        await self.database.exec(migration.read_text(encoding="utf-8"))
        await self.database.prepare(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)"
        ).bind(1, int(time.time())).run()

    async def _run(self, sql: str, *params: Any) -> Any:
        return _plain(await self.database.prepare(sql).bind(*params).run())

    async def _first(self, sql: str, *params: Any) -> dict[str, Any] | None:
        value = await self.database.prepare(sql).bind(*params).first()
        value = _plain(value)
        return None if value is None else dict(value)

    async def upsert_config(self, config: RouterConfig) -> None:
        payload = config.model_dump(mode="json")
        statements: list[Any] = [
            self.database.prepare("DELETE FROM models"),
            self.database.prepare("DELETE FROM task_routes"),
            self.database.prepare("DELETE FROM budgets"),
            self.database.prepare("DELETE FROM tiers"),
            self.database.prepare("DELETE FROM global_settings"),
        ]
        for model in payload["models"]:
            columns = tuple(model)
            values = tuple(_sqlite_value(model[name]) for name in columns)
            updates = ", ".join(f"{name}=excluded.{name}" for name in columns if name != "id")
            statements.append(self.database.prepare(
                f"INSERT INTO models ({', '.join(columns)}) VALUES ({','.join('?' for _ in columns)}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}"
            ).bind(*values))
        for task, route in payload["task_routes"].items():
            statements.append(self.database.prepare(
                "INSERT INTO task_routes VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            ).bind(
                task, json.dumps(route["tier_order"]), json.dumps(route["requires"]),
                route["default_temperature"], route["max_attempts"], route["budget_group"],
                route["sensitivity"], route["cache_ttl_seconds"],
            ))
        for group, budget in payload["budgets"].items():
            statements.append(self.database.prepare(
                "INSERT INTO budgets(budget_group, monthly_cap_usd, soft_warn_pct, hard_stop, month, spent_usd) "
                "VALUES (?, ?, ?, ?, ?, 0) ON CONFLICT(budget_group) DO UPDATE SET "
                "monthly_cap_usd=excluded.monthly_cap_usd, soft_warn_pct=excluded.soft_warn_pct, "
                "hard_stop=excluded.hard_stop"
            ).bind(
                group, budget["monthly_cap_usd"], budget["soft_warn_pct"],
                int(budget["hard_stop"]), time.strftime("%Y-%m", time.gmtime()),
            ))
        statements.extend(
            self.database.prepare("INSERT INTO tiers(name, ordinal) VALUES (?, ?)").bind(name, index)
            for index, name in enumerate(payload["tiers"])
        )
        statements.extend(
            self.database.prepare("INSERT INTO global_settings(key, value_json) VALUES (?, ?)").bind(
                key, json.dumps(value)
            ) for key, value in payload["settings"].items()
        )
        await self.database.batch(statements)

    async def reserve_model_usage(
        self, model_id: str, starts: dict[str, int], request_limits: dict[str, int | None],
        token_limits: dict[str, int | None], input_tokens: int, output_tokens: int,
    ) -> bool:
        # Keep the quota check and all three counter upserts in one D1 write.
        # A read-then-write sequence would allow two Worker isolates to both
        # observe the same remaining capacity.
        windows = ("second", "minute", "day")
        current_columns = []
        current_params: list[Any] = []
        for window in windows:
            current_columns.extend([
                f"COALESCE((SELECT requests FROM usage_windows WHERE model_id=? AND window_type='{window}' AND window_start=?), 0) AS {window}_requests",
                f"COALESCE((SELECT input_tokens FROM usage_windows WHERE model_id=? AND window_type='{window}' AND window_start=?), 0) AS {window}_input",
                f"COALESCE((SELECT output_tokens FROM usage_windows WHERE model_id=? AND window_type='{window}' AND window_start=?), 0) AS {window}_output",
            ])
            current_params.extend([model_id, starts[window], model_id, starts[window], model_id, starts[window]])
        request_conditions = [
            "(? IS NULL OR second_requests + 1 <= ?)",
            "(? IS NULL OR minute_requests + 1 <= ?)",
            "(? IS NULL OR day_requests + 1 <= ?)",
            "(? IS NULL OR minute_input + minute_output + ? + ? <= ?)",
            "(? IS NULL OR day_input + day_output + ? + ? <= ?)",
        ]
        limit_params = [
            request_limits.get("second"), request_limits.get("second"),
            request_limits.get("minute"), request_limits.get("minute"),
            request_limits.get("day"), request_limits.get("day"),
            token_limits.get("minute"), input_tokens, output_tokens, token_limits.get("minute"),
            token_limits.get("day"), input_tokens, output_tokens, token_limits.get("day"),
        ]
        sql = (
            "WITH current AS (SELECT " + ", ".join(current_columns) + "), "
            "windows(window_type, window_start) AS (VALUES (?, ?), (?, ?), (?, ?)) "
            "INSERT INTO usage_windows(model_id, window_type, window_start, requests, input_tokens, output_tokens, cost_usd) "
            "SELECT ?, windows.window_type, windows.window_start, 1, ?, ?, 0 FROM windows, current WHERE "
            + " AND ".join(request_conditions)
            + " ON CONFLICT(model_id, window_type, window_start) DO UPDATE SET requests=requests+1, "
            "input_tokens=input_tokens+excluded.input_tokens, output_tokens=output_tokens+excluded.output_tokens"
        )
        params = current_params + [value for window in windows for value in (window, starts[window])] + [model_id, input_tokens, output_tokens] + limit_params
        result = await self._run(sql, *params)
        return _changes(result) == len(windows)

    async def reconcile_model_usage(
        self, model_id: str, starts: dict[str, int], estimated_input: int, estimated_output: int,
        actual_input: int, actual_output: int, cost_usd: float,
    ) -> None:
        statements = [self.database.prepare(
            "UPDATE usage_windows SET input_tokens=input_tokens-?+?, output_tokens=output_tokens-?+?, "
            "cost_usd=cost_usd+? WHERE model_id=? AND window_type=? AND window_start=?"
        ).bind(estimated_input, actual_input, estimated_output, actual_output, cost_usd,
               model_id, window, start) for window, start in starts.items()]
        await self.database.batch(statements)

    async def read_usage_window(self, model_id: str, window: str, start: int) -> tuple[int, int, int] | None:
        row = await self._first(
            "SELECT requests, input_tokens, output_tokens FROM usage_windows "
            "WHERE model_id=? AND window_type=? AND window_start=?", model_id, window, start,
        )
        return None if row is None else (int(row["requests"]), int(row["input_tokens"]), int(row["output_tokens"]))

    async def read_cooldown(self, model_id: str, now: int) -> tuple[int, str, int] | None:
        row = await self._first(
            "SELECT cooldown_until, reason, consecutive_failures FROM cooldowns WHERE model_id=?", model_id
        )
        if row is None:
            return None
        if int(row["cooldown_until"]) <= now:
            await self._run("DELETE FROM cooldowns WHERE model_id=?", model_id)
            return None
        return int(row["cooldown_until"]), str(row["reason"]), int(row["consecutive_failures"])

    async def write_cooldown(self, model_id: str, cooldown_until: int, reason: str, consecutive_failures: int) -> None:
        await self._run(
            "INSERT INTO cooldowns(model_id, cooldown_until, reason, consecutive_failures) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(model_id) DO UPDATE SET cooldown_until=excluded.cooldown_until, "
            "reason=excluded.reason, consecutive_failures=excluded.consecutive_failures",
            model_id, cooldown_until, reason[:200], consecutive_failures,
        )

    async def delete_cooldown(self, model_id: str) -> None:
        await self._run("DELETE FROM cooldowns WHERE model_id=?", model_id)

    async def log_call(self, entry: CallLogEntry) -> None:
        await self._insert_call(entry)

    async def _insert_call(self, entry: CallLogEntry) -> None:
        await self._run(
            "INSERT INTO call_log(id, ts, request_id, task, caller, model_id, attempt_no, status, http_status, "
            "latency_ms, input_tokens, output_tokens, cost_usd, shadow_cost_usd, error_message, prompt_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            entry.id, entry.ts, entry.request_id, entry.task, entry.caller, entry.model_id, entry.attempt_no,
            entry.status, entry.http_status, entry.latency_ms, entry.input_tokens, entry.output_tokens,
            entry.cost_usd, entry.shadow_cost_usd, entry.error_message, entry.prompt_hash,
        )

    async def reserve_budget(self, reservation_id: str, request_id: str, budget_group: str,
                             worst_case_usd: float, expires_at: int) -> tuple[float, float, float]:
        now = int(time.time())
        month = time.strftime("%Y-%m", time.gmtime(now))
        month_start = calendar.timegm(time.strptime(month + "-01", "%Y-%m-%d"))
        await self._run("UPDATE budget_reservations SET status='expired' WHERE status='active' AND expires_at<=?", now)
        budget = await self._first(
            "SELECT monthly_cap_usd, soft_warn_pct, hard_stop FROM budgets WHERE budget_group=?", budget_group
        )
        if budget is None:
            raise ValueError(f"Unknown budget group {budget_group!r}")
        cap = float(budget["monthly_cap_usd"])
        inserted = await self._run(
            "INSERT INTO budget_reservations(id, request_id, budget_group, reserved_usd, status, created_at, expires_at) "
            "SELECT ?, ?, ?, ?, 'active', ?, ? FROM budgets b WHERE b.budget_group=? AND "
            "(NOT b.hard_stop OR "
            "(SELECT COALESCE(SUM(cost_usd), 0) FROM call_log WHERE ts>=? AND "
            "task IN (SELECT task FROM task_routes WHERE budget_group=?)) + "
            "(SELECT COALESCE(SUM(reserved_usd), 0) FROM budget_reservations WHERE budget_group=? "
            "AND status='active' AND created_at>=?) + ? <= b.monthly_cap_usd)",
            reservation_id, request_id, budget_group, worst_case_usd, now, expires_at, budget_group,
            month_start, budget_group, budget_group, month_start, worst_case_usd,
        )
        if _changes(inserted) != 1:
            raise BudgetExceeded(
                f"Budget group {budget_group!r} has insufficient available budget for this attempt",
                budget_group=budget_group,
            )
        spent_row = await self._first(
            "SELECT COALESCE(SUM(cost_usd), 0) AS spent FROM call_log WHERE ts>=? "
            "AND task IN (SELECT task FROM task_routes WHERE budget_group=?)", month_start, budget_group,
        )
        spent = float((spent_row or {}).get("spent", 0) or 0)
        return spent, cap, float(budget["soft_warn_pct"])

    async def release_budget(self, reservation_id: str, *, expired: bool = False) -> None:
        await self._run(
            "UPDATE budget_reservations SET status=? WHERE id=? AND status='active'",
            "expired" if expired else "released", reservation_id,
        )

    async def settle_budget_and_log(self, reservation_id: str, entry: CallLogEntry) -> None:
        reservation = await self._first(
            "SELECT budget_group, status FROM budget_reservations WHERE id=?", reservation_id
        )
        if reservation is None or reservation["status"] != "active":
            raise RuntimeError("Budget reservation is missing or no longer active")
        await self.database.batch([
            self.database.prepare("UPDATE budget_reservations SET status='settled' WHERE id=?").bind(reservation_id),
            self.database.prepare("UPDATE budgets SET spent_usd=spent_usd+? WHERE budget_group=?").bind(
                entry.cost_usd, reservation["budget_group"]
            ),
            self.database.prepare(
                "INSERT INTO call_log(id, ts, request_id, task, caller, model_id, attempt_no, status, http_status, "
                "latency_ms, input_tokens, output_tokens, cost_usd, shadow_cost_usd, error_message, prompt_hash) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
            ).bind(
                entry.id, entry.ts, entry.request_id, entry.task, entry.caller, entry.model_id, entry.attempt_no,
                entry.status, entry.http_status, entry.latency_ms, entry.input_tokens, entry.output_tokens,
                entry.cost_usd, entry.shadow_cost_usd, entry.error_message, entry.prompt_hash,
            ),
        ])

    async def cache_get(self, key: str, *, now: int | None = None) -> dict[str, object] | None:
        timestamp = int(time.time()) if now is None else now
        row = await self._first("SELECT response_json, expires_at FROM cache WHERE key=?", key)
        if row is None:
            return None
        if row["expires_at"] is not None and int(row["expires_at"]) <= timestamp:
            await self._run("DELETE FROM cache WHERE key=?", key)
            return None
        await self._run("UPDATE cache SET hits=hits+1 WHERE key=?", key)
        return json.loads(str(row["response_json"]))

    async def cache_put(self, key: str, task: str, model_family: str,
                        response: dict[str, object], ttl_seconds: int | None) -> None:
        now = int(time.time())
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        await self._run(
            "INSERT INTO cache(key, task, model_family, response_json, created_at, expires_at, hits) "
            "VALUES (?, ?, ?, ?, ?, ?, 0) ON CONFLICT(key) DO UPDATE SET task=excluded.task, "
            "model_family=excluded.model_family, response_json=excluded.response_json, "
            "created_at=excluded.created_at, expires_at=excluded.expires_at, hits=0",
            key, task, model_family, json.dumps(response, ensure_ascii=False), now, expires_at,
        )

    async def cache_clear(self, task: str | None = None) -> int:
        result = await self._run("DELETE FROM cache" if task is None else "DELETE FROM cache WHERE task=?", *(() if task is None else (task,)))
        return _changes(result)

    async def cache_stats(self, *, now: int | None = None) -> dict[str, object]:
        timestamp = int(time.time()) if now is None else now
        total = await self._first(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits FROM cache "
            "WHERE expires_at IS NULL OR expires_at>?", timestamp
        ) or {}
        rows = _rows(await self.database.prepare(
            "SELECT task, COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS hits FROM cache "
            "WHERE expires_at IS NULL OR expires_at>? GROUP BY task ORDER BY task"
        ).bind(timestamp).all())
        return {
            "entries": int(total.get("entries", 0)), "hits": int(total.get("hits", 0)),
            "by_task": {str(row["task"]): {"entries": int(row["entries"]), "hits": int(row["hits"])} for row in rows},
        }

    async def budget_report(self, month: str, start: int, end: int) -> dict[str, object]:
        # Reports are intentionally read-only and computed from the durable log.
        rows = _rows(await self.database.prepare(
            "SELECT task, model_id, SUM(cost_usd) AS actual, SUM(shadow_cost_usd) AS shadow, "
            "SUM(CASE WHEN status NOT IN ('cache_hit','budget_blocked') THEN 1 ELSE 0 END) AS calls, "
            "SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) AS successes, "
            "SUM(CASE WHEN status='invalid_output' THEN 1 ELSE 0 END) AS validation_failures, "
            "AVG(CASE WHEN status NOT IN ('cache_hit','budget_blocked') THEN latency_ms END) AS mean_latency "
            "FROM call_log WHERE ts>=? AND ts<? GROUP BY task, model_id"
        ).bind(start, end).all())
        groups = _rows(await self.database.prepare(
            "SELECT budget_group, monthly_cap_usd, soft_warn_pct, hard_stop FROM budgets ORDER BY budget_group"
        ).all())
        task_groups = {str(row["task"]): str(row["budget_group"]) for row in _rows(await self.database.prepare("SELECT task, budget_group FROM task_routes").all())}
        by_group = {str(row["budget_group"]): {"spent_usd": 0.0, "monthly_cap_usd": float(row["monthly_cap_usd"]), "soft_warn_pct": float(row["soft_warn_pct"]), "hard_stop": bool(row["hard_stop"])} for row in groups}
        by_task: dict[str, dict[str, float]] = {}
        by_model: dict[str, dict[str, float]] = {}
        for row in rows:
            actual, shadow = float(row.get("actual", 0) or 0), float(row.get("shadow", 0) or 0)
            calls, successes, failures = int(row.get("calls", 0) or 0), int(row.get("successes", 0) or 0), int(row.get("validation_failures", 0) or 0)
            latency = row.get("mean_latency")
            task = str(row["task"])
            item = {"actual_cost_usd": actual, "shadow_cost_usd": shadow, "calls": float(calls), "successes": float(successes), "validation_failures": float(failures), "mean_latency_ms": float(latency or 0)}
            by_task[task] = item
            if row.get("model_id") is not None:
                by_model[str(row["model_id"])] = dict(item)
            group = task_groups.get(task)
            if group in by_group:
                by_group[group]["spent_usd"] = float(by_group[group]["spent_usd"]) + actual
        return {"month": month, "by_budget_group": by_group, "by_task": by_task, "by_model": by_model, "calls": sum(int(item["calls"]) for item in by_task.values()), "savings_usd": sum(item["shadow_cost_usd"] - item["actual_cost_usd"] for item in by_model.values())}

    async def close(self) -> None:
        # D1 bindings are owned by the Worker runtime; callers must not close them.
        return None


def _sqlite_value(value: object) -> object:
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, bool):
        return int(value)
    return value
