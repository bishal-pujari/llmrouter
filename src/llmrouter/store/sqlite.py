"""SQLite implementation of the Store boundary."""

from __future__ import annotations

import json
import calendar
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

from llmrouter.config import RouterConfig
from llmrouter.errors import BudgetExceeded
from llmrouter.store.protocol import CallLogEntry


class SQLiteStore:
    """A small WAL-backed store; all timestamps are UTC epoch seconds."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        if self.db_path != ":memory:":
            self._prepare_private_database_path(Path(self.db_path))
        self._connection = sqlite3.connect(
            self.db_path, timeout=30, isolation_level=None, check_same_thread=False
        )
        # A connection is shared by threads in this process; this lock keeps a
        # transaction's BEGIN/COMMIT pair from interleaving with another caller.
        self._lock = threading.RLock()
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        self.migrate()
        self._connection.execute(
            "UPDATE budget_reservations SET status='expired' "
            "WHERE status='active' AND expires_at<=?", (int(time.time()),)
        )
        self._harden_database_files()

    @staticmethod
    def _prepare_private_database_path(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
            if candidate.is_symlink():
                raise ValueError(
                    f"Refusing to open SQLite database through symlink: {candidate}"
                )
        if not path.exists():
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags, 0o600)
            except FileExistsError:
                # Another process created the database between exists() and
                # open(); re-check it below before SQLite opens it.
                pass
            else:
                os.close(descriptor)
        if path.is_symlink():
            raise ValueError(f"Refusing to open SQLite database through symlink: {path}")
        os.chmod(path, 0o600)

    def _harden_database_files(self) -> None:
        """Keep the database and SQLite sidecars owner-readable only."""
        for suffix in ("", "-wal", "-shm"):
            path = Path(self.db_path + suffix)
            if path.exists() and not path.is_symlink():
                os.chmod(path, 0o600)

    @property
    def connection(self) -> sqlite3.Connection:
        """Exposed for store-layer components; application code should use Store methods."""
        return self._connection

    def migrate(self) -> None:
        migration = files("llmrouter.store").joinpath("migrations/001_initial.sql")
        sql = migration.read_text(encoding="utf-8")
        self._connection.executescript(sql)
        self._connection.execute(
            "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
            (1, int(time.time())),
        )

    @contextmanager
    def transaction(self):
        """Serialize this connection and acquire SQLite's cross-process write lock."""
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise

    def upsert_config(self, config: RouterConfig) -> None:
        """Replace the DB's config snapshot atomically after YAML validation."""
        payload = config.model_dump(mode="json")
        connection = self._connection
        with self.transaction():
            for model in payload["models"]:
                columns = tuple(model)
                values = tuple(_sqlite_value(model[name]) for name in columns)
                updates = ", ".join(f"{name}=excluded.{name}" for name in columns if name != "id")
                connection.execute(
                    f"INSERT INTO models ({', '.join(columns)}) VALUES ({', '.join('?' for _ in columns)}) "
                    f"ON CONFLICT(id) DO UPDATE SET {updates}", values
                )
            configured_ids = [model["id"] for model in payload["models"]]
            if configured_ids:
                marks = ",".join("?" for _ in configured_ids)
                connection.execute(f"DELETE FROM models WHERE id NOT IN ({marks})", configured_ids)
            else:
                connection.execute("DELETE FROM models")

            for task, route in payload["task_routes"].items():
                connection.execute(
                    "INSERT INTO task_routes VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(task) DO UPDATE SET tier_order=excluded.tier_order, "
                    "requires=excluded.requires, default_temperature=excluded.default_temperature, "
                    "max_attempts=excluded.max_attempts, budget_group=excluded.budget_group, "
                    "sensitivity=excluded.sensitivity, cache_ttl_seconds=excluded.cache_ttl_seconds",
                    (task, json.dumps(route["tier_order"]), json.dumps(route["requires"]),
                     route["default_temperature"], route["max_attempts"], route["budget_group"],
                     route["sensitivity"], route["cache_ttl_seconds"]),
                )
            tasks = list(payload["task_routes"])
            if tasks:
                connection.execute(
                    f"DELETE FROM task_routes WHERE task NOT IN ({','.join('?' for _ in tasks)})", tasks
                )
            else:
                connection.execute("DELETE FROM task_routes")

            for group, budget in payload["budgets"].items():
                connection.execute(
                    "INSERT INTO budgets(budget_group, monthly_cap_usd, soft_warn_pct, hard_stop, month, spent_usd) "
                    "VALUES (?, ?, ?, ?, strftime('%Y-%m','now'), 0) "
                    "ON CONFLICT(budget_group) DO UPDATE SET monthly_cap_usd=excluded.monthly_cap_usd, "
                    "soft_warn_pct=excluded.soft_warn_pct, hard_stop=excluded.hard_stop",
                    (group, budget["monthly_cap_usd"], budget["soft_warn_pct"], int(budget["hard_stop"])),
                )
            groups = list(payload["budgets"])
            if groups:
                connection.execute(
                    f"DELETE FROM budgets WHERE budget_group NOT IN ({','.join('?' for _ in groups)})", groups
                )
            else:
                connection.execute("DELETE FROM budgets")

            connection.execute("DELETE FROM tiers")
            connection.executemany(
                "INSERT INTO tiers(name, ordinal) VALUES (?, ?)",
                [(name, index) for index, name in enumerate(payload["tiers"])],
            )
            connection.execute("DELETE FROM global_settings")
            connection.executemany(
                "INSERT INTO global_settings(key, value_json) VALUES (?, ?)",
                [(key, json.dumps(value)) for key, value in payload["settings"].items()],
            )

    def reserve_model_usage(
        self,
        model_id: str,
        starts: dict[str, int],
        request_limits: dict[str, int | None],
        token_limits: dict[str, int | None],
        input_tokens: int,
        output_tokens: int,
    ) -> bool:
        """Reserve all configured windows atomically before contacting a provider."""
        with self.transaction() as connection:
            rows = {}
            for name, start in starts.items():
                row = connection.execute(
                    "SELECT requests, input_tokens, output_tokens FROM usage_windows "
                    "WHERE model_id=? AND window_type=? AND window_start=?",
                    (model_id, name, start),
                ).fetchone()
                rows[name] = (0, 0, 0) if row is None else tuple(row)

            for window, limit in request_limits.items():
                if limit is not None and rows[window][0] + 1 > limit:
                    return False
            token_reservation = input_tokens + output_tokens
            for window, limit in token_limits.items():
                current = rows[window][1] + rows[window][2]
                if limit is not None and current + token_reservation > limit:
                    return False

            for window, start in starts.items():
                connection.execute(
                    "INSERT INTO usage_windows "
                    "(model_id, window_type, window_start, requests, input_tokens, output_tokens, cost_usd) "
                    "VALUES (?, ?, ?, 1, ?, ?, 0) "
                    "ON CONFLICT(model_id, window_type, window_start) DO UPDATE SET "
                    "requests=requests+1, input_tokens=input_tokens+excluded.input_tokens, "
                    "output_tokens=output_tokens+excluded.output_tokens",
                    (model_id, window, start, input_tokens, output_tokens),
                )
        return True

    def reconcile_model_usage(
        self,
        model_id: str,
        starts: dict[str, int],
        estimated_input: int,
        estimated_output: int,
        actual_input: int,
        actual_output: int,
        cost_usd: float,
    ) -> None:
        with self.transaction() as connection:
            for window, start in starts.items():
                connection.execute(
                    "UPDATE usage_windows SET input_tokens=input_tokens-?+?, "
                    "output_tokens=output_tokens-?+?, cost_usd=cost_usd+? "
                    "WHERE model_id=? AND window_type=? AND window_start=?",
                    (estimated_input, actual_input, estimated_output, actual_output, cost_usd,
                     model_id, window, start),
                )

    def read_usage_window(self, model_id: str, window: str, start: int) -> tuple[int, int, int] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT requests, input_tokens, output_tokens FROM usage_windows "
                "WHERE model_id=? AND window_type=? AND window_start=?",
                (model_id, window, start),
            ).fetchone()
        return None if row is None else tuple(row)

    def read_cooldown(self, model_id: str, now: int) -> tuple[int, str, int] | None:
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT cooldown_until, reason, consecutive_failures FROM cooldowns WHERE model_id=?",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            if row[0] <= now:
                connection.execute("DELETE FROM cooldowns WHERE model_id=?", (model_id,))
                return None
            return tuple(row)

    def write_cooldown(
        self, model_id: str, cooldown_until: int, reason: str, consecutive_failures: int
    ) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO cooldowns(model_id, cooldown_until, reason, consecutive_failures) "
                "VALUES (?, ?, ?, ?) ON CONFLICT(model_id) DO UPDATE SET "
                "cooldown_until=excluded.cooldown_until, reason=excluded.reason, "
                "consecutive_failures=excluded.consecutive_failures",
                (model_id, cooldown_until, reason[:200], consecutive_failures),
            )

    def delete_cooldown(self, model_id: str) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM cooldowns WHERE model_id=?", (model_id,))

    def log_call(self, entry: CallLogEntry) -> None:
        """Persist one completed provider attempt without storing prompt contents."""
        with self.transaction() as connection:
            self._insert_call_log(connection, entry)

    def reserve_budget(
        self,
        reservation_id: str,
        request_id: str,
        budget_group: str,
        worst_case_usd: float,
        expires_at: int,
    ) -> tuple[float, float, float]:
        """Atomically sweep stale holds, enforce the monthly cap, and reserve cost."""
        now = int(time.time())
        month = time.strftime("%Y-%m", time.gmtime(now))
        month_start = calendar.timegm(time.strptime(month + "-01", "%Y-%m-%d"))
        with self.transaction() as connection:
            connection.execute(
                "UPDATE budget_reservations SET status='expired' "
                "WHERE status='active' AND expires_at<=?", (now,)
            )
            budget = connection.execute(
                "SELECT monthly_cap_usd, soft_warn_pct, hard_stop, month, spent_usd "
                "FROM budgets WHERE budget_group=?", (budget_group,)
            ).fetchone()
            if budget is None:
                raise ValueError(f"Unknown budget group {budget_group!r}")
            cap, soft_warn_pct, hard_stop, budget_month, spent = budget
            if budget_month != month:
                spent = 0.0
            # Treat the durable log as the source of truth, including after a crash
            # or an upgrade from a version that did not maintain budget totals.
            spent = connection.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM call_log "
                "WHERE ts>=? AND task IN (SELECT task FROM task_routes WHERE budget_group=?)",
                (month_start, budget_group),
            ).fetchone()[0]
            if budget_month != month or abs(float(budget[4]) - float(spent)) > 1e-12:
                connection.execute(
                    "UPDATE budgets SET month=?, spent_usd=? WHERE budget_group=?",
                    (month, spent, budget_group),
                )
            active = connection.execute(
                "SELECT COALESCE(SUM(reserved_usd), 0) FROM budget_reservations "
                "WHERE budget_group=? AND status='active' AND created_at>=?",
                (budget_group, month_start),
            ).fetchone()[0]
            if hard_stop and spent + active + worst_case_usd > cap + 1e-12:
                raise BudgetExceeded(
                    f"Budget group {budget_group!r} has ${max(0.0, cap-spent-active):.6f} "
                    f"available; this attempt may cost ${worst_case_usd:.6f}",
                    budget_group=budget_group,
                )
            connection.execute(
                "INSERT INTO budget_reservations "
                "(id, request_id, budget_group, reserved_usd, status, created_at, expires_at) "
                "VALUES (?, ?, ?, ?, 'active', ?, ?)",
                (reservation_id, request_id, budget_group, worst_case_usd, now, expires_at),
            )
            return float(spent), float(cap), float(soft_warn_pct)

    def release_budget(self, reservation_id: str, *, expired: bool = False) -> None:
        with self.transaction() as connection:
            connection.execute(
                "UPDATE budget_reservations SET status=? WHERE id=? AND status='active'",
                ("expired" if expired else "released", reservation_id),
            )

    def settle_budget_and_log(self, reservation_id: str, entry: CallLogEntry) -> None:
        """Settle actual spend and persist its attempt log in one transaction."""
        with self.transaction() as connection:
            reservation = connection.execute(
                "SELECT budget_group, status FROM budget_reservations WHERE id=?",
                (reservation_id,),
            ).fetchone()
            if reservation is None or reservation[1] != "active":
                raise RuntimeError("Budget reservation is missing or no longer active")
            connection.execute(
                "UPDATE budget_reservations SET status='settled' WHERE id=?", (reservation_id,)
            )
            connection.execute(
                "UPDATE budgets SET spent_usd=spent_usd+? WHERE budget_group=?",
                (entry.cost_usd, reservation[0]),
            )
            self._insert_call_log(connection, entry)

    def cache_get(self, key: str, *, now: int | None = None) -> dict[str, object] | None:
        timestamp = int(time.time()) if now is None else now
        with self.transaction() as connection:
            row = connection.execute(
                "SELECT response_json, expires_at FROM cache WHERE key=?", (key,)
            ).fetchone()
            if row is None:
                return None
            if row[1] is not None and row[1] <= timestamp:
                connection.execute("DELETE FROM cache WHERE key=?", (key,))
                return None
            connection.execute("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
            return json.loads(row[0])

    def cache_put(
        self, key: str, task: str, model_family: str,
        response: dict[str, object], ttl_seconds: int | None,
    ) -> None:
        now = int(time.time())
        expires_at = None if ttl_seconds is None else now + ttl_seconds
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO cache(key, task, model_family, response_json, created_at, expires_at, hits) "
                "VALUES (?, ?, ?, ?, ?, ?, 0) ON CONFLICT(key) DO UPDATE SET "
                "task=excluded.task, model_family=excluded.model_family, response_json=excluded.response_json, "
                "created_at=excluded.created_at, expires_at=excluded.expires_at, hits=0",
                (key, task, model_family, json.dumps(response, ensure_ascii=False), now, expires_at),
            )

    def cache_clear(self, task: str | None = None) -> int:
        with self.transaction() as connection:
            if task is None:
                cursor = connection.execute("DELETE FROM cache")
            else:
                cursor = connection.execute("DELETE FROM cache WHERE task=?", (task,))
            return cursor.rowcount

    def budget_report(self, month: str, start: int, end: int) -> dict[str, object]:
        with self._lock:
            budgets = self._connection.execute(
                "SELECT budget_group, monthly_cap_usd, soft_warn_pct, hard_stop "
                "FROM budgets ORDER BY budget_group"
            ).fetchall()
            task_groups = {
                row[0]: row[1] for row in self._connection.execute(
                    "SELECT task, budget_group FROM task_routes"
                ).fetchall()
            }
            rows = self._connection.execute(
                "SELECT task, model_id, SUM(cost_usd), SUM(shadow_cost_usd), "
                "SUM(CASE WHEN status NOT IN ('cache_hit','budget_blocked') THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN status='invalid_output' THEN 1 ELSE 0 END), "
                "AVG(CASE WHEN status NOT IN ('cache_hit','budget_blocked') THEN latency_ms END) "
                "FROM call_log WHERE ts>=? AND ts<? GROUP BY task, model_id",
                (start, end),
            ).fetchall()

        by_group: dict[str, dict[str, float | bool]] = {
            row[0]: {"spent_usd": 0.0, "monthly_cap_usd": float(row[1]),
                     "soft_warn_pct": float(row[2]), "hard_stop": bool(row[3])}
            for row in budgets
        }
        by_task: dict[str, dict[str, float]] = {}
        by_model: dict[str, dict[str, float]] = {}
        for task, model_id, actual, shadow, calls, successes, validation_failures, mean_latency in rows:
            actual_value, shadow_value = float(actual or 0), float(shadow or 0)
            group = task_groups.get(task)
            if group in by_group:
                by_group[group]["spent_usd"] = float(by_group[group]["spent_usd"]) + actual_value
            _add_report_cost(
                by_task, task, actual_value, shadow_value,
                int(calls), int(successes or 0), int(validation_failures or 0), mean_latency,
            )
            if model_id is not None:
                _add_report_cost(
                    by_model, model_id, actual_value, shadow_value,
                    int(calls), int(successes or 0), int(validation_failures or 0), mean_latency,
                )
        savings = sum(value["shadow_cost_usd"] - value["actual_cost_usd"] for value in by_model.values())
        for group in (by_task, by_model):
            for item in group.values():
                item.pop("latency_total_ms", None)
        return {
            "month": month,
            "by_budget_group": by_group,
            "by_task": by_task,
            "by_model": by_model,
            "calls": sum(int(item["calls"]) for item in by_task.values()),
            "savings_usd": savings,
        }

    def cache_stats(self, *, now: int | None = None) -> dict[str, object]:
        timestamp = int(time.time()) if now is None else now
        with self._lock:
            total = self._connection.execute(
                "SELECT COUNT(*), COALESCE(SUM(hits), 0) FROM cache "
                "WHERE expires_at IS NULL OR expires_at>?", (timestamp,)
            ).fetchone()
            by_task = self._connection.execute(
                "SELECT task, COUNT(*), COALESCE(SUM(hits), 0) FROM cache "
                "WHERE expires_at IS NULL OR expires_at>? GROUP BY task ORDER BY task",
                (timestamp,),
            ).fetchall()
        return {
            "entries": int(total[0]), "hits": int(total[1]),
            "by_task": {
                row[0]: {"entries": int(row[1]), "hits": int(row[2])}
                for row in by_task
            },
        }

    @staticmethod
    def _insert_call_log(connection: sqlite3.Connection, entry: CallLogEntry) -> None:
        connection.execute(
            "INSERT INTO call_log (id, ts, request_id, task, caller, model_id, attempt_no, "
            "status, http_status, latency_ms, input_tokens, output_tokens, cost_usd, "
            "shadow_cost_usd, error_message, prompt_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (entry.id, entry.ts, entry.request_id, entry.task, entry.caller, entry.model_id,
             entry.attempt_no, entry.status, entry.http_status, entry.latency_ms,
             entry.input_tokens, entry.output_tokens, entry.cost_usd, entry.shadow_cost_usd,
             entry.error_message, entry.prompt_hash),
        )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _add_report_cost(
    destination: dict[str, dict[str, float]], key: str,
    actual: float, shadow: float, calls: int, successes: int,
    validation_failures: int, mean_latency_ms: float | None,
) -> None:
    item = destination.setdefault(key, {
        "actual_cost_usd": 0.0, "shadow_cost_usd": 0.0, "calls": 0.0,
        "successes": 0.0, "validation_failures": 0.0, "latency_total_ms": 0.0,
    })
    item["actual_cost_usd"] += actual
    item["shadow_cost_usd"] += shadow
    item["calls"] += calls
    item["successes"] += successes
    item["validation_failures"] += validation_failures
    item["latency_total_ms"] += (mean_latency_ms or 0.0) * calls
    item["success_rate"] = item["successes"] / item["calls"] if item["calls"] else 0.0
    item["validation_failure_rate"] = (
        item["validation_failures"] / item["calls"] if item["calls"] else 0.0
    )
    item["mean_latency_ms"] = item["latency_total_ms"] / item["calls"] if item["calls"] else 0.0


def _sqlite_value(value: object) -> object:
    if isinstance(value, bool):
        return int(value)
    return value
