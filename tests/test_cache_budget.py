from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pytest

from llmrouter.adapters.base import NormalizedResponse
from llmrouter.adapters.fake import ScriptedFakeAdapter
from llmrouter.config import RouterConfig
from llmrouter.errors import BudgetExceeded, NoModelAvailable
from llmrouter.router import Router
from llmrouter.store.sqlite import SQLiteStore


def config(*, cap: float = 10.0, free: bool = True, ttl: int | None = None) -> RouterConfig:
    return RouterConfig.model_validate({
        "tiers": ["T0"],
        "models": [{
            "id": "fake", "provider": "fake", "model_name": "fake-model",
            "api_key_env": "UNUSED", "tier": "T0", "enabled": True,
            "context_window": 8192, "max_output_tokens": 32,
            "input_price_per_mtok": 1000, "output_price_per_mtok": 1000,
            "is_free_tier": free, "data_policy": "no_train",
        }],
        "task_routes": {"answer": {
            "tier_order": ["T0"], "requires": {}, "default_temperature": 0,
            "max_attempts": 1, "budget_group": "default", "sensitivity": "public",
            "cache_ttl_seconds": ttl,
        }},
        "budgets": {"default": {
            "monthly_cap_usd": cap, "soft_warn_pct": 80, "hard_stop": True,
        }},
    })


def response(text: str = "hello") -> NormalizedResponse:
    return NormalizedResponse(text=text, input_tokens=4, output_tokens=2)


def make_router(tmp_path: Path, fake: ScriptedFakeAdapter, **config_options) -> Router:
    return Router(
        config(**config_options), SQLiteStore(tmp_path / "router.db"),
        adapters={"fake": fake},
    )


def test_cache_hit_skips_provider_and_cache_clear_is_scoped(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"fake": [response(), response("different")]})
    router = make_router(tmp_path, fake)
    request = [{"role": "user", "content": "same input"}]
    try:
        first = router.call("answer", request)
        second = router.call("answer", request)
        assert first.cached is False
        assert second.cached is True
        assert second.text == "hello"
        assert second.cost_usd == 0
        assert len(fake.calls) == 1
        assert [row[0] for row in router.store.connection.execute(
            "SELECT status FROM call_log ORDER BY rowid"
        )] == ["ok", "cache_hit"]
        assert router.cache_clear("other-task") == 0
        assert router.cache_clear("answer") == 1
        third = router.call("answer", request, use_cache=False)
        assert third.text == "different"
        assert len(fake.calls) == 2
    finally:
        router.close()


def test_cache_does_not_bypass_model_disablement(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"fake": [response()]})
    router = make_router(tmp_path, fake, ttl=3600)
    request = [{"role": "user", "content": "policy-sensitive input"}]
    try:
        router.call("answer", request)
        router.config.models[0].enabled = False
        with pytest.raises(NoModelAvailable):
            router.call("answer", request)
        assert len(fake.calls) == 1
    finally:
        router.close()


def test_cache_ttl_expiration(tmp_path: Path) -> None:
    store = SQLiteStore(tmp_path / "cache.db")
    try:
        now = int(time.time())
        store.cache_put("key", "answer", "family", {"text": "x"}, 2)
        assert store.cache_get("key", now=now + 1) == {"text": "x"}
        assert store.cache_get("key", now=now + 3) is None
    finally:
        store.close()


def test_per_call_cost_ceiling_blocks_before_provider_request(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"fake": [response()]})
    router = make_router(tmp_path, fake, free=False)
    try:
        with pytest.raises(BudgetExceeded, match="max_cost_usd"):
            router.call("answer", [{"role": "user", "content": "charge"}], max_cost_usd=0)
        assert fake.calls == []
        assert router.store.connection.execute(
            "SELECT status FROM call_log"
        ).fetchone()[0] == "budget_blocked"
        assert router.store.connection.execute(
            "SELECT COUNT(*) FROM budget_reservations"
        ).fetchone()[0] == 0
    finally:
        router.close()


def test_monthly_budget_reservations_are_atomic_under_concurrency(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter()
    router = make_router(tmp_path, fake, cap=0.01)
    store = router.store

    def reserve(index: int) -> bool:
        try:
            store.reserve_budget(str(index), str(index), "default", 0.001, 4_000_000_000)
            return True
        except BudgetExceeded:
            return False

    try:
        with ThreadPoolExecutor(max_workers=20) as pool:
            accepted = list(pool.map(reserve, range(50)))
        assert sum(accepted) == 10
        assert store.connection.execute(
            "SELECT COUNT(*) FROM budget_reservations WHERE status='active'"
        ).fetchone()[0] == 10
    finally:
        router.close()


def test_actual_cost_settles_reservation_and_report_groups_spend(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"fake": [response()]})
    router = make_router(tmp_path, fake, free=False)
    try:
        result = router.call("answer", [{"role": "user", "content": "charge"}], use_cache=False)
        row = router.store.connection.execute(
            "SELECT status, reserved_usd FROM budget_reservations"
        ).fetchone()
        assert row[0] == "settled"
        assert row[1] > result.cost_usd
        report = router.report()
        assert report["by_budget_group"]["default"]["spent_usd"] == pytest.approx(result.cost_usd)
        assert report["by_task"]["answer"]["actual_cost_usd"] == pytest.approx(result.cost_usd)
        assert report["calls"] == 1
        assert report["by_task"]["answer"]["success_rate"] == 1
        assert report["by_task"]["answer"]["mean_latency_ms"] >= 0
    finally:
        router.close()


def test_fifty_concurrent_calls_cannot_overrun_monthly_hard_cap(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"fake": [response() for _ in range(50)]})
    configured = config(cap=0.10, free=False)
    configured.models[0].max_concurrency = 50
    router = Router(
        configured, SQLiteStore(tmp_path / "router.db"), adapters={"fake": fake}
    )
    closed = False

    def invoke(index: int) -> bool:
        try:
            router.call(
                "answer", [{"role": "user", "content": f"parallel request {index}"}],
                use_cache=False,
            )
            return True
        except BudgetExceeded:
            return False

    try:
        with ThreadPoolExecutor(max_workers=50) as pool:
            accepted = list(pool.map(invoke, range(50)))
        assert 0 < sum(accepted) < 50
        settled = router.store.connection.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM call_log WHERE status='ok'"
        ).fetchone()[0]
        assert settled <= 0.10
        assert router.store.connection.execute(
            "SELECT COUNT(*) FROM budget_reservations WHERE status='active'"
        ).fetchone()[0] == 0

        expected_report_spend = router.report()["by_budget_group"]["default"]["spent_usd"]
        router.close()
        closed = True
        reopened = Router(config(cap=0.10, free=False), SQLiteStore(tmp_path / "router.db"))
        try:
            assert reopened.report()["by_budget_group"]["default"]["spent_usd"] == pytest.approx(
                expected_report_spend
            )
        finally:
            reopened.close()
    finally:
        if not closed:
            router.close()


def test_expired_budget_hold_is_recovered_on_database_reopen(tmp_path: Path) -> None:
    path = tmp_path / "crash.db"
    original = Router(config(), SQLiteStore(path), adapters={"fake": ScriptedFakeAdapter()})
    closed = False
    try:
        original.store.reserve_budget(
            "abandoned", "interrupted-call", "default", 5.0, int(time.time()) - 1
        )
        original.close()
        closed = True
        reopened_store = SQLiteStore(path)
        try:
            state = reopened_store.connection.execute(
                "SELECT status FROM budget_reservations WHERE id='abandoned'"
            ).fetchone()[0]
            assert state == "expired"
            assert reopened_store.reserve_budget(
                "after-restart", "new-call", "default", 0.0, int(time.time()) + 60
            )[1] == pytest.approx(10.0)
        finally:
            reopened_store.close()
    finally:
        if not closed:
            original.close()


def test_report_calculates_free_tier_shadow_savings(tmp_path: Path) -> None:
    fake = ScriptedFakeAdapter({"fake": [response()]})
    configured = config(free=True)
    configured.models[0].shadow_input_price_per_mtok = 1000
    configured.models[0].shadow_output_price_per_mtok = 1000
    router = Router(configured, SQLiteStore(tmp_path / "savings.db"), adapters={"fake": fake})
    try:
        result = router.call(
            "answer", [{"role": "user", "content": "shadow pricing"}], use_cache=False
        )
        report = router.report()
        assert result.cost_usd == 0
        assert report["by_model"]["fake"]["shadow_cost_usd"] == pytest.approx(0.006)
        assert report["savings_usd"] == pytest.approx(0.006)
    finally:
        router.close()
