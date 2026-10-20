from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from llmrouter.config import ModelConfig, TaskRoute
from llmrouter.errors import NoModelAvailable
from llmrouter.limiter import RateLimiter, window_start
from llmrouter.selector import ModelSelector
from llmrouter.store.sqlite import SQLiteStore


def model(model_id: str = "m", **overrides: object) -> ModelConfig:
    values = dict(
        id=model_id, provider="fake", model_name=model_id, api_key_env="UNUSED", tier="T0",
        context_window=1000, max_output_tokens=100, input_price_per_mtok=0,
        output_price_per_mtok=0, rps=None, rpm=None, rpd=None, tpm=None, tpd=None,
        max_concurrency=100, enabled=True, data_policy="no_train",
    )
    values.update(overrides)
    return ModelConfig(**values)


def test_fixed_windows_align_to_utc_epoch() -> None:
    assert window_start("second", 123) == 123
    assert window_start("minute", 123) == 120
    assert window_start("day", 86_401) == 86_400
    with pytest.raises(ValueError):
        window_start("week", 123)


def test_atomic_daily_request_reservation_under_threads(tmp_path: Path) -> None:
    configured = model("limited", rpd=10, max_concurrency=60)
    with SQLiteStore(tmp_path / "limits.db") as store:
        limiter = RateLimiter(store, safety_fraction=0.9)

        def attempt(_: int):
            return limiter.reserve(configured, 1, 1, now=1_700_000_000)

        with ThreadPoolExecutor(max_workers=50) as pool:
            reservations = list(pool.map(attempt, range(50)))

        active = [reservation for reservation in reservations if reservation is not None]
        assert len(active) == 9
        usage = limiter.usage(configured, now=1_700_000_000)
        assert usage["day"]["requests"] == 9
        assert usage["day"]["input_tokens"] == 9
        for reservation in active:
            reservation.release()
        usage_after_release = limiter.usage(configured, now=1_700_000_000)
        assert usage_after_release["day"]["requests"] == 9
        assert usage_after_release["day"]["input_tokens"] == 0


def test_daily_token_limit_counts_input_plus_output(tmp_path: Path) -> None:
    configured = model("tokens", tpd=10, max_concurrency=2)
    with SQLiteStore(tmp_path / "tokens.db") as store:
        limiter = RateLimiter(store, safety_fraction=0.9)
        reservation = limiter.reserve(configured, 4, 5, now=100)
        assert reservation is not None
        assert limiter.reserve(configured, 1, 0, now=100) is None
        reservation.settle(3, 2)
        usage = limiter.usage(configured, now=100)
        assert usage["day"]["input_tokens"] == 3
        assert usage["day"]["output_tokens"] == 2


def test_model_concurrency_slot_is_held_until_resolved(tmp_path: Path) -> None:
    configured = model("one-at-a-time", max_concurrency=1)
    with SQLiteStore(tmp_path / "concurrency.db") as store:
        first_limiter = RateLimiter(store)
        second_limiter = RateLimiter(store)
        first = first_limiter.reserve(configured, 1, 1, now=100)
        assert first is not None
        assert second_limiter.reserve(configured, 1, 1, now=100) is None
        first.release()
        after_release = second_limiter.reserve(configured, 1, 1, now=100)
        assert after_release is not None
        after_release.release()


def test_cooldown_expires_and_status_reports_active_state(tmp_path: Path) -> None:
    configured = model("cooling")
    with SQLiteStore(tmp_path / "cooldowns.db") as store:
        limiter = RateLimiter(store)
        limiter.set_cooldown(configured.id, 200, "429", consecutive_failures=2)
        assert limiter.is_cooling_down(configured.id, now=199)
        assert limiter.status([configured], now=199)[0]["cooldown"] == {
            "until": 200, "reason": "429", "consecutive_failures": 2
        }
        assert not limiter.is_cooling_down(configured.id, now=200)
        assert limiter.status([configured], now=200)[0]["cooldown"] is None


def test_selector_blocks_private_unknown_and_skips_cooling_model(tmp_path: Path) -> None:
    unknown = model("unknown", data_policy="unknown", tier="T0")
    private = model("private", data_policy="no_train", tier="T1")
    route = TaskRoute(
        tier_order=["T0", "T1"], requires={}, budget_group="default", sensitivity="private"
    )
    with SQLiteStore(tmp_path / "selector.db") as store:
        limiter = RateLimiter(store)
        limiter.set_cooldown("private", 500, "test")
        selector = ModelSelector([unknown, private], limiter)
        with pytest.raises(NoModelAvailable):
            selector.select_and_reserve(
                route, estimated_input_tokens=10, max_output_tokens=5, now=400
            )
        limiter.clear_cooldown("private")
        selected = selector.select_and_reserve(
            route, estimated_input_tokens=10, max_output_tokens=5, now=400
        )
        assert selected.model.id == "private"
        selected.reservation.release()
