from pathlib import Path

import pytest
from pydantic import ValidationError

from llmrouter.config import ModelConfig, load_config
from llmrouter.store.sqlite import SQLiteStore


def test_example_config_is_valid_and_references_resolve() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "router.example.yaml")
    assert config.task_routes["example_task"].budget_group == "default"


def test_sqlite_migration_and_config_upsert(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "config" / "router.example.yaml")
    db_path = tmp_path / "router.db"
    with SQLiteStore(db_path) as store:
        store.upsert_config(config)
        tables = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"models", "task_routes", "usage_windows", "budgets", "budget_reservations"} <= tables
        model_row = store.connection.execute("SELECT id, enabled FROM models").fetchone()
        assert model_row["id"] == "placeholder-free-model"
        assert model_row["enabled"] == 0
        reservation_columns = {
            row[1] for row in store.connection.execute("PRAGMA table_info(budget_reservations)")
        }
        assert {"reserved_usd", "status", "expires_at"} <= reservation_columns

    # Reopening the same file proves that the migration marker and stored config persist.
    with SQLiteStore(db_path) as reopened:
        assert reopened.connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0] == 1
        assert reopened.connection.execute("SELECT COUNT(*) FROM task_routes").fetchone()[0] == 1


@pytest.mark.parametrize(
    "base_url",
    [
        "http://remote.example/v1",
        "https://remote.example/v1?api_key=leak",
        "https://user:password@remote.example/v1",
        "file:///tmp/provider",
    ],
)
def test_model_endpoint_rejects_credential_leaking_urls(base_url: str) -> None:
    with pytest.raises(ValidationError):
        ModelConfig.model_validate({
            "id": "model", "provider": "openai_compat", "model_name": "model",
            "base_url": base_url, "api_key_env": "TEST_KEY", "tier": "T0",
            "context_window": 100, "max_output_tokens": 10,
            "input_price_per_mtok": 0, "output_price_per_mtok": 0,
        })


def test_model_endpoint_allows_loopback_http_for_local_provider() -> None:
    model = ModelConfig.model_validate({
        "id": "model", "provider": "openai_compat", "model_name": "model",
        "base_url": "http://127.0.0.1:8000/v1", "api_key_env": "TEST_KEY", "tier": "T0",
        "context_window": 100, "max_output_tokens": 10,
        "input_price_per_mtok": 0, "output_price_per_mtok": 0,
    })
    assert model.base_url.startswith("http://127.0.0.1")
