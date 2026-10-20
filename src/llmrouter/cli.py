"""Command-line interface for inspecting and operating a local router."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from dotenv import load_dotenv

from llmrouter import Router


app = typer.Typer(help="Route LLM calls and inspect local usage.", no_args_is_help=True)
cache_app = typer.Typer(help="Inspect or clear cached responses.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@app.callback()
def configure(
    ctx: typer.Context,
    config: Path = typer.Option(Path("config/router.yaml"), "--config", help="Router YAML file."),
    db: Path = typer.Option(Path("data/router.db"), "--db", help="SQLite database file."),
) -> None:
    ctx.ensure_object(dict)
    ctx.obj.update(config=config, db=db)


def _router(ctx: typer.Context) -> Router:
    return Router.from_config(ctx.obj["config"], db_path=ctx.obj["db"])


def _print_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False, default=str))


@app.command()
def status(ctx: typer.Context) -> None:
    """Show live model usage, limits, and cooldowns."""
    with _router(ctx) as router:
        _print_json(router.status())


@app.command()
def report(
    ctx: typer.Context,
    month: str | None = typer.Option(None, "--month", help="UTC calendar month in YYYY-MM format."),
) -> None:
    """Show monthly spend, savings, calls, latency, and validation rates."""
    with _router(ctx) as router:
        _print_json(router.report(month))


@app.command("models")
def models_command(ctx: typer.Context) -> None:
    """List configured models, enabled state, and latest health signal."""
    with _router(ctx) as router:
        status_by_id = {item["model_id"]: item for item in router.status()}
        output = []
        for model in router.config.models:
            latest = router.store.connection.execute(
                "SELECT status FROM call_log WHERE model_id=? AND status NOT IN ('cache_hit','budget_blocked') "
                "ORDER BY ts DESC, rowid DESC LIMIT 1", (model.id,)
            ).fetchone()
            cooldown = status_by_id[model.id]["cooldown"]
            health = (
                "disabled" if not model.enabled else
                "cooldown" if cooldown is not None else
                "healthy" if latest is not None and latest[0] == "ok" else
                "degraded" if latest is not None else "unknown"
            )
            output.append({
                "id": model.id, "provider": model.provider,
                "model_name": model.model_name, "tier": model.tier,
                "enabled": model.enabled, "health": health,
                "last_status": None if latest is None else latest[0],
                "cooldown": cooldown,
            })
        _print_json(output)


@app.command("test-model")
def test_model(
    ctx: typer.Context,
    model_id: str = typer.Argument(..., help="Configured model ID to check."),
    task: str | None = typer.Option(None, "--task", help="Task route whose budget group to use."),
) -> None:
    """Send a tiny live prompt to one configured model."""
    load_dotenv()
    try:
        with _router(ctx) as router:
            result = router.test_model(model_id, task=task)
        _print_json({
            "model_id": result.model_id, "cost_usd": result.cost_usd,
            "estimated_cost_usd": result.estimated_cost_usd,
            "comparison_cost_usd": result.comparison_cost_usd,
            "savings_usd": result.savings_usd,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "text": result.text,
        })
    except Exception as exc:
        typer.echo(f"Model test failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def call(
    ctx: typer.Context,
    task: str = typer.Argument(..., help="Configured task route."),
    message: str = typer.Option(..., "--message", help="User message to send."),
    caller: str | None = typer.Option(None, "--caller", help="Optional caller label."),
    no_cache: bool = typer.Option(False, "--no-cache", help="Skip cache reads and writes."),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd", min=0),
) -> None:
    """Send a text request through configured routing."""
    load_dotenv()
    try:
        with _router(ctx) as router:
            result = router.call(
                task, [{"role": "user", "content": message}], caller=caller,
                use_cache=not no_cache, max_cost_usd=max_cost_usd,
            )
        _print_json({
            "request_id": result.request_id, "model_id": result.model_id,
            "cached": result.cached, "cost_usd": result.cost_usd,
            "estimated_cost_usd": result.estimated_cost_usd,
            "comparison_cost_usd": result.comparison_cost_usd,
            "savings_usd": result.savings_usd,
            "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
            "text": result.text,
        })
    except Exception as exc:
        typer.echo(f"llmrouter call failed: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@cache_app.command("stats")
def cache_stats(ctx: typer.Context) -> None:
    """Show active cache entry and hit counts."""
    with _router(ctx) as router:
        _print_json(router.store.cache_stats())


@cache_app.command("clear")
def cache_clear(
    ctx: typer.Context,
    task: str | None = typer.Option(None, "--task", help="Clear one task's entries."),
) -> None:
    """Clear all cache entries, or just entries for one task."""
    with _router(ctx) as router:
        deleted = router.cache_clear(task)
    typer.echo(f"Deleted {deleted} cache entr{'y' if deleted == 1 else 'ies'}.")


@app.command()
def replay(
    ctx: typer.Context,
    request_id: str = typer.Argument(..., help="Logical request ID to inspect."),
) -> None:
    """Show all persisted attempts for a logical request."""
    with _router(ctx) as router:
        rows = router.store.connection.execute(
            "SELECT ts, task, caller, model_id, attempt_no, status, http_status, latency_ms, "
            "input_tokens, output_tokens, cost_usd, error_message, prompt_hash "
            "FROM call_log WHERE request_id=? ORDER BY attempt_no, rowid",
            (request_id,),
        ).fetchall()
    if not rows:
        typer.echo(f"No attempts found for request {request_id}.", err=True)
        raise typer.Exit(code=1)
    _print_json({"request_id": request_id, "attempts": [dict(row) for row in rows]})


def main() -> None:
    app()


if __name__ == "__main__":
    main()
