"""Run a small live batch using the configured task route."""

from __future__ import annotations

import argparse
import asyncio

from dotenv import load_dotenv

from llmrouter import Router


async def run(config: str, db: str, task: str, messages: list[str], concurrency: int) -> None:
    # .env is loaded by this development example, not by the library itself.
    load_dotenv()
    with Router.from_config(config, db_path=db) as router:
        items = [
            [{"role": "user", "content": message}]
            for message in messages
        ]
        outcomes = await router.batch(
            items,
            task=task,
            concurrency=concurrency,
            on_progress=lambda done, total, item: print(
                f"completed {done}/{total} (item {item.index})"
            ),
        )
        for item in outcomes:
            if item.ok:
                print(f"item {item.index} [{item.result.model_id}]: {item.result.text}")
            else:
                print(f"item {item.index} failed: {item.error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/router.yaml")
    parser.add_argument("--db", default="data/router.db")
    parser.add_argument("--task", default="example_task")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument(
        "--message", action="append", dest="messages",
        help="Message to send; repeat this option to add more batch items.",
    )
    args = parser.parse_args()
    asyncio.run(run(
        args.config, args.db, args.task,
        args.messages or ["Reply with the word OK."], args.concurrency,
    ))


if __name__ == "__main__":
    main()
