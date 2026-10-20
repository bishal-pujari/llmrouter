"""Small manual smoke client for a configured OpenAI-compatible endpoint."""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv

from llmrouter import Router


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/router.yaml")
    parser.add_argument("--db", default="data/router.db")
    parser.add_argument("--task", default="example_task")
    parser.add_argument("--message", default="Reply with the word OK.")
    parser.add_argument("--caller", default="manual_smoke_test")
    args = parser.parse_args()

    # Loading .env is kept in this development client; the library reads only env vars.
    load_dotenv()
    try:
        with Router.from_config(args.config, db_path=args.db) as router:
            result = router.call(
                args.task,
                [{"role": "user", "content": args.message}],
                caller=args.caller,
            )
            print(f"model: {result.model_id}")
            print(f"cost_usd: {result.cost_usd:.8f}")
            print(f"tokens: input={result.input_tokens}, output={result.output_tokens}")
            print(f"attempts: {len(result.attempts)}")
            print("response:")
            print(result.text)
        return 0
    except Exception as exc:
        print(f"llmrouter call failed: {exc}", file=sys.stderr)
        for attempt in getattr(exc, "attempts", []):
            detail = f": {attempt.error}" if attempt.error else ""
            print(
                f"  attempt {attempt.attempt_no} on {attempt.model_id}: "
                f"{attempt.status}{detail}",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
