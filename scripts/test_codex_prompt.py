#!/usr/bin/env python3
"""Send one prompt through OpenEvolve's native Codex provider.

Usage:
    uv run python scripts/test_codex_prompt.py
    uv run python scripts/test_codex_prompt.py "Explain this result in one sentence."
    uv run python scripts/test_codex_prompt.py --model gpt-5.6-luna --retries 0
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from openevolve.config import Config
from openevolve.llm.codex import CodexLLM


DEFAULT_PROMPT = "Reply with exactly: Codex provider works"
DEFAULT_MODEL = "gpt-5.6-luna"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one prompt through OpenEvolve's native Codex provider."
    )
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Codex model name")
    parser.add_argument("--auth-path", default=None, help="Path to Codex OAuth credentials")
    parser.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    parser.add_argument("--retries", type=int, default=0, help="Retries after request failures")
    return parser.parse_args()


async def run_prompt(args: argparse.Namespace) -> str:
    config = Config.from_dict(
        {
            "llm": {
                "provider": "codex",
                "codex_auth_path": args.auth_path,
                "models": [
                    {
                        "name": args.model,
                        "timeout": args.timeout,
                        "retries": args.retries,
                    }
                ],
            }
        }
    )
    llm = CodexLLM(config.llm.models[0])
    return await llm.generate(args.prompt)


def main() -> int:
    args = parse_args()
    try:
        print(asyncio.run(run_prompt(args)))
    except Exception as exc:
        print(f"Codex prompt test failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
