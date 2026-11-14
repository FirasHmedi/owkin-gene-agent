"""Terminal UI.

    python cli.py                      interactive chat
    python cli.py "your question"      one question, then exit
    python cli.py --trace "..."        also print the data lookups behind the answer
"""

from __future__ import annotations

import argparse
import sys

from gene_agent.agent import Agent, Reply
from gene_agent.errors import friendly_error


def show(reply: Reply, trace: bool) -> None:
    if trace:
        for call in reply.tool_calls:
            status = "error" if call.is_error else "ok"
            origin = ", added by the agent" if call.added_by_agent else ""
            print(f"  [{call.name} {call.input} -> {status}{origin}] {call.output}", file=sys.stderr)
    print(reply.text.replace("**", ""))


def progress(done: int, total: int) -> None:
    print(f"\rDownloading the model (first run only): {100 * done // total}%", end="", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask questions about the cancer gene dataset.")
    parser.add_argument("question", nargs="?", help="Ask one question and exit.")
    parser.add_argument("--trace", action="store_true", help="Print the data lookups behind each answer.")
    args = parser.parse_args()

    agent = Agent()
    try:
        agent.ensure_model(progress)
    except Exception as exc:
        print(friendly_error(exc), file=sys.stderr)
        return 1

    if args.question:
        try:
            show(agent.ask(args.question), args.trace)
        except Exception as exc:
            print(friendly_error(exc), file=sys.stderr)
            return 1
        return 0

    print(f"Gene data assistant (local model: {agent.model}). Ask a question, or type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question.lower() in {"exit", "quit"}:
            return 0
        if not question:
            continue
        try:
            reply = agent.ask(question)
        except Exception as exc:
            print(friendly_error(exc), file=sys.stderr)
            continue
        print("\nAssistant: ", end="")
        show(reply, args.trace)


if __name__ == "__main__":
    sys.exit(main())
