"""End-to-end smoke test. Exercises persona load, retrieval, extraction, store.

Runs one 'session A' where we tell Sherlock about ourselves, then simulates a
restart by making a fresh session and asking a callback question.
"""
from __future__ import annotations

import time

import chat as chat_mod
import memory
import personas

PERSONA = "sherlock"


def banner(msg: str) -> None:
    print(f"\n=== {msg} ===")


def main() -> int:
    banner("personas available")
    print(personas.available_personas())

    banner("session A — introduce yourself")
    a = chat_mod.ChatSession(PERSONA)
    for user in [
        "Hello. My name is Pranshu and I'm a high school student researching AI memory.",
        "I hate coffee — it makes me jittery. I drink black tea instead.",
    ]:
        print(f"\nYou: {user}")
        t0 = time.time()
        reply, stats = a.reply(user)
        print(f"{a.persona['name']}: {reply}")
        print(f"[{stats['eval_count']} tok · {stats['tok_per_sec']:.1f} tok/s]")

    print("\nwaiting for background extraction threads to finish...")
    a.flush()
    print("last extraction stats:", a.last_extraction_stats())

    banner("stored memories after session A")
    for m in memory.all_memories(PERSONA):
        print(f"  [{m['type']}] {m['text']}")

    banner("session B — fresh session, callback question")
    b = chat_mod.ChatSession(PERSONA)  # new session_id, no in-memory history
    for user in [
        "Good to see you again. Do you remember what I do?",
        "Care to offer me a beverage?",
    ]:
        print(f"\nYou: {user}")
        reply, stats = b.reply(user)
        print(f"{b.persona['name']}: {reply}")
        print(f"[{stats['eval_count']} tok · {stats['tok_per_sec']:.1f} tok/s]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
