"""Day-1 baseline CLI. Talk to a persona, no memory. Sanity-checks the stack."""
from __future__ import annotations

import argparse
import sys
import time

import ollama

import config
import personas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="sherlock",
                        choices=personas.available_personas())
    parser.add_argument("--model", default=config.CHAT_MODEL)
    args = parser.parse_args()

    persona = personas.load_persona(args.persona)
    system = personas.system_prompt(persona)
    messages = [{"role": "system", "content": system}]

    print(f"[baseline] persona={args.persona} model={args.model}")
    print(f"[baseline] type 'quit' to exit\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user.lower() in {"quit", "exit"}:
            return 0
        if not user:
            continue

        messages.append({"role": "user", "content": user})
        t0 = time.time()
        resp = ollama.chat(model=args.model, messages=messages)
        elapsed = time.time() - t0
        reply = resp["message"]["content"]
        eval_count = resp.get("eval_count", 0)
        tok_s = eval_count / elapsed if elapsed > 0 else 0.0

        messages.append({"role": "assistant", "content": reply})
        print(f"\n{persona['name']}: {reply}")
        print(f"[{eval_count} tok in {elapsed:.1f}s = {tok_s:.1f} tok/s]\n")


if __name__ == "__main__":
    sys.exit(main())
