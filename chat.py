"""Dual-track prompt assembly + response generation.

Character-core (persona) is always injected. Relational track (user memories)
is retrieved top-k per turn and shown to the model in a scoped slot so the two
tracks don't blur.
"""
from __future__ import annotations

import threading
import time
import uuid

import ollama

import config
import memory
import personas


def _new_session_id() -> str:
    return f"sess-{int(time.time())}-{uuid.uuid4().hex[:6]}"


class ChatSession:
    """One live conversation with one persona.

    Owns:
      - the persona (character-core)
      - a rolling short history (working memory)
      - a persistent session_id used for tagging extracted memories
    """

    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        self.persona = personas.load_persona(persona_id)
        self.session_id = _new_session_id()
        self.history: list[dict] = []
        self._last_extraction_stats: dict = {"stored": 0, "at": None}
        self._pending: list[threading.Thread] = []

    # --- prompt assembly ---------------------------------------------------

    def _assemble_messages(self, user_message: str) -> tuple[list[dict], int]:
        system = personas.system_prompt(self.persona)

        mems = memory.retrieve(user_message, persona_id=self.persona_id)
        if mems:
            mem_block = "\n\nWHAT YOU REMEMBER ABOUT THIS PERSON (from past sessions):\n" + \
                        "\n".join(f"- {m}" for m in mems)
            system = system + mem_block

        recent = self.history[-config.HISTORY_TURNS * 2:]
        messages = [{"role": "system", "content": system}, *recent,
                    {"role": "user", "content": user_message}]
        return messages, len(mems)

    # --- generation --------------------------------------------------------

    def reply(self, user_message: str) -> tuple[str, dict]:
        messages, retrieved = self._assemble_messages(user_message)
        t0 = time.time()
        resp = ollama.chat(model=config.CHAT_MODEL, messages=messages)
        elapsed = time.time() - t0

        reply_text = resp["message"]["content"]
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": reply_text})

        eval_count = resp.get("eval_count", 0)
        tok_per_sec = eval_count / elapsed if elapsed > 0 else 0.0

        # Fire-and-forget extraction so the user isn't blocked by it.
        if memory.should_extract(user_message):
            t = threading.Thread(
                target=self._extract_and_store,
                args=(user_message,),
                daemon=True,
            )
            t.start()
            self._pending.append(t)

        stats = {
            "tok_per_sec": tok_per_sec,
            "eval_count": eval_count,
            "seconds": elapsed,
            "retrieved": retrieved,
        }
        return reply_text, stats

    def _extract_and_store(self, user_message: str) -> None:
        # Small window of context — just enough for the extractor.
        ctx_pairs = self.history[-6:]
        context = "\n".join(f"{m['role']}: {m['content']}" for m in ctx_pairs)
        try:
            mems = memory.extract(user_message, context)
            stored = memory.store(mems, persona_id=self.persona_id,
                                  session_id=self.session_id)
            self._last_extraction_stats = {"stored": stored, "at": time.time(),
                                           "candidates": len(mems)}
        except Exception as e:  # never crash the chat on a bad extraction
            self._last_extraction_stats = {"stored": 0, "at": time.time(), "error": str(e)}

    def flush(self, timeout: float | None = None) -> None:
        """Wait for pending extractions to finish. Call before exiting a script."""
        for t in self._pending:
            t.join(timeout)
        self._pending = [t for t in self._pending if t.is_alive()]

    # --- introspection -----------------------------------------------------

    def last_extraction_stats(self) -> dict:
        return dict(self._last_extraction_stats)
