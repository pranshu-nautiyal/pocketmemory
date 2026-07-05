"""Relational-track memory: extract, embed, store, retrieve.

This is the miniature of the DualTrack relational track. The character-core
lives in personas.py; here we handle everything that changes about the *user*.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Iterable

import chromadb
import ollama

import config

_client = chromadb.PersistentClient(path=str(config.MEMORY_DIR))
_collection = _client.get_or_create_collection(
    name="relational",
    metadata={"hnsw:space": "cosine"},
)


def embed(text: str) -> list[float]:
    resp = ollama.embeddings(model=config.EMBED_MODEL, prompt=text)
    return resp["embedding"]


# --- Router (heuristic v1) ------------------------------------------------

_TRIGGERS = (
    "i am", "i'm", "im ",
    "my ", "mine ",
    "i like", "i love", "i hate", "i prefer", "i enjoy",
    "i work", "i live", "i study", "i grew up",
    "i feel", "i felt", "i think",
    "yesterday", "last week", "last year", "remember when",
    "you promised", "you said",
)


def should_extract(user_message: str) -> bool:
    """Cheap filter — skip pure chitchat before spending an LLM call."""
    words = user_message.split()
    if len(words) < 4:
        return False
    lower = user_message.lower()
    return any(t in lower for t in _TRIGGERS)


# --- Extraction (LLM v1) --------------------------------------------------

_EXTRACT_PROMPT = """You are a memory extractor for a character-driven chatbot.
Given the user's latest message and a few turns of context, extract two kinds of memory:

1. USER_FACTS: verifiable, durable facts about the user (name, job, location, tastes, relationships).
2. RELATIONAL_EVENTS: how the user feels toward the character, promises made either way,
   unresolved threads, callbacks the character should remember next session.

Rules:
- Only extract things worth remembering across sessions. Skip small talk.
- One item per line, terse (under 15 words).
- If nothing meets the bar, output exactly: NONE

Output format:
USER_FACTS:
- <fact>
RELATIONAL_EVENTS:
- <event>

Recent context:
{context}

Latest user message:
{user_message}
"""


def _parse_extraction(raw: str) -> list[dict]:
    """Turn the LLM's structured output into a list of (text, type) dicts."""
    if "NONE" in raw and "USER_FACTS" not in raw:
        return []
    memories: list[dict] = []
    current_type: str | None = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("USER_FACTS"):
            current_type = "fact"
            continue
        if line.upper().startswith("RELATIONAL_EVENTS"):
            current_type = "event"
            continue
        m = re.match(r"^[-*•]\s*(.+)", line)
        if m and current_type:
            text = m.group(1).strip()
            if text and text.upper() != "NONE":
                memories.append({"text": text, "type": current_type})
    return memories


def extract(user_message: str, context: str) -> list[dict]:
    prompt = _EXTRACT_PROMPT.format(user_message=user_message, context=context or "(none)")
    resp = ollama.generate(
        model=config.EXTRACT_MODEL,
        prompt=prompt,
        options={"temperature": 0.1},
    )
    return _parse_extraction(resp["response"])


# --- Store / retrieve ----------------------------------------------------

def store(memories: Iterable[dict], persona_id: str, session_id: str) -> int:
    docs, embs, metas, ids = [], [], [], []
    for m in memories:
        docs.append(m["text"])
        embs.append(embed(m["text"]))
        metas.append({
            "type": m["type"],
            "persona": persona_id,
            "session": session_id,
            "ts": time.time(),
        })
        ids.append(str(uuid.uuid4()))
    if docs:
        _collection.add(documents=docs, embeddings=embs, metadatas=metas, ids=ids)
    return len(docs)


def retrieve(query: str, persona_id: str, k: int = config.TOP_K_MEMORIES) -> list[str]:
    """Retrieve top-k memories scoped to this persona relationship."""
    if _collection.count() == 0:
        return []
    results = _collection.query(
        query_embeddings=[embed(query)],
        n_results=min(k, _collection.count()),
        where={"persona": persona_id},
    )
    docs = results.get("documents") or [[]]
    return docs[0] if docs else []


def all_memories(persona_id: str | None = None) -> list[dict]:
    """Dump every stored memory (for the viewer UI)."""
    where = {"persona": persona_id} if persona_id else None
    results = _collection.get(where=where)
    out = []
    for i, doc in enumerate(results.get("documents", [])):
        meta = results["metadatas"][i]
        out.append({
            "id": results["ids"][i],
            "text": doc,
            "type": meta.get("type", "?"),
            "persona": meta.get("persona", "?"),
            "session": meta.get("session", "?"),
            "ts": meta.get("ts", 0),
        })
    return sorted(out, key=lambda m: m["ts"], reverse=True)


def delete(memory_id: str) -> None:
    _collection.delete(ids=[memory_id])


def wipe(persona_id: str | None = None) -> int:
    """Delete all memories for one persona, or all memories."""
    memories = all_memories(persona_id)
    if not memories:
        return 0
    _collection.delete(ids=[m["id"] for m in memories])
    return len(memories)
