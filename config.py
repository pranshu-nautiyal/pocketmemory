"""Central config. Swap models here; the rest of the code reads from this."""
from __future__ import annotations

from pathlib import Path

CHAT_MODEL = "dolphin-llama3:8b"
EXTRACT_MODEL = "dolphin-llama3:8b"
EMBED_MODEL = "nomic-embed-text"

MEMORY_DIR = Path(__file__).parent / "memory_store"
PERSONA_DIR = Path(__file__).parent / "personas"

TOP_K_MEMORIES = 5
HISTORY_TURNS = 8
