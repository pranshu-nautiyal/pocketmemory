"""Persona loading — the character-core track of the dual-track memory system."""
from __future__ import annotations

import json
from pathlib import Path

PERSONA_DIR = Path(__file__).parent / "personas"


def available_personas() -> list[str]:
    return sorted(p.stem for p in PERSONA_DIR.glob("*.json"))


def load_persona(persona_id: str) -> dict:
    path = PERSONA_DIR / f"{persona_id}.json"
    with path.open() as f:
        return json.load(f)


def system_prompt(persona: dict) -> str:
    """Always-in-context character-core block."""
    knows = "\n".join(f"- {w}" for w in persona.get("world_knowledge", []))
    return (
        f"You are {persona['name']}.\n\n"
        f"STYLE:\n{persona['style_block']}\n\n"
        f"WHAT YOU KNOW:\n{knows}\n\n"
        f"BOUNDARIES:\n{persona['boundaries']}"
    )
