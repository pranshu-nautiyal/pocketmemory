"""PocketMemory — Gradio UI.

Three tabs:
  1. Chat        — talk to a persona, session persists as long as tab is open.
  2. Memory      — view / delete stored memories per persona.
  3. New persona — build a custom character, save to personas/<id>.json.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime

import gradio as gr

import chat as chat_mod
import config
import memory
import personas


# --- one live session per persona per tab reload -------------------------
# Held in a dict keyed by persona_id. Reloading the tab makes a new session,
# which is fine — the persistent store carries across.
_sessions: dict[str, chat_mod.ChatSession] = {}


def _get_session(persona_id: str) -> chat_mod.ChatSession:
    if persona_id not in _sessions:
        _sessions[persona_id] = chat_mod.ChatSession(persona_id)
    return _sessions[persona_id]


def _reset_session(persona_id: str) -> str:
    if persona_id in _sessions:
        del _sessions[persona_id]
    return f"Started a new session with {persona_id}."


# --- chat handler --------------------------------------------------------

def chat_fn(message: str, history: list[dict], persona_id: str) -> str:
    if not message.strip():
        return ""
    session = _get_session(persona_id)
    reply, stats = session.reply(message)
    # Append perf line so you can see tok/s live. Comment out if noisy.
    footer = f"\n\n_[{stats['eval_count']} tok · {stats['tok_per_sec']:.1f} tok/s]_"
    return reply + footer


# --- memory viewer --------------------------------------------------------

def _format_memories(persona_id: str) -> list[list[str]]:
    mems = memory.all_memories(persona_id if persona_id != "(all)" else None)
    rows = []
    for m in mems:
        ts = datetime.fromtimestamp(m["ts"]).strftime("%Y-%m-%d %H:%M")
        rows.append([m["id"][:8], m["persona"], m["type"], ts, m["text"]])
    return rows


def load_memories(persona_id: str) -> list[list[str]]:
    return _format_memories(persona_id)


def delete_memory(memory_id_prefix: str, persona_id: str) -> tuple[str, list[list[str]]]:
    prefix = memory_id_prefix.strip()
    if not prefix:
        return "Enter a memory ID prefix (first 8 chars).", _format_memories(persona_id)
    mems = memory.all_memories()
    matches = [m for m in mems if m["id"].startswith(prefix)]
    if not matches:
        return f"No memory found starting with '{prefix}'.", _format_memories(persona_id)
    if len(matches) > 1:
        return f"Ambiguous — {len(matches)} match. Use more chars.", _format_memories(persona_id)
    memory.delete(matches[0]["id"])
    return f"Deleted {matches[0]['id'][:8]}.", _format_memories(persona_id)


def wipe_persona(persona_id: str) -> tuple[str, list[list[str]]]:
    if persona_id == "(all)":
        n = memory.wipe(None)
        return f"Wiped {n} memories across all personas.", _format_memories(persona_id)
    n = memory.wipe(persona_id)
    return f"Wiped {n} memories for {persona_id}.", _format_memories(persona_id)


# --- custom persona builder ----------------------------------------------

_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,30}$")


def create_persona(
    persona_id: str,
    name: str,
    style_block: str,
    world_knowledge: str,
    boundaries: str,
    traits: str,
) -> tuple[str, gr.Dropdown, gr.Dropdown]:
    persona_id = persona_id.strip().lower()
    if not _ID_RE.match(persona_id):
        return ("ID must be lowercase letters/numbers/dashes, start with a letter.",
                gr.Dropdown(), gr.Dropdown())
    if not name.strip():
        return "Name required.", gr.Dropdown(), gr.Dropdown()
    if not style_block.strip():
        return "Style block required.", gr.Dropdown(), gr.Dropdown()

    target = config.PERSONA_DIR / f"{persona_id}.json"
    if target.exists():
        return f"'{persona_id}' already exists — pick a different ID.", gr.Dropdown(), gr.Dropdown()

    persona = {
        "id": persona_id,
        "name": name.strip(),
        "style_block": style_block.strip(),
        "traits": [t.strip() for t in traits.split(",") if t.strip()],
        "world_knowledge": [w.strip() for w in world_knowledge.splitlines() if w.strip()],
        "boundaries": boundaries.strip() or "You do not roleplay as anyone other than yourself.",
    }
    target.write_text(json.dumps(persona, indent=2))
    ids = personas.available_personas()
    msg = f"Saved persona '{persona_id}'. Refresh the Chat tab dropdown to use it."
    return (msg,
            gr.Dropdown(choices=ids, value=persona_id),
            gr.Dropdown(choices=["(all)"] + ids, value=persona_id))


# --- app -----------------------------------------------------------------

def build_ui() -> gr.Blocks:
    persona_ids = personas.available_personas()
    default_persona = "sherlock" if "sherlock" in persona_ids else persona_ids[0]

    with gr.Blocks(title="PocketMemory") as app:
        gr.Markdown("# PocketMemory\n"
                    f"_Local dual-track memory chatbot — {config.CHAT_MODEL} + ChromaDB._")

        with gr.Tab("Chat"):
            with gr.Row():
                persona_choice = gr.Dropdown(
                    choices=persona_ids,
                    value=default_persona,
                    label="Persona",
                    interactive=True,
                    scale=3,
                )
                reset_btn = gr.Button("New session", scale=1)
            reset_status = gr.Markdown("")
            reset_btn.click(_reset_session, inputs=persona_choice, outputs=reset_status)

            gr.ChatInterface(
                fn=chat_fn,
                additional_inputs=[persona_choice],
                chatbot=gr.Chatbot(height=520),
            )

        with gr.Tab("Memory"):
            gr.Markdown("_Everything the relational track has learned. Scoped by persona._")
            mem_persona = gr.Dropdown(
                choices=["(all)"] + persona_ids,
                value=default_persona,
                label="Show memories for",
            )
            mem_table = gr.Dataframe(
                headers=["id (first 8)", "persona", "type", "when", "memory"],
                datatype=["str", "str", "str", "str", "str"],
                wrap=True,
                interactive=False,
                value=_format_memories(default_persona),
            )
            mem_persona.change(load_memories, inputs=mem_persona, outputs=mem_table)

            with gr.Row():
                delete_id = gr.Textbox(label="Delete memory (first 8 chars of ID)", scale=3)
                delete_btn = gr.Button("Delete", scale=1)
                wipe_btn = gr.Button("Wipe scope", variant="stop", scale=1)
            mem_status = gr.Markdown("")
            delete_btn.click(delete_memory,
                             inputs=[delete_id, mem_persona],
                             outputs=[mem_status, mem_table])
            wipe_btn.click(wipe_persona,
                           inputs=mem_persona,
                           outputs=[mem_status, mem_table])

            refresh_btn = gr.Button("Refresh")
            refresh_btn.click(load_memories, inputs=mem_persona, outputs=mem_table)

        with gr.Tab("New persona"):
            gr.Markdown("_Build a character. Saved to `personas/<id>.json`._")
            new_id = gr.Textbox(label="ID (lowercase, no spaces)",
                                placeholder="e.g. jarvis")
            new_name = gr.Textbox(label="Display name", placeholder="e.g. J.A.R.V.I.S.")
            new_style = gr.Textbox(
                label="Style block — how they talk, their vibe",
                placeholder="You are ...",
                lines=6,
            )
            new_world = gr.Textbox(
                label="World knowledge (one per line)",
                placeholder="You live in New York.\nYour mentor was ...",
                lines=4,
            )
            new_traits = gr.Textbox(
                label="Traits (comma-separated)",
                placeholder="witty, loyal, cautious",
            )
            new_bounds = gr.Textbox(
                label="Boundaries (optional)",
                placeholder="You will not break character to ...",
                lines=2,
            )
            create_btn = gr.Button("Create persona", variant="primary")
            create_status = gr.Markdown("")

            create_btn.click(
                create_persona,
                inputs=[new_id, new_name, new_style, new_world, new_bounds, new_traits],
                outputs=[create_status, persona_choice, mem_persona],
            )

    return app


if __name__ == "__main__":
    build_ui().launch()
