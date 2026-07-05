# PocketMemory

**A local, no-API-key, dual-track memory chatbot.** Talk to a character. It
remembers *you* across sessions — not just facts, but the shape of the
relationship.

Built as a Week-1 warmup for a research project on relational memory in
conversational agents. Everything runs on your laptop.

---

## Why

Current AI companion platforms (Character.AI, Replika, Talkie) forget who
you are the moment a session ends. Existing memory research (Mem0,
Letta / MemGPT, Zep) optimizes for *factual* recall — "what did the user say
their job was" — and skips the relational half: how the user felt toward the
character, what they promised, what was unresolved.

PocketMemory is a miniature of the **DualTrack** architecture I'm scoping for
a paper: split memory into two tracks and inject both separately at generation
time.

- **Character core** (always-in-context): who the character is, how they
  speak, what they know about their world. Loaded from a persona JSON.
- **Relational track** (retrieved per turn): what the character has learned
  about *this specific user* over time. Extracted after every meaningful turn,
  embedded, stored in ChromaDB, retrieved top-k on the next turn.

The two tracks don't blur. The persona is stable across users; the relational
track is scoped per persona-user pair.

---

## Architecture

```
                       User turn
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   Router (regex)   Retrieve top-k     Load persona JSON
        │           (Chroma + Nomic)          │
        ▼                  │                  │
  worth extracting?        │                  │
        │                  ▼                  ▼
        │        ┌────────────────────────────────┐
        │        │ Prompt assembly                │
        │        │  SYSTEM: persona style         │
        │        │  WHAT YOU REMEMBER: top-k mems │
        │        │  last N turns                  │
        │        │  USER: this turn               │
        │        └───────────────┬────────────────┘
        │                        │
        │                        ▼
        │                Ollama /chat
        │                        │
        │                        ▼
        │                 Reply to user
        │
        ▼
  Extract in background
  ├─ USER_FACTS (durable)
  └─ RELATIONAL_EVENTS (episodic)
        │
        ▼
   Embed + store (Chroma)
```

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Model runtime | [Ollama](https://ollama.com) | Uses Metal on Apple Silicon automatically |
| Base LLM | `dolphin-llama3:8b` | Uncensored — refuses less, so character stays in character |
| Embeddings | `nomic-embed-text` | Runs on the same Ollama server |
| Vector store | [ChromaDB](https://www.trychroma.com/) | Local, sqlite-backed, zero-config |
| UI | [Gradio](https://gradio.app) | One file, browser, no HTML |
| Language | Python 3.11+ | — |

Everything runs locally. No API keys, no bills, no rate limits.

---

## Quickstart

```bash
# 1. Install Ollama and pull the models
brew install ollama
ollama serve &                          # leave running
ollama pull dolphin-llama3:8b           # ~4.7 GB
ollama pull nomic-embed-text            # ~274 MB

# 2. Set up the project
git clone <this repo> && cd pocketmemory
python3 -m venv .venv && source .venv/bin/activate
pip install ollama chromadb gradio

# 3. Talk
python app.py                           # browser UI on :7860
# or, for a CLI:
python baseline_chat.py --persona sherlock
```

---

## Using it

The UI has three tabs:

- **Chat** — pick a persona from the dropdown, start talking. Perf footer
  under each reply shows tokens/sec so you know when the model is loaded.
- **Memory** — see everything the relational track has stored, scoped by
  persona. Delete individual memories, or wipe a whole persona's history.
- **New persona** — build your own character (ID, style block, world
  knowledge, boundaries). Saved to `personas/<id>.json` and immediately
  usable.

Memory persists across restarts by default (Chroma writes to
`memory_store/`). To reset everything: delete that folder.

---

## Repo layout

```
pocketmemory/
├─ app.py               # Gradio UI (chat + memory viewer + persona builder)
├─ baseline_chat.py     # Day-1 CLI baseline, no memory
├─ chat.py              # ChatSession — dual-track prompt assembly, generation
├─ memory.py            # ChromaDB store, embedder, extraction prompt, router
├─ personas.py          # Persona loading + system-prompt assembly
├─ config.py            # Model IDs, paths, top-k, history window
├─ personas/            # sherlock.json, bartender.json, yoda.json
├─ memory_store/        # ChromaDB persistent files (gitignored)
├─ smoketest.py         # End-to-end multi-session verification
└─ WRITEUP.md           # Notes on what worked, what broke, what's next
```

---

## What it can't do yet

Documented after real testing. See [WRITEUP.md](WRITEUP.md) for full detail.

- **Persona breaks on refusals.** Even on an uncensored base, the model
  sometimes says *"as an AI..."* mid-roleplay. Character-core injection
  helps but doesn't fully suppress it.
- **Extractor confabulation.** The extractor sees a few turns of context
  and sometimes lifts facts from the *character's* replies instead of the
  user's. Needs role-scoped grounding.
- **No temporal reasoning.** Memories carry a timestamp but no decay, no
  contradiction handling. If you say "I love X" then "I hate X" a week
  later, both live in the store with equal weight.
- **Top-k retrieval is dumb.** Pure cosine similarity, no diversification,
  no re-ranking, no persona-aware weighting.

Each of these is a paper-scale problem, not a v0.1 bug. That's the point of
building the warmup.

---

## What's next

This repo is the prototype for the DualTrack system in a longer research
project on **relational memory in conversational agents**, targeting
NeurIPS Evaluations & Datasets. The paper's contributions build on top of
this warmup:

- A benchmark for relational memory (not just factual recall)
- LLM-as-judge protocol for scoring persona stability + callback fidelity
- Ablations of extraction, retrieval, and injection independently
- Failure taxonomy grounded in real chat traces

The three failure modes above are seeds for that taxonomy.

---

## License

MIT. Personas are fictional. Do what you want with the code.
