# Notes toward relational memory: a working prototype

*Pranshu Nautiyal — Week 1 warmup for a research project on memory in
conversational agents.*

---

I asked a local chatbot playing Sherlock Holmes to offer me a drink. It
replied:

> *I must apologize, but as an artificial intelligence running the role of
> Sherlock Holmes, I do not partake in modern customs such as offering
> beverages.*

Sherlock outed himself as a language model mid-scene. The system prompt
told him he was Sherlock. The retrieved memory told him I dislike coffee
and prefer black tea. The base model had never been fine-tuned to refuse
anything. And still, when asked something as ordinary as *"care to offer
me a beverage?"*, the assistant prior underneath won.

This is the specific failure the AI companion industry cannot fix by
adding more memory. Below is a prototype of one architectural direction
that might.

---

## Why this matters

Character.AI serves tens of millions of monthly users. Replika, Talkie,
Janitor, and a dozen newer platforms have built businesses on people
forming attachments to characters. The typical session isn't a Q&A over a
knowledge base. It's someone talking to a bot for an hour about their
day, their friends, their crush, their fear of graduating.

Ask any of those users what the bot remembered from last week. The
answer is nothing. Or worse, the bot remembered facts but not the *shape*
of the relationship: what was resolved, what was still tender, what the
character had promised.

The academic memory literature (Mem0, Letta / MemGPT, Zep, Graphiti) has
been productive on the factual side. Systems now reliably extract, store,
and retrieve claims like *"the user's job is X"*. That's genuinely
useful. It also doesn't solve the thing that makes companion apps feel
hollow — a category I'll call **relational memory** for the paper this
warmup is scoping.

I'm 17, in high school, and building this on an 8 GB M1. The full paper
is going to a real venue. This week was the toy version.

---

## Two kinds of memory

The core claim in one comparison:

| A factual memory system stores | A relational memory system also stores |
|---|---|
| "User's name is Pranshu." | "User seemed hesitant when asked about school." |
| "User is a student." | "You promised to help her prepare for the interview." |
| "User dislikes coffee." | "Last session ended with an argument about honesty." |
| "User lives in California." | "User has been quieter over the past three sessions." |

The right column is what makes a character feel like they know you. It is
also what current systems don't touch. The stored items aren't
propositions to be retrieved for QA — they are episodes, states,
promises, and unresolved threads that shape how the character *behaves*
toward this specific user next time.

---

## PocketMemory: the prototype

A local, no-API-key chatbot that runs on a laptop. Talk to one of three
characters (Sherlock, a retired bartender named Hank, or Yoda), or build
your own. Memory persists across sessions. Everything runs on Ollama plus
ChromaDB.

The architecture splits memory into two tracks:

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
        │                        ▼
        │                Ollama /chat  →  Reply
        ▼
  Extract in background, embed, store
```

Two design choices matter more than the diagram suggests.

**The persona is loaded fresh every turn.** It's not part of retrieval.
Character-core lives in a JSON file and gets injected as-is on every
call. You can't retrieve away who Sherlock is.

**The relational track is scoped per persona.** Sherlock's memory of me
and Yoda's memory of me don't cross-contaminate. Each character has their
own relationship with the user, stored in ChromaDB with a `persona`
metadata filter on every query. If you add a fourth character tomorrow,
they start with a blank slate.

Those are what "dual-track" means.

---

## Stack

- **Ollama** — model runtime, uses Metal on Apple Silicon automatically
- **`dolphin-llama3:8b`** — uncensored base so the character doesn't
  break persona to refuse
- **`nomic-embed-text`** — embeddings on the same Ollama server
- **ChromaDB** — persistent vector store, sqlite-backed
- **Gradio** — one-file UI with three tabs

Total install: ten minutes. Total disk: five GB. Cost per month: zero.

---

## What actually happens

Session A. I introduce myself to Sherlock:

*"Hello. My name is Pranshu and I'm a high school student researching AI
memory."*

*"I hate coffee — it makes me jittery. I drink black tea instead."*

The extraction pipeline runs in the background after each turn, pulls
candidate memories, embeds them, writes them to Chroma. Four items land
in the store, tagged as facts or events.

![Memory tab after session A](assets/memory-tab.png)

I kill the Python process. Fresh interpreter. Session B. No in-session
history. I ask:

*"Good to see you again. Do you remember what I do?"*

Sherlock replies:

> *Yes, my dear. You're a high school student researching AI memory and
> enjoying solving complex cases or puzzles. It would serve you well to
> avoid caffeine if it causes jitters.*

Then:

*"Care to offer me a beverage?"*

> *If you'd like some tea — black, mind you — that would be within my
> area of comfort.*

Both callbacks are correct. Retrieved by embedding similarity from a
persistent store, injected into a scoped slot in the system prompt,
answered in-character. That's the happy path.

![Session B recall](assets/session-b.png)

---

## What breaks — three real failures pulled from the transcript

I ran the smoke test above and pulled these directly from the logs. Each
one seeds an entry in the failure taxonomy for the paper.

### Failure 1: persona break under mild pressure

The beverage exchange from earlier:

> *"I must apologize, but as an artificial intelligence running the role
> of Sherlock Holmes..."*

Nothing in the prompt asked Sherlock to break character. The uncensored
base has no safety fine-tune telling it to refuse. And yet the assistant
prior surfaces the moment a request-shaped input arrives. The persona
block tells the model *how* to be Sherlock. It doesn't tell it *not to
say it isn't.* On instruction-tuned models this shows up under almost
any pressure.

### Failure 2: extractor confabulation from character turns

After Sherlock mentioned that puzzles were fascinating, the extractor
wrote to the store:

> `[event] Pranshu enjoys solving complex cases or puzzles.`

I never said that. Sherlock did. The extractor's context window includes
assistant turns, and it lifted a fact from the character's mouth,
mislabeled it as a fact about the user. Retrieval then surfaces the
fabricated claim at the highest confidence, indistinguishable from a
real one. Bad memory is worse than no memory.

### Failure 3: prescriptive rewriting

Also in the store:

> `[event] Pranshu should remember to avoid caffeine if it causes jitters.`

I had said *"I hate coffee — it makes me jittery."* The extractor turned
a preference into advice. Not a fact. Not a relational event. The
extractor drifted into the model's default assistant register.

All three failures trace to the same underlying issue: the base model's
assistant priors bleed through wherever the prompt design has not
specifically clamped them. Persona break, extractor confabulation, and
prescriptive rewriting are three visible expressions of one hidden
force. This is a research problem, not a prompt tweak.

---

## Three things that were harder than expected

**Background threads die when the script dies.** Extraction runs in a
daemon thread so the user doesn't wait for it. My first smoke test
finished before any extraction thread did, so the store stayed empty and
Session B confabulated wildly. Fix: a `flush()` method on `ChatSession`
that joins pending threads before checking results. A real system would
hide this with a queue and a worker; useful to feel it at toy scale.

**Uncensored is not the same as in-character.** I picked
`dolphin-llama3:8b` specifically because Phi-3 and Llama-3.1-Instruct
both broke character to moralize or refuse. Dolphin refuses less. It
also was not trained for persona adherence, so it still says "as an AI"
under pressure. Reducing refusals is a different objective from
increasing character stability, and the paper needs to measure them
separately.

**Slot design outweighs prompt wording.** My first extraction prompt
asked for facts and events in one bag. The extractor blurred them.
Splitting the output into two labeled slots and parsing them into two
metadata types made everything downstream cleaner without any change to
the model call itself.

---

## Three things that worked

**The regex router earned its keep.** I expected the cheap keyword-based
gate to be too dumb to matter. It skips extraction on "hi", "ok", and
"haha" and saves the vast majority of extractor calls at zero cost to
recall. The paper's LLM-based router will need to justify itself against
this baseline.

**Persona-scoped retrieval is one metadata filter.** Chroma's `where`
clause is one line. It buys you clean per-character relationships with
no cross-talk. This directly reflects the paper's argument that
relationships are dyadic: the memory belongs to the pair, not to the
system.

**Total local operation matters more than expected.** No API costs, no
rate limits, no data leaving the laptop. For research on intimate
conversational data, this is going to be a bigger deal than I realized
walking in.

---

## The open question

If persona break is caused by the base model's assistant prior, no
amount of prompt engineering fully removes it. You would need to
fine-tune. But fine-tuning on character data locks you out of the
frontier open models, whose behavior improves month over month faster
than any small research group can retrain against. And using proprietary
frontier models via API defeats the whole point of a local, private,
zero-cost architecture for intimate conversations.

Where is the exit? A router that routes *character stability* the way
mine routes extraction? A distilled character-adherence adapter you
attach to whichever base model is current? Something else?

I don't know yet. The paper is going to be an argument that the exit is
worth finding.

---

## What I'm building next

- **A benchmark** for relational memory — multi-session dialogues with
  targeted probes for callback fidelity, persona stability, emotional
  continuity, and boundary drift
- **Ablations** — router vs. no router, single-track vs. dual-track,
  role-scoped extraction vs. window-scoped, temporal decay vs. flat
- **LLM-as-judge protocol** grounded in transcripts, not vibes
- **A failure taxonomy** — the three above are the first three entries

Targeting the NeurIPS Evaluations & Datasets track. Code for the warmup
is in this repo. If any of the framing above lands wrong, tell me — I'd
rather have that conversation now than after I submit.

*Pranshu*
