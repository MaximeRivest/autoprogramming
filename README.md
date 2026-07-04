# AutoProgramming

> **⭐ Aspirational — this README is the golden objective.**
> Nothing described here exists yet. This document is written as if the tool were finished, and it defines what we are building in this repo. When the real behavior and this document disagree, this document wins until we deliberately amend it.

Define your inputs and outputs, and let AutoProgramming find the best implementation.

**Why this is different.** Prompt optimizers (DSPy, GEPA, TextGrad) search over prompts inside a framework you must ship with. In AutoProgramming, a candidate implementation is a **plain `.py` file** — so the search space is anything Python can express (an LLM call, a regex table, scikit-learn, a local transformer, a pipeline of all four), the optimizer is a **coding agent** that reflects, edits, and evaluates, and the output is a **portable Python package** with zero runtime dependence on the optimizer. Optimization happens at dev time; what you ship is just code.

## Define a program

Types are subclasses of builtins. Docstrings become descriptions the agent uses.

```py
import autoprogramming as ap

class French(str):
    """Natural, idiomatic French. Formal register unless the input is casual."""

@ap.program
def translate(english: str) -> French:
    """Translate English text to French."""
```

Multiple outputs are a tuple of distinct types (two outputs of the same type is a schema error — output names come from their types):

```py
class Answer(str):
    """Direct answer to the question, one sentence."""

class Confidence(float):
    """Calibrated probability that the answer is correct, 0.0–1.0."""

@ap.program
def qa(question: str) -> tuple[Answer, Confidence]:
    """Answer a factual question with a calibrated confidence."""
```

## Optimize it

```py
translate.optimize(data=pairs_df, budget=ap.Budget(dollars=20))
translate.save("translate_ap")

translate("Hello, how are you?")
# => French('Bonjour, comment allez-vous ?')
```

`.optimize()` launches a coding agent that iterates on complete implementations — instructions, model choice, SDK, parsing, even the algorithmic approach — using reflective evolution (inspired by [GEPA](https://github.com/gepa-ai/gepa)), under a strict data-splitting discipline described below.

**Budget is explicit and has units.** `ap.Budget(dollars=20)`, `ap.Budget(eval_calls=2000)`, or `ap.Budget(minutes=30)` — combinable; optimization stops when the first limit is hit. Evaluation cost counts against the budget (LLM candidates cost money to *score*, not just to mutate). There is no default; you must say what you're willing to spend.

## Data discipline

This is the part most optimizers get wrong, so it is load-bearing here.

On `optimize()`, the data is split **once** into three sets:

| Split | Who touches it | Purpose |
|-------|---------------|---------|
| `train` | Agent, freely | Reflection: run candidates, read traces, study failures, mutate |
| `val` | Scoring harness only | Selection: which candidates survive (Pareto bookkeeping) |
| `test` | Nobody, until the end | The final report card, evaluated exactly once |

The rules the harness enforces (not the agent's good manners):

- **The agent reflects on `train` failures only.** `prg.run()` and per-row trace inspection are refused on val and test rows. The agent never sees *why* a val row scored low — only the aggregate — so it cannot edit candidates to fix specific val examples.
- **`val` is for selection, and selection pressure is capped.** Every candidate is scored on the identical val set, and the harness tracks how many candidates have been compared against it. When val has been used to make many selection decisions relative to its size, the harness warns that val scores are losing meaning and refuses to report them as final.
- **`test` is evaluated once, at the end, on the top candidates only.** The number you tell your boss is the test score. If test drops far below val, the run report says so loudly — that's overfitting to val, and the report shows it instead of hiding it.
- **Minimum data is enforced, not suggested.** Below ~30 examples, `optimize()` runs in **bootstrap mode**: it will build and compare baseline candidates but refuses to run fine-grained mutation loops, because a 0.92-vs-0.86 difference on 5 val rows is one row of noise. It tells you this and offers to generate synthetic examples for you to validate.
- **Score differences come with uncertainty.** Candidate comparisons on val report a bootstrap confidence interval; "improved" means the interval excludes zero, not that the point estimate went up.
- **Memorization is checked.** Any candidate whose train score vastly exceeds its val score, or that contains verbatim training outputs (lookup tables over train inputs), is flagged as a memorizer and excluded from selection. A regex candidate may legitimately win — but only on data it never saw.
- **Stochastic candidates are scored honestly.** Candidates that call an LLM are evaluated with `n` repeats per row (default 3); the score is the mean, and the report includes the variance. A candidate whose win disappears across repeats didn't win.

## What the agent actually does

`optimize()` creates a workspace that is itself a valid, installable Python package:

```
translate_ap/
├── pyproject.toml          # name, deps of the ACTIVE candidate, entry point
├── __init__.py             # exports `translate`, dispatches to active candidate
├── schema.py               # the @ap.program definition (immutable)
├── metric.py               # evaluation metric (agent-proposed, USER-APPROVED)
├── active.json             # which candidate is live + pinned eval scores
├── data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv            # harness-controlled; agent cannot eval against it
├── candidates/
│   ├── candidate_0.py      # seed
│   ├── candidate_1.py      # mutated from 0
│   └── candidate_2.py      # mutated from 1
├── artifacts/              # model weights, pickles, lookup tables (git-lfs friendly)
└── scores.json             # per-candidate, per-row val scores + repeat variance
```

### The metric is proposed, then approved — never silently trusted

The entire search optimizes whatever `metric.py` says, so a wrong metric produces a confidently-scored wrong program. The agent therefore **proposes** a metric, demonstrates it on real examples, and requires sign-off:

```
Agent: I propose scoring translations with chrF (character n-gram F-score) — it
       rewards meaning-preserving variation better than edit distance. Watch it
       score three examples:

       expected: "Où est la gare ?"
       predicted: "Où se trouve la gare ?"     → 0.81  (synonymous phrasing, mild penalty)
       predicted: "Où est la gare ?"           → 1.00  (exact)
       predicted: "La gare est fermée."        → 0.34  (wrong meaning, big penalty)

       Does 0.81 for the synonymous phrasing match your intuition of "good"?
       If synonyms should score ~1.0, I'll use embedding similarity instead.

User:  synonyms should be fine → use embeddings.
```

```py
# metric.py  — approved by user on 2026-07-04
from sentence_transformers import SentenceTransformer, util

_model = None

def metric(predicted: str, expected: str) -> float:
    global _model
    if _model is None:
        _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return float(util.cos_sim(_model.encode(predicted), _model.encode(expected)))
```

For multi-output programs the metric receives and returns per-field scores:

```py
# metric.py for qa: tuple[Answer, Confidence]
def metric(predicted: dict, expected: dict) -> dict:
    return {
        "Answer": answer_similarity(predicted["Answer"], expected["Answer"]),
        "Confidence": 1.0 - abs(predicted["Confidence"] - expected["Confidence"]),
    }
# aggregate weighting is also part of the user sign-off
```

The metric file records who approved it and when. Changing the metric invalidates all existing scores (the harness clears `scores.json` — scores under different metrics are never comparable).

### The agent writes candidates

The seed candidate is a complete module. Note: no work at import time — clients and models load lazily, so importing the package never requires an API key or network:

```py
# candidates/candidate_0.py
from openai import OpenAI
from ..schema import French

_client = None

def predict(english: str) -> French:
    global _client
    if _client is None:
        _client = OpenAI()
    response = _client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "Translate the following English text to French. Reply with only the translation, nothing else."},
            {"role": "user", "content": english},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return French(response.choices[0].message.content.strip())
```

Each candidate declares its own dependencies in a header the packager reads:

```py
# candidates/candidate_0.py
# ap-requires: openai>=1.0
```

The agent evaluates, reflects on **train** failures, then copies and edits the file to create a new candidate:

```py
prg.eval("candidate_0")               # val score: 0.83 ± 0.04 (n=3 repeats)
prg.eval("candidate_0", split="train", per_instance=True)
prg.run("candidate_0", split="train", row=17)   # full trace of a failing train row
# ... reflect, copy candidate_0 → candidate_1, edit the prompt ...
prg.eval("candidate_1")               # val score: 0.91 ± 0.03 — CI excludes 0 vs candidate_0, keep
```

Each candidate is a readable, diffable `.py` file. The agent repeats — reflect on train, mutate, select on val — until the budget is exhausted. Then the harness (not the agent) runs the one-time test evaluation on the top candidates and activates the winner:

```py
prg.finalize()
# test scores (evaluated once):
#   candidate_1: 0.89   (val was 0.91 — healthy gap)
#   candidate_4: 0.84   (val was 0.92 — overfit to val, demoted)
# activated: candidate_1
```

## Agent API

Inside the optimization loop, the agent holds `prg` — the agent-side handle to the same program you call `translate` on the outside. Two names, one object, two trust levels: `prg` can create and score candidates; `translate` can only run the active one.

```py
prg.schema                                   # inspect inputs/outputs & docstrings
prg.eval("candidate_0")                      # score on val (aggregate only, with CI)
prg.eval("candidate_0", split="train", per_instance=True)   # per-row, train only
prg.run("candidate_0", split="train", row=17)  # single traced run — train rows only
prg.frontier()                               # Pareto frontier: best candidate per train row
prg.data.train                               # readable
prg.data.val                                 # scoring only — rows not readable
prg.budget                                   # remaining dollars / eval calls / time
```

Everything else — creating candidates, editing prompts, changing SDKs, rewriting parsers — is file operations on `candidates/*.py`. There is deliberately no `prg.eval(split="test")`: test belongs to `finalize()`.

## Use it like a normal function

The workspace is a normal Python package (underscore name — it's importable):

```py
from translate_ap import translate

translate("Hello, how are you?")
# => French('Bonjour, comment allez-vous ?')
```

`French` is a `str` subclass — works everywhere a string does.

**What `activate()` does, mechanically:** it writes `active.json` with the chosen candidate's name and its pinned test score, and regenerates `pyproject.toml` so the package's dependencies are exactly the active candidate's `ap-requires` lines (plus its artifacts). `__init__.py` reads `active.json` and imports that one candidate. Switching candidates is a one-line diff you can review, commit, and revert.

### Log production traffic

```py
translate.enable_logging()

translate("Where is the train station?")
# => French('Où est la gare ?')
# appends to translate_ap/logs/2026-07-04.jsonl
```

```json
{"inputs": {"english": "Where is the train station?"}, "outputs": {"French": "Où est la gare ?"}, "candidate": "candidate_1", "n_repeat": 1, "timestamp": "2026-07-04T14:22:01Z"}
```

Input keys are parameter names; output keys are output type names (guaranteed unique by the schema rule above). Inputs and outputs live in separate objects, so they can never collide.

### Improve from production — two different things

**Distill** — compress the current program into something cheaper. Logs are perfect for this: the program's own outputs *are* the training target, because the goal is imitation:

```py
translate.distill(model="gpt-4.1-nano", data="logs", output="translate_ft_ap")
```

**Re-optimize** — make the program *better*. Logs alone cannot do this: they record what the current program predicted, and optimizing toward your own outputs reinforces your own errors. Re-optimization requires a correction signal:

```py
# a human (or downstream check) marks/corrects log entries...
translate.review_logs()            # TUI: accept / correct / reject each sampled entry

# ...and only the corrected entries become new training data
translate.optimize(data="logs:reviewed", budget=ap.Budget(dollars=10))
```

`optimize(data="logs")` without review is an error, with a message explaining exactly this.

### Distribute

It's a package with a `pyproject.toml`, so it distributes like one:

```sh
pip install ./translate_ap          # deps of the active candidate install automatically
```

Heavy candidates keep their weights in `artifacts/` (tracked with git-lfs) or declare a download step (`# ap-fetch: huggingface:Helsinki-NLP/opus-mt-en-fr`) that runs on first use. Zipping the directory works too — `artifacts/` goes with it.

## Building a program conversationally

`@ap.program` + `.optimize()` is the library API. The conversational flow is a front-end to the *same* API: when you tell the agent "I want a translator", the conversation's job is to produce the decorator, the data, the budget, and the approved metric — then it calls `optimize()` like anyone else would. Five questions:

**1. What goes in, what comes out?** The schema. "English in, French out" is enough; so is "CSV row in, (label, confidence) out". Becomes the `@ap.program` definition.

**2. Do you have examples?** 30+ pairs unlocks full optimization; 5–10 gets bootstrap mode. If none: "Can I generate synthetic pairs and you validate a sample?"

**3. What does "good" mean?** Exact match? Same meaning? Latency under 100ms? Cost under $0.001/call? Shapes the metric proposal.

**4. Sign off on the metric.** The agent shows the proposed metric scoring real examples (as above) and iterates until the scores match your intuition. *This is the most important question — everything downstream optimizes this number.*

**5. Any constraints?** Data allowed to leave the machine? Max cost per call? Offline? Which packages may be installed? Narrows the search space and configures the sandbox.

Everything else the agent figures out on its own.

## What the agent explores

A candidate is just a `predict` function — the agent can write anything:

**LLM call** — prompt engineering, SDK choice, model selection (see `candidate_0` above).

**Classical ML** — lightweight and fast:
```py
# candidates/candidate_3.py
# ap-requires: scikit-learn>=1.4
import pickle
from ..paths import artifacts    # ap-provided: resolves to the package's artifacts/ dir

_model = None

def predict(english: str) -> French:
    global _model
    if _model is None:
        _model = pickle.loads((artifacts / "candidate_3_model.pkl").read_bytes())
    return French(_model.predict([english])[0])
```

**Rules / regex** — when patterns are enough (and it generalizes past train — the memorization check applies):
```py
# candidates/candidate_4.py
import re
RULES = {r"\bhello\b": "bonjour", r"\bthank you\b": "merci", ...}

def predict(english: str) -> French:
    result = english.lower()
    for pattern, replacement in RULES.items():
        result = re.sub(pattern, replacement, result)
    return French(result)
```

**Local deep learning** — no API cost per call:
```py
# candidates/candidate_5.py
# ap-requires: transformers>=4.40, torch
# ap-fetch: huggingface:Helsinki-NLP/opus-mt-en-fr
from transformers import MarianMTModel, MarianTokenizer

_tok, _model = None, None

def predict(english: str) -> French:
    global _tok, _model
    if _model is None:
        _tok = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
        _model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    tokens = _tok(english, return_tensors="pt", padding=True)
    return French(_tok.decode(_model.generate(**tokens)[0], skip_special_tokens=True))
```

**Decomposed pipeline** — idiom table → local model → LLM refinement for long sentences, each part handling what it's best at.

The agent tries different approaches, scores them on the same val set with the same approved metric, and keeps what wins. A rule-based candidate that scores 0.95 beats an LLM candidate that scores 0.90 — *provided the confidence intervals separate and the memorization check passes*. The agent doesn't care how it works, only that it satisfies the schema and honestly beats the alternatives.

## Execution model & trust

The agent writes and runs code, and installs packages. That happens inside a sandbox, not on your bare machine:

- Candidates execute in an isolated environment with access to `data/`, `artifacts/`, and (if question 5 allowed it) the network.
- `pip install` is restricted to an allowlist derived from question 5; anything outside it prompts you.
- Your data leaves the machine only via APIs you approved in question 5. "Must stay local" means LLM-API candidates are simply not in the search space.
- Every candidate the agent produces is a small, readable `.py` file — review the winner before you `activate` it, the same way you'd review a PR.

### Tools the agent needs

```
filesystem       create the workspace, write/copy/edit candidate files
python_repl      run candidates in the sandbox, inspect traces
llm_apis         call approved providers from candidate code
web_search       domain knowledge, pretrained model discovery
pip_install      allowlisted installs into the sandbox
user_confirm     metric sign-off, synthetic-data validation, allowlist escalations
prg.*            eval / traced runs / frontier / budget (as above)
```

## Exploration strategy

Structured search, so budget isn't wasted on redundant candidates.

**Phase 1: Baseline sweep (~10% of budget).** Establish the floor and a cheap ceiling estimate:
1. A trivial candidate (rules, lookup, simplest heuristic) — candidate_0, which everything must beat.
2. The cheapest viable LLM with a minimal prompt — often already hard to beat.
3. A pretrained model, if one exists for the domain.

**Phase 2: Targeted mutation (~60%).** Improve the leader by studying its **train** failures:
1. `prg.eval(best, split="train", per_instance=True)` — find the worst train rows.
2. `prg.run(best, split="train", row=…)` — read full traces to understand *why*.
3. Copy, make **one** targeted edit (prompt, model, parsing, or a preprocessing step — never several at once, or improvement can't be attributed).
4. `prg.eval(new)` on val. Keep only if the confidence interval vs the parent excludes zero.

**Phase 3: Structural exploration (~20%).** When mutation plateaus (3 consecutive non-improvements), change the approach: LLM → local model, single model → decomposed pipeline, expensive winner → distilled cheap version.

**Phase 4: Finalize (~10%).** The harness re-scores the top candidates on val with extra repeats, then runs the **one-time test evaluation** and activates the winner. Val-vs-test gaps are reported per candidate; a candidate that collapses on test is demoted no matter its val rank.

**Diversity via Pareto.** Don't keep only the global best. `prg.frontier()` tracks which candidate is best on *each train row*; a candidate that's worse on average but uniquely solves hard rows carries information. Mutation ancestors are sampled from the frontier, not just the leader — this is what keeps the search out of local optima.
