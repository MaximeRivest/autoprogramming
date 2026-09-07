# AutoProgramming

> **Experimental, implemented library (v0.2).**
> The core API, guarded evaluation harness, package workspace, multi-objective
> scoring, and Pi portfolio backend are implemented and tested. The remaining
> security limitation is explicit: cooperative path isolation is not an
> adversarial OS sandbox; `strict_isolation=True` currently refuses rather than
> silently weakening that guarantee.

Define your inputs and outputs, and let AutoProgramming find the best implementation.

**Why this is different.** Prompt optimizers (DSPy, GEPA, TextGrad) search over prompts inside a framework you must ship with. In AutoProgramming, a candidate implementation is a **plain `.py` file** — so the search space is anything Python can express (an LLM call, a regex table, scikit-learn, a local transformer, a pipeline of all four), and the optimizer is a **coding agent** that reflects, edits, and evaluates. Most frontier points ship as ordinary portable code with no optimizer dependency. A Pi-subscription-backed runtime is also allowed as an explicitly labelled frontier point (`requires Pi + login`), so the user—not an early feasibility heuristic—decides whether that restriction is acceptable.

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

**Budget is explicit and has units.** `ap.Budget(dollars=20)`, `ap.Budget(eval_calls=2000)`, or `ap.Budget(minutes=30)` — combinable; optimization stops when the first limit is hit. Evaluation cost and Pi orchestrator/worker cost count against the dollar budget. There is no default; you must say what you're willing to spend.

## Orchestrated portfolio search with Pi

The Pi model already speaking with the human is the strategy orchestrator, not a candidate author. In a live Pi session AutoProgramming uses `HostOrchestratorBackend` and never starts a second strategy process; this preserves the user's requirements, research, and failure reasoning in one conversation. A trusted Python controller launches only isolated implementation workers and independent auditors. Workers receive a generic function contract, development examples, their assigned mechanism, and their own files—never optimizer context, metric code or weights, scores, other workers, val, or test.

Before planning, the host runs and records current web searches. It then plans across runtime agents, model graphs, single calls, fine-tunes, specialized models, classical ML, and direct code/rules. Mechanisms are hard contracts, but engineering plans are adaptable: workers may change package/model variants within the family, batching, parsing, device placement, and setup. A broken worker, malformed output, noncompliant fallback, or suspicious zero result does not satisfy breadth. The controller investigates, repairs, and requests a materially independent configuration; ambiguity pauses for the human.

Search is breadth-first by policy: every feasible family must produce evaluated faithful evidence or be explicitly excluded by the human, each successful family gets a second independent pass, and only then does the host allocate deeper rounds and cross-tier composition. See [`docs/orchestrated-search.md`](docs/orchestrated-search.md).

Search-time hardware and deployment-time resources are separate contracts:

```py
resources = ap.Resources(
    search=ap.SearchResources(
        max_parallel_agents=4,
        max_dollars_per_agent_call=0.05,  # reserves in-flight budget headroom
        candidate_api_providers=(),
        allow_package_installs=True,
        allow_model_downloads=True,
        # The active Pi model/thinking level is captured automatically.
        # If offered: remote_compute=ap.RemoteCompute(
        #     endpoint="...", transport="ssh", ...)
        # Transport is explicit; AutoProgramming never assumes remote means SSH.
    ),
    runtime=ap.RuntimeResources(network=False),
    data=ap.DataPolicy(external_egress=True),  # permits abstract web-research queries
    confirmed=True,
)

prepared = translate.prepare(
    pairs_df, resources=resources, budget=ap.Budget(dollars=2)
)
prg = ap.attach(prepared.workspace.root)
# The current Pi session proposes/demonstrates the metric suite and obtains sign-off.
print(prg.web_search("latest efficient approaches for <abstract task>"))
print(prg.web_search("2026 open source <task> benchmark models"))
prg.plan_portfolio(web_informed_avenue_specs)
prg.orchestrate_portfolio("breadth", budget=ap.Budget(dollars=20))
state = prg.portfolio_status()
prg.orchestrate_portfolio("deepen", avenue_ids=[...])
prg.orchestrate_portfolio("compose")
report = prg.finalize()
```

Hardware can be detected, but AutoProgramming never interprets a network connection, GPU, or remote address as permission. Remote compute is used only when the user supplies `RemoteCompute`; then heavy worker/evaluation operations are staged there while orchestration stays local. Pi-runtime candidates stay beside the authenticated host because they are network-bound and OAuth credentials are never copied to the target. GPU-heavy avenues use per-target concurrency/VRAM admission (one concurrent GPU job and an 80%-free floor when card VRAM is supplied, both configurable), so contention and OOM trigger waiting/exclusive retry rather than a false approach failure.

Pi subscription access is discovered from Pi's authenticated model registry and stored only as provider/model capability names—OAuth tokens remain in Pi's auth store. Workers default to the exact host model and thinking level; the host may deliberately assign another discovered subscription model. A Pi-backed candidate uses Pi CLI/RPC without raw API keys and its final-report entry states the deployment requirement. Under a dollar budget, parallel calls reserve confirmed headroom. For an explicitly headless run, pass `PiOrchestratorBackend(...)`; live Pi sessions use the same-session host instead.

Metric suites distinguish **acceptance lenses** (user-approved and eligible to choose the final program) from **diagnostic lenses** (orchestrator-managed search feedback). Suite-aware search uses acceptance floors and a Pareto frontier rather than requiring one weighted scalar. The legacy primary remains a report headline for older workspaces.

If bounded implementation diagnosis, same-family repairs, an independent configuration, and resource verification still leave an approach blocked, the controller pauses for human confirmation instead of silently dropping it. The user can fix the capability and request a retry, or explicitly confirm exclusion with `prg.resolve_blocker(...)`; fallback code is never the answer.

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
│   ├── val.csv              # legacy/manual workspaces
│   └── test.csv             # legacy/manual workspaces
# Resource-confirmed Pi runs keep val/test in controller-private storage,
# outside this coding-agent workspace.
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

Every candidate is a **PEP 723 single-file script** — inline metadata that `uv run` understands natively. The seed candidate is a complete module. Note: no work at import time — clients and models load lazily, so importing the package never requires an API key or network:

```py
# candidates/candidate_0.py
# /// script
# requires-python = ">=3.11"
# dependencies = ["openai>=1.0", "translate-ap"]
#
# [tool.uv.sources]
# translate-ap = { path = "..", editable = true }
# ///
from openai import OpenAI
from translate_ap.schema import French

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

if __name__ == "__main__":
    import sys
    print(predict(sys.argv[1]))
```

This buys three things at once:

```sh
uv run candidates/candidate_0.py "Hello, how are you?"
# => Bonjour, comment allez-vous ?
```

- **Every candidate runs standalone.** `uv run` reads the `# /// script` block, builds an ephemeral venv with exactly that candidate's deps, and executes it. Debugging a candidate is running a file — no project setup.
- **Candidates with conflicting dependencies coexist.** candidate_2 can need `transformers==4.40` while candidate_5 needs `4.51`; the eval harness runs each in its own uv-resolved environment. Stable, dependency-keyed drivers let UV reuse environments instead of creating one for every evaluation call. The dependency solver is uv's, not ours.
- **The packager reads the same block.** No parallel metadata format to keep in sync — the active candidate's `dependencies` list *is* the package's dependency list.

Implementation workers use an AutoProgramming-owned UV cache under
`~/.cache/ap-work/<run>/`. Successful finalization removes only owned local and
remote UV caches and disposable `.venv*` directories. Worker source, artifacts,
outputs, and inactive-candidate diagnostics remain available for downstream
evaluation. Paused runs retain package caches unless you explicitly call
`prg.cleanup_search_cache(force=True)`. Existing global UV cache left by older
versions can be reclaimed once with `uv cache prune`.

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
prg.web_search("latest ...")                 # host research; run 2+ before planning
prg.plan_portfolio([...])                    # authored by the current Pi session
prg.orchestrate_portfolio("breadth")         # controller launches workers, not strategy
prg.portfolio_status()                       # audits/failures/objectives for host review
prg.eval("candidate_0")                      # manual/legacy val scoring
prg.eval("candidate_0", split="train", per_instance=True)   # per-row, train only
prg.run("candidate_0", split="train", row=17)  # single traced run — train rows only
prg.frontier()                               # Pareto frontier: best candidate per train row
prg.tradeoffs()                              # quality / cost / latency frontier
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

**What `activate()` does, mechanically:** it writes `active.json` with the chosen candidate's name and its pinned test score, and regenerates `pyproject.toml` so the package's dependencies are exactly the active candidate's PEP 723 `dependencies` list (minus the self-reference, plus its artifacts). `__init__.py` reads `active.json` and imports that one candidate. Switching candidates is a one-line diff you can review, commit, and revert.

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
translate.distill(
    model="gpt-4.1-nano", data="logs", output="translate_ft_ap",
    budget=ap.Budget(dollars=5),
)
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

Heavy candidates keep their weights in `artifacts/` (tracked with git-lfs). A PEP 723 hint such as `[tool.ap] fetch = ["huggingface:Helsinki-NLP/opus-mt-en-fr"]` records the artifact source for tooling, but candidates must currently implement their own lazy first-use download. Zipping the directory works too — `artifacts/` goes with it.

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
# /// script
# dependencies = ["scikit-learn>=1.4", "translate-ap"]
# [tool.uv.sources]
# translate-ap = { path = "..", editable = true }
# ///
import pickle
from translate_ap.paths import artifacts   # resolves to the package's artifacts/ dir

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
# /// script
# dependencies = ["transformers>=4.40", "torch", "translate-ap"]
# [tool.uv.sources]
# translate-ap = { path = "..", editable = true }
# [tool.ap]
# fetch = ["huggingface:Helsinki-NLP/opus-mt-en-fr"]
# ///
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
