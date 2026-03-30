# AutoProgramming

Define your inputs and outputs, and let AutoProgramming generate the code for you.

## Define a program

Types are subclasses of builtins. Docstrings become descriptions.

```py
import autoprogramming as ap

class Answer(str): ...
class Summary(str): ..

@ap.program
def my_program(question: str) -> tuple[Answer, Summary]: ...
```

## Initialize manually

```py
my_program.initialize(
    instructions="Given a question, produce a response.",
    descriptions=dict(
        question="The user's question",
        Answer="A concise answer",
        Summary="A brief summary",
    ),
    demos=[
        dict(question="What is the capital of France?", Answer="Paris", Summary="Paris is the capital"),
    ],
    model="gpt-4.1",
    adapter=XMLAdapter(),
    temperature=0.7,
    max_tokens=100,
)
```

## Or let AutoProgramming optimize it

```py
class French(str): ...

@ap.program
def translate(english: str) -> French: ...

translate.optimize(data=train_df)
translate.save("translate.ap")

translate("Hello, how are you?")
# => 'Bonjour, comment ça va?'
```

`.optimize()` uses a coding agent that iterates on instructions, descriptions, demos, adapter, and model config using reflective prompt evolution (inspired by [GEPA](https://github.com/gepa-ai/gepa)).

### What the agent actually does

Given the program above, `.optimize()` creates a workspace:

```
translate.ap/
├── schema.py              # the @ap.program definition (immutable)
├── metric.py              # evaluation metric (written by agent)
├── data/
│   ├── train.csv
│   └── val.csv
├── candidates/
│   ├── candidate_0.py     # seed candidate
│   ├── candidate_1.py     # mutated from 0
│   └── candidate_2.py     # mutated from 1
└── scores.json            # per-candidate, per-row scores
```

The agent writes the seed candidate — a complete Python module with formatter, SDK call, and parser:

```py
# candidates/candidate_0.py
from openai import OpenAI
from ..schema import French

client = OpenAI()

def predict(english: str) -> French:
    response = client.chat.completions.create(
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

Writes a metric:

```py
# metric.py
from difflib import SequenceMatcher

def metric(predicted: str, expected: str) -> float:
    return SequenceMatcher(None, predicted.lower(), expected.lower()).ratio()
```

Evaluates:

```py
prg.eval("candidate_0")
# scores.json: {"candidate_0": {"row_0": 0.91, "row_1": 0.85, "row_2": 0.78, "row_3": 0.82, "row_4": 0.95}}
# avg: 0.86
```

Reflects on traces of low-scoring rows, then copies and edits the file to create a new candidate:

```py
# candidates/candidate_1.py  (copied from candidate_0.py, prompt mutated)
from openai import OpenAI
from ..schema import French

client = OpenAI()

def predict(english: str) -> French:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": (
                "Translate the following English text to French. "
                "Use simple, direct phrasing. Prefer short forms "
                "(e.g. 'Où est' over 'Où se trouve'). "
                "Reply with only the translation, nothing else."
            )},
            {"role": "user", "content": english},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return French(response.choices[0].message.content.strip())
```

```py
prg.eval("candidate_1")
# avg: 0.92 — better, keep it
```

Each candidate is a readable, diffable `.py` file. The agent repeats — reflect, copy, edit, evaluate — until the budget is exhausted. Then:

```py
prg.best()        # => "candidate_1"
prg.activate("candidate_1")
```

## Agent API

The agent uses file operations and these helpers:

```py
prg.schema                                    # inspect inputs/outputs
prg.eval("candidate_0")                       # run candidate against val data
prg.eval("candidate_0", per_instance=True)    # per-row scores
prg.run("candidate_0", english="Hello")       # single run with trace
prg.best()                                    # best candidate by aggregate score
prg.activate("candidate_1")                   # set as the live one
prg.data.train                                # training data
prg.data.val                                  # validation data
prg.data.sample(20)                           # random minibatch
```

Everything else — creating candidates, editing prompts, changing SDKs, rewriting parsers — is just file operations on `candidates/*.py`.

## Use it like a normal function

A `.ap` directory is a Python package. Import and call it:

```py
from translate_ap import translate

translate("Hello, how are you?")
# => French('Bonjour, comment allez-vous ?')
```

It's just a function. The return type is `French`, which is a `str` subclass — works everywhere a string does.

### Log inputs/outputs for later

```py
translate.enable_logging()

translate("Where is the train station?")
# => French('Où est la gare ?')
# also writes to translate.ap/logs/2026-03-30.jsonl
```

```json
{"english": "Where is the train station?", "French": "Où est la gare ?", "timestamp": "2026-03-30T14:22:01Z"}
```

Logs accumulate as JSONL. Use them later to evaluate, retrain, or distill to a smaller model:

```py
# re-optimize using production traffic
translate.optimize(data="logs")

# or distill the prompt-based program into a fine-tuned small model
translate.distill(model="gpt-4.1-nano", data="logs", output="translate-ft.ap")
```

### Distribute

It's a directory with `.py` files — publish it however you publish Python:

```sh
# as a package
pip install ./translate.ap

# or just copy the directory
cp -r translate.ap /path/to/other/project/

# or zip it
zip -r translate.ap.zip translate.ap/
```

## How the agent builds a program

When a user says "I want a translator", the agent asks 4 questions, then explores freely.

### Questions the agent asks

**1. What goes in, what comes out?**

The schema. "English text in, French text out" is enough. Could also be "CSV row in, (label, confidence) out" or "image path in, description out". This defines the types.

**2. Do you have examples?**

Even 5-10 input/output pairs are enough to start. If not: "Can I generate synthetic pairs and you validate a few?"

**3. What does "good" mean to you?**

Exact match? Semantic similarity? Latency under 100ms? Cost under $0.001/call? This shapes the metric and what approaches are worth exploring.

**4. Any constraints?**

Can data go to external APIs or must it stay local? Max cost per call? Must it run offline? This narrows the search space.

That's it. Everything else the agent figures out on its own.

### What the agent explores

LLM prompting is one option. But a candidate is just a `.py` file with a `predict` function — the agent can write anything:

**LLM call** — prompt engineering, SDK choice, model selection:
```py
# candidates/candidate_0.py
def predict(english: str) -> French:
    response = openai.chat.completions.create(...)
    return French(response.choices[0].message.content.strip())
```

**Classical ML** — feature engineering, scikit-learn, lightweight and fast:
```py
# candidates/candidate_3.py
import pickle
from sklearn.pipeline import Pipeline

model = pickle.loads((_root / "model.pkl").read_bytes())

def predict(english: str) -> French:
    return French(model.predict([english])[0])
```

**Regex / rule-based** — when patterns are enough:
```py
# candidates/candidate_4.py
import re

RULES = {
    r"\bhello\b": "bonjour",
    r"\bthank you\b": "merci",
    r"\bplease\b": "s'il vous plaît",
    ...
}

def predict(english: str) -> French:
    result = english.lower()
    for pattern, replacement in RULES.items():
        result = re.sub(pattern, replacement, result)
    return French(result)
```

**Deep learning** — fine-tuned transformer, runs locally:
```py
# candidates/candidate_5.py
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

def predict(english: str) -> French:
    tokens = tokenizer(english, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return French(tokenizer.decode(translated[0], skip_special_tokens=True))
```

**Decomposed pipeline** — combine approaches, each handling what it's best at:
```py
# candidates/candidate_6.py
import re
from transformers import MarianMTModel, MarianTokenizer
from openai import OpenAI

# local model for bulk translation
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
client = OpenAI()

IDIOMS = {
    "raining cats and dogs": "il pleut des cordes",
    "break a leg": "bonne chance",
    "piece of cake": "un jeu d'enfant",
}

def predict(english: str) -> French:
    # step 1: regex substitution for known idioms
    text = english.lower()
    for idiom, translation in IDIOMS.items():
        if idiom in text:
            return French(re.sub(re.escape(idiom), translation, text, flags=re.IGNORECASE))

    # step 2: local model for simple sentences
    tokens = tokenizer(english, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    result = tokenizer.decode(translated[0], skip_special_tokens=True)

    # step 3: LLM refinement only if local model confidence is low
    if len(english.split()) > 15:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": "Refine this French translation for naturalness. Reply with only the translation."},
                {"role": "user", "content": f"English: {english}\nDraft: {result}"},
            ],
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip()

    return French(result)
```

The agent tries different approaches, evaluates them on the same data with the same metric, and keeps the best. A regex candidate that scores 0.95 beats an LLM candidate that scores 0.90 — the agent doesn't care how it works, only that it satisfies the schema and the metric.

### Tools the agent needs

```
filesystem       create .ap dir, write candidates, copy/edit .py files
python_repl      run candidates, evaluate, inspect traces
llm_apis         call OpenAI / Anthropic / Gemini from candidate code
web_search       look up domain knowledge, find pretrained models
pip_install      install libraries candidates need (transformers, sklearn, etc.)
user_confirm     show synthetic data or ask clarifying questions
prg.eval()       score a candidate against train/val data
prg.run()        single call with trace for debugging
prg.data         access train/val splits
```

### Exploration strategy

The agent follows a structured search to avoid wasting budget on redundant candidates.

**Phase 1: Baseline sweep.** Start cheap and fast to establish a floor.

1. Write a trivial candidate first (regex, lookup table, or simplest possible heuristic). This is candidate_0 — the baseline everything else must beat.
2. Try the cheapest LLM (gpt-4.1-nano, haiku) with a minimal prompt. Often this is already hard to beat.
3. Try a pretrained model if one exists for the domain (e.g. Helsinki-NLP/opus-mt-en-fr for translation, a sentence-transformers model for classification).

Evaluate all three. Now you know: what's the floor, what's the ceiling, and where the gap is.

**Phase 2: Targeted mutation.** Improve the best candidate by reflecting on its failures.

1. Run `prg.eval(best, per_instance=True)` — find the rows that score lowest.
2. Run `prg.run(best, ...)` on those rows with tracing — read the full trace to understand *why* it failed.
3. Copy the best candidate, make a targeted edit that addresses the failure mode.
4. Evaluate the new candidate. Keep it only if it improves val_avg.

Each iteration should change one thing: the prompt, the model, the parsing logic, a preprocessing step. Never change everything at once — you can't attribute improvement.

**Phase 3: Structural exploration.** If targeted mutations plateau, try a different approach entirely.

- If an LLM candidate plateaus, try a local model or a pipeline.
- If a single model plateaus, decompose the problem (e.g. idiom detection + translation + refinement).
- If cost is a constraint, try distilling the best LLM candidate into a smaller model or a rule set.

**Phase 4: Pareto selection.** Maintain diversity across candidates.

Don't just keep the single best. Track which candidate scores best on *each row*. A candidate that's worse on average but best on specific hard examples carries information. When mutating, sample from the Pareto frontier — candidates that are best at something — not just the global best. This prevents getting stuck in a local optimum.

**Budget discipline:**

- Spend ~10% of budget on Phase 1 (baselines).
- Spend ~60% on Phase 2 (targeted mutation of the most promising approach).
- Spend ~20% on Phase 3 (structural alternatives).
- Spend ~10% on re-evaluating top candidates on full val set at the end.
- Stop mutating a candidate after 3 consecutive failures to improve. Move to a different ancestor or a different approach.
- Every candidate must be evaluated on the same val set. No exceptions — otherwise scores aren't comparable.
