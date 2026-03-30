"""Mutated from candidate_1.

Reflection: candidate_1 improved idioms (+0.08 avg) but still struggles
with present perfect -> present tense mapping (row 14: "He has been
working here for ten years" -> "Il a travaillé ici pendant dix ans"
instead of "Il travaille ici depuis dix ans"). Also inconsistent
punctuation spacing with French conventions. Added tense guidance
and few-shot examples targeting these failure modes.
"""
from openai import OpenAI

from ..schema import French

client = OpenAI()

SYSTEM_PROMPT = """\
You are a professional English-to-French translator.

Rules:
- Produce natural, idiomatic French in standard register.
- Translate idioms to their French equivalents, never literally.
- English present perfect continuous ("has been doing") maps to French \
present tense + "depuis" ("fait depuis"), not passé composé.
- Follow French punctuation conventions: space before ? ! ; :

Examples:
- "It's raining cats and dogs." -> "Il pleut des cordes."
- "He has been working here for ten years." -> "Il travaille ici depuis dix ans."
- "I'll be there in five minutes." -> "Je serai là dans cinq minutes."

Reply with only the translation, nothing else.\
"""


def predict(english: str) -> French:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": english},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return French(response.choices[0].message.content.strip())
