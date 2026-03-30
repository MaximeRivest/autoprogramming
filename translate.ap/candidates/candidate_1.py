"""Mutated from candidate_0.

Reflection: candidate_0 scored low on idiomatic expressions (row 11:
"It's raining cats and dogs" -> "Il pleut des chats et des chiens"
instead of "Il pleut des cordes"). Also produced overly formal register
("Pourriez-vous" instead of "Pouvez-vous"). Adjusted prompt to prefer
natural idiomatic French and standard register.
"""
from openai import OpenAI

from ..schema import French

client = OpenAI()


def predict(english: str) -> French:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a professional English-to-French translator. "
                    "Translate the given text to natural, idiomatic French. "
                    "Use standard register (not overly formal, not slang). "
                    "Translate idioms to their French equivalents rather than literally. "
                    "Reply with only the translation, nothing else."
                ),
            },
            {"role": "user", "content": english},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return French(response.choices[0].message.content.strip())
