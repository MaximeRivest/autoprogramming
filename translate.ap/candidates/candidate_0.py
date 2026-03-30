"""Seed candidate. Auto-generated from schema."""
from openai import OpenAI

from ..schema import French

client = OpenAI()


def predict(english: str) -> French:
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": "Translate the following English text to French. Reply with only the translation, nothing else.",
            },
            {"role": "user", "content": english},
        ],
        temperature=0.0,
        max_tokens=256,
    )
    return French(response.choices[0].message.content.strip())
