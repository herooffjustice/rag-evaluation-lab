"""
llm.py - small helper to call Google Gemini.

The API key is read from a .env file (GEMINI_API_KEY).
The .env file can be in this folder or in any parent folder.
"""

import os
import time

from dotenv import load_dotenv, find_dotenv
from google import genai

load_dotenv(find_dotenv())

DEFAULT_MODEL = "gemini-3.5-flash-lite"   # small, fast model; works on the free tier


def get_model_name():
    return os.getenv("GEMINI_MODEL", DEFAULT_MODEL)


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "paste_your_key_here":
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the .env file "
            "(see .env.example)."
        )
    return genai.Client(api_key=api_key)


def generate(prompt, retries=4):
    """Send one prompt to Gemini and return the text reply.

    If the free-tier rate limit is hit (error 429) or the server is busy
    (error 503), wait and try again.
    """
    client = get_client()
    wait = 10
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model=get_model_name(),
                contents=prompt,
            )
            return (response.text or "").strip()
        except Exception as error:  # noqa: BLE001
            message = str(error)
            if "PerDay" in message:
                # Daily free-tier quota is used up: waiting a few seconds will not help.
                raise RuntimeError(
                    f"Daily free-tier limit reached for {get_model_name()}. "
                    "Try again tomorrow or set GEMINI_MODEL in .env to another model."
                ) from error
            busy = "429" in message or "503" in message or "RESOURCE_EXHAUSTED" in message
            if busy and attempt < retries - 1:
                time.sleep(wait)
                wait *= 2
                continue
            raise
