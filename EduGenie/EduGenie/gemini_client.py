
"""Shared Google Gemini client used by every EduGenie module."""

import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = (
    os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
    or "gemini-3.8-flash"
)

_client = None


class GeminiError(Exception):
    """Raised for any problem talking to Gemini."""


def _get_client():
    global _client

    if _client is not None:
        return _client

    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key or api_key == "your_gemini_api_key_here":
        raise GeminiError(
            "GEMINI_API_KEY is not set. "
            "Copy .env.example to .env and add your Gemini API key."
        )

    try:
        from google import genai
    except ImportError as exc:
        raise GeminiError(
            "The 'google-genai' package is not installed. "
            "Run: pip install -r requirements.txt"
        ) from exc

    _client = genai.Client(api_key=api_key)
    return _client


def generate(prompt: str, json_mode: bool = False) -> str:
    """Send a prompt to Gemini and return the response text."""

    client = _get_client()

    kwargs = {
        "model": MODEL_NAME,
        "contents": prompt,
    }

    if json_mode:
        try:
            from google.genai import types

            kwargs["config"] = types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        except Exception:
            pass

    # Retry temporary 503 errors
    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(**kwargs)

            text = getattr(response, "text", None)

            if not text or not text.strip():
                raise GeminiError(
                    "Gemini returned an empty response. "
                    "Try rephrasing your input."
                )

            return text.strip()

        except Exception as exc:
            error_message = str(exc)

            # Retry only temporary service-unavailable errors
            if "503" in error_message or "UNAVAILABLE" in error_message:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)
                    continue

            raise GeminiError(
                f"Gemini API error: {exc}"
            ) from exc

    raise GeminiError("Gemini API is temporarily unavailable. Please try again later.")


def clean_json_block(text: str) -> str:
    """Strip Markdown JSON code fences from a model reply."""

    return re.sub(
        r"```(?:json)?\s*(.*?)```",
        r"\1",
        text,
        flags=re.DOTALL,
    ).strip()


# successfully completed 
