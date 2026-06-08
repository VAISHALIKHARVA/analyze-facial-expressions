import json
import os
import re

import google.generativeai as genai

from feature.services.image_processor import EMOTION_KEYS, normalize_emotions

PROMPT = """
Analyze the facial expression in this image.

Return ONLY JSON.

{
  "main_emotion": "",
  "confidence": 0,
  "emotions": {
      "happy": 0,
      "sad": 0,
      "neutral": 0,
      "surprise": 0,
      "angry": 0,
      "fear": 0,
      "disgust": 0
  }
}

Rules:
- Total must equal 100
- Detect strongest emotion
- Return percentages
- No explanation
"""


def _configure_client() -> None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            raise ValueError("Model did not return valid JSON") from None
        return json.loads(match.group())


def analyze_face_emotions(face_jpeg: bytes) -> dict[str, float]:
    _configure_client()
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content(
        [
            PROMPT,
            {"mime_type": "image/jpeg", "data": face_jpeg},
        ]
    )
    payload = _extract_json(response.text or "")
    raw_emotions = payload.get("emotions", {})
    if not isinstance(raw_emotions, dict):
        raw_emotions = {}

    normalized_input = {
        key: float(raw_emotions.get(key, 0) or 0) for key in EMOTION_KEYS
    }
    return normalize_emotions(normalized_input)
