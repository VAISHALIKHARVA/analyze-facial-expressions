import io

import cv2
import numpy as np
from PIL import Image, ImageDraw

_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

EMOTION_KEYS = (
    "happy",
    "sad",
    "neutral",
    "surprise",
    "angry",
    "fear",
    "disgust",
)

DISPLAY_NAMES = {
    "happy": "Happiness",
    "sad": "Sadness",
    "neutral": "Neutral",
    "surprise": "Surprise",
    "angry": "Angry",
    "fear": "Fear",
    "disgust": "Disgust",
}


def bytes_to_cv2(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not decode image")
    return image


def detect_largest_face(image_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return int(x), int(y), int(w), int(h)


def crop_face(image_bgr: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    return image_bgr[y : y + h, x : x + w]


def bgr_to_jpeg_bytes(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        raise ValueError("Could not encode face image")
    return encoded.tobytes()


def normalize_emotions(raw: dict[str, float]) -> dict[str, float]:
    cleaned: dict[str, float] = {}
    for key in EMOTION_KEYS:
        value = float(raw.get(key, raw.get(DISPLAY_NAMES.get(key, ""), 0)) or 0)
        cleaned[DISPLAY_NAMES[key]] = max(0.0, value)

    total = sum(cleaned.values())
    if total <= 0:
        cleaned[DISPLAY_NAMES["neutral"]] = 100.0
        return cleaned

    return {name: round((v / total) * 100, 1) for name, v in cleaned.items()}


def pick_main_emotion(emotions: dict[str, float]) -> tuple[str, float]:
    main = max(emotions.items(), key=lambda item: item[1])
    return main[0], main[1]


def draw_bounding_box(image_bytes: bytes, box: tuple[int, int, int, int]) -> bytes:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = box
    draw.rectangle([x, y, x + w, y + h], outline=(0, 200, 83), width=3)
    out = io.BytesIO()
    image.save(out, format="JPEG")
    return out.getvalue()
