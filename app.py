import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import cv2
import numpy as np
import json
import io
import os
from dotenv import load_dotenv

# =========================
# Load API Key
# =========================
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# =========================
# Streamlit UI
# =========================
st.title("AI Facial Emotion Rating")

st.write(
    "Upload a clear image with one visible face."
)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# Face Detection Function
# =========================
def detect_face(image):
    image_np = np.array(image)

    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades +
        "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=8,
        minSize=(80, 80)
    )

    return faces, image_np

# =========================
# Gemini Emotion Analysis
# =========================
def analyze_emotion(face_image):

    prompt = """
    Analyze the facial expression of this person.

    Detect emotional signals from:
    - eyes
    - eyebrows
    - lips
    - smile
    - forehead tension
    - facial muscle tension

    Return ONLY valid JSON.

    Example:
    {
      "primary_emotion": "Confused",
      "emotions": {
        "Happy": 10,
        "Sad": 15,
        "Angry": 5,
        "Confused": 85,
        "Neutral": 20,
        "Surprised": 12
      },
      "explanation": "The eyebrows appear tightened and the eyes show uncertainty."
    }
    """

    buf = io.BytesIO()
    face_image.save(buf, format="JPEG")
    image_bytes = buf.getvalue()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
    )

    return response.text

# =========================
# Main Logic
# =========================
if uploaded_file:

    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width='stretch')

    with st.spinner("Detecting face..."):

        faces, image_np = detect_face(image)

        if len(faces) == 0:
            st.error("No face detected.")
        elif len(faces) > 1:
            st.error("Multiple faces detected. Upload one person image.")
        else:

            x, y, w, h = faces[0]

            # Draw rectangle
            cv2.rectangle(
                image_np,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            face_crop = image_np[y:y+h, x:x+w]

            face_pil = Image.fromarray(face_crop)

            st.image(
                face_pil,
                caption="Detected Face",
                width=250
            )

            with st.spinner("Analyzing emotions with Gemini..."):

                result = analyze_emotion(face_pil)

                try:
                    cleaned = result.replace("```json", "").replace("```", "")

                    data = json.loads(cleaned)

                    st.subheader(
                        f"Primary Emotion: {data['primary_emotion']}"
                    )

                    st.write(data["explanation"])

                    st.subheader("Emotion Ratings")

                    for emotion, score in data["emotions"].items():
                        st.write(f"{emotion}: {score}%")
                        st.progress(score / 100)

                except Exception as e:
                    st.error("Failed to parse Gemini response.")
                    st.code(result)