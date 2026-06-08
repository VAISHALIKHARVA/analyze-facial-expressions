# Emotion Analyzer API

FastAPI service for facial expression analysis using OpenCV (face detection) and Gemini 2.5 Flash (emotions).

## Project structure

```text
Analyze-Facial-Expressions/
├── feature/
│   ├── app.py
│   ├── routers/emotion.py
│   ├── services/
│   ├── models/
│   └── uploads/
├── requirements.txt
└── .env
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

## Run API

```bash
uvicorn feature.app:app --reload --host 127.0.0.1 --port 8000
```

## API

`POST /analyze` — multipart form field `file` (image).

Returns JSON with `main_emotion`, `confidence`, `emotions`, `face_detected`, and `bounding_box`.

Docs: http://127.0.0.1:8000/docs

## Ionic Angular app

Create your Ionic Angular app separately and call:

```text
POST http://127.0.0.1:8000/analyze
```

CORS is enabled for all origins.
