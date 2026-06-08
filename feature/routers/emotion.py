from fastapi import APIRouter, File, HTTPException, UploadFile

from feature.models.emotion_response import BoundingBox, EmotionAnalyzeResponse
from feature.services import gemini_service, image_processor

router = APIRouter(prefix="/analyze", tags=["emotion"])


@router.post("", response_model=EmotionAnalyzeResponse)
async def analyze_emotion(file: UploadFile = File(...)) -> EmotionAnalyzeResponse:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image_bgr = image_processor.bytes_to_cv2(image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    face_box = image_processor.detect_largest_face(image_bgr)
    if face_box is None:
        return EmotionAnalyzeResponse(
            main_emotion="Neutral",
            confidence=0.0,
            emotions={name: 0.0 for name in image_processor.DISPLAY_NAMES.values()},
            face_detected=False,
            bounding_box=None,
        )

    x, y, w, h = face_box
    face_crop = image_processor.crop_face(image_bgr, face_box)
    face_jpeg = image_processor.bgr_to_jpeg_bytes(face_crop)

    try:
        emotions = gemini_service.analyze_face_emotions(face_jpeg)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Emotion analysis failed: {exc}") from exc

    main_emotion, confidence = image_processor.pick_main_emotion(emotions)

    return EmotionAnalyzeResponse(
        main_emotion=main_emotion,
        confidence=confidence,
        emotions=emotions,
        face_detected=True,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
    )
