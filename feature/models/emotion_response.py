from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class EmotionAnalyzeResponse(BaseModel):
    main_emotion: str
    confidence: float = Field(ge=0, le=100)
    emotions: dict[str, float]
    face_detected: bool
    bounding_box: BoundingBox | None = None
    image_url: str | None = None
