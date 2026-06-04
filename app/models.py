from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from enum import Enum

class Mood(str, Enum):
    happy = "happy"
    calm = "calm"
    sad = "sad"
    energetic = "energetic"

class FeedbackValue(str, Enum):
    like = "like"
    dislike = "dislike"
    skip = "skip"

# Sinais

class FeedbackSignal(BaseModel):
    track_id: str = Field(..., example="ABC123")
    mood: Mood
    bpm: int = Field(..., ge=40, le=200, example=90)
    feedback: FeedbackValue
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "track_id": "ABC123",
                "mood": "calm",
                "bpm": 75,
                "feedback": "like",
                "timestamp": "2026-06-01T22:00:00"
            }
        }

class MoodSignal(BaseModel):
    mood: Mood
    timestamp: Optional[datetime] = Field(default_factory=datetime.utcnow)

# Faixas

class GenerateRequest(BaseModel):
    mood: Mood
    bpm: Optional[int] = Field(None, ge=40, le=200, description="Se não fornecido, o sistema escolhe com base no mood")
    density: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Densidade de notas (0=esparso, 1=denso)")

    class Config:
        json_schema_extra = {
            "example": {
                "mood": "Calm",
                "bpm": 75,
                "density": 0.4
            }
        }

class TrackResponse(BaseModel):
    track_id: str
    mood: Mood
    bpm: int
    density: float
    name: str

class StarterSong(BaseModel):
    track_id: str
    mood: Mood
    bpm: int
    density: float
    name: str