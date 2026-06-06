import uuid
import random
from fastapi import APIRouter, Path, Request, Query
from typing import Optional
from typing import List
from models import GenerateRequest, TrackResponse, StarterSong, Mood, FeedbackSignal, FeedbackValue
from storage import store_starter_song, get_starter_songs, store_feedback
import os
import base64

router = APIRouter()

MOOD_BPM_DEFAULTS = {
    Mood.happy: (120, 160),
    Mood.calm: (60, 90),
    Mood.sad: (50, 80),
    Mood.energetic: (140, 180),
    Mood.happy: (120, 160),
    Mood.calm: (60, 90),
    Mood.sad: (50, 80),
    Mood.energetic: (140, 180),
}

TRACK_NAMES = [
    "Nebula Drift", "Solar Haze", "Coastal Echo", "Midnight Bloom",
    "Crystal Fog", "Ember Flow", "Soft Collapse", "Neon Quiet",
]

@router.get("/generate", response_model=TrackResponse, summary="Enviar nova faixa 100% adaptativa")
def generate_track(
    request: Request,
    mood: Optional[str] = Query(
        default=None,
        description="Mood da faixa: happy | sad | energetic | calm. "
                    "Se omitido, o Bandit escolhe automaticamente.",
    ),
):
    engine = request.app.state.engine

    if mood is not None:
        chosen_mood = mood
        extra_info = {"source": "frontend_override"}
        print(f"\n\n\n🎨 Mood definido pelo frontend: {chosen_mood.upper()}\n\n\n")

    else:
        chosen_mood, extra_info = engine._pick_mood()
        print(f"🎲 O Bandit decidiu que o próximo mood será: {chosen_mood.upper()}")

    # 2. Gerar a faixa com o mood escolhido
    track_info = engine._generate_track(chosen_mood)

    filename = os.path.basename(track_info.path)
    print(f"✅ Faixa gerada: {filename} (Mood: {chosen_mood}, BPM: {track_info.bpm}, Density: {track_info.density})")
    print(track_info)

    return {
        "track_id": str(track_info.id),
        "name": filename,
        "bpm": int(track_info.bpm),
        "density": track_info.density,
        "mood": chosen_mood,
        "base64_file": track_info.base64_file,
    }

@router.get("/starter", response_model=List[StarterSong], summary="Buscar starter songs (usado pelo front-end)")
def get_starter_songs_endpoint():
    """ Devolve todas as músicas iniciais. O front-end chama isto no arranque da aplicação. """
    return get_starter_songs()

@router.post("/{track_id}/feedback", summary="Enviar feedback por track ID")
def post_feedback_by_track(
    track_id: str = Path(..., example="ABC123"),
    mood: str = "",
    bpm: int = 90,
    feedback: FeedbackValue = FeedbackValue.like,
):
    """
    Regista feedback (like/dislike/skip) para uma faixa específica pelo ID.
    Inclui mood e bpm para o Bandit ter contexto completo.
    """
    signal = FeedbackSignal(
        track_id=track_id,
        mood=mood,
        bpm=bpm,
        feedback=feedback,
    )
    store_feedback(signal)
    return {"status": "ok", "track_id": track_id, "feedback": feedback}