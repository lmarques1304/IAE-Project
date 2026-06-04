import uuid
import random
from fastapi import APIRouter, Path, Request
from typing import List
from models import GenerateRequest, TrackResponse, StarterSong, Mood, FeedbackSignal, FeedbackValue
from storage import store_starter_song, get_starter_songs, store_feedback
import os

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

@router.post("/generate", response_model=TrackResponse, summary="Gerar nova faixa 100% adaptativa")
def generate_track(request: Request): 
    engine = request.app.state.engine
    
    # 1. 🧠 O BANDIT ASSUME O CONTROLO TOTAL
    chosen_mood, extra_info = engine._pick_mood()
    print(f"🎲 O Bandit decidiu que o próximo mood será: {chosen_mood.upper()}")
    
    # 2. Gerar a faixa com o mood estatisticamente escolhido
    track_info = engine._generate_track(chosen_mood)
    
    filename = os.path.basename(track_info.path)
    file_url = f"{request.base_url}midi/{filename}"
    
    print(f"✅ Faixa gerada: {filename} (Mood: {chosen_mood}, BPM: {track_info.bpm}, Density: {track_info.density})")
    return {
        "track_id": str(track_info.id),
        "name": filename,
        "bpm": int(track_info.bpm),
        "density": track_info.density,
        "mood": chosen_mood,
        "url": file_url
    }

@router.post("/starter", summary="Adicionar starter song")
def add_starter_song(song: StarterSong):
    """
    Endpoint para injetar músicas iniciais na app.
    O front-end usa GET /tracks/starter para as ir buscar.
    """
    store_starter_song(song.model_dump())
    return {"status": "ok", "track_id": song.track_id}

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