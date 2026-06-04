import uuid
import random
from fastapi import APIRouter, Path
from typing import List
from app.models import GenerateRequest, TrackResponse, StarterSong, Mood, FeedbackSignal, FeedbackValue
from app.storage import store_starter_song, get_starter_songs, store_feedback

router = APIRouter()

MOOD_BPM_DEFAULTS = {
    Mood.happy: (120, 160),
    Mood.calm: (60, 90),
    Mood.sad: (50, 80),
    Mood.energetic: (140, 180),
}

TRACK_NAMES = [
    "Nebula Drift", "Solar Haze", "Coastal Echo", "Midnight Bloom",
    "Crystal Fog", "Ember Flow", "Soft Collapse", "Neon Quiet",
]

def call_midi_generator(mood: Mood, bpm: int, density: float) -> dict:
    """
    Stub que simula a chamada ao gerador MIDI.
    Na Etapa 2 substituir por:
        response = requests.post("http://<host-do-J>/generate", json={...})
        return response.json()
    """
    return {
        "track_id": uuid.uuid4().hex[:8].upper(),
        "name": random.choice(TRACK_NAMES),
        "bpm": bpm,
        "density": density,
        "mood": mood,
    }

@router.post("/generate", response_model=TrackResponse, summary="Gerar nova faixa")
def generate_track(req: GenerateRequest):
    """Gera uma nova faixa com base no mood e parâmetros fornecidos."""
    bpm_min, bpm_max = MOOD_BPM_DEFAULTS[req.mood]
    bpm = req.bpm if req.bpm else random.randint(bpm_min, bpm_max)
    density = req.density if req.density is not None else 0.5
    track = call_midi_generator(mood=req.mood, bpm=bpm, density=density)
    return TrackResponse(**track)

@router.post("/starter", summary="Adicionar starter song (usado pelo J)")
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