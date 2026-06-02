import uuid
import random
from fastapi import APIRouter
from app.models import GenerateRequest, TrackResponse, Mood

router = APIRouter()

MOOD_BPM_DEFAULTS = {
    Mood.feliz: (120, 160),
    Mood.calmo: (60, 90),
    Mood.triste: (50, 80),
    Mood.energetico: (140, 180),
}

TRACK_NAMES = [
    "Nebula Drift", "Solar Haze", "Coastal Echo", "Midnight Bloom",
    "Crystal Fog", "Ember Flow", "Soft Collapse", "Neon Quiet",
]


def call_midi_generator(mood: Mood, bpm: int, density: float) -> dict:
    """
    Stub que simula a chamada ao gerador MIDI. Etapa 2
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
    """
    Gera uma nova faixa com base no mood e parâmetros fornecidos.
    Chamado pelo front-end quando o utilizador clica em 'Gerar'.
    """
    bpm_min, bpm_max = MOOD_BPM_DEFAULTS[req.mood]
    bpm = req.bpm if req.bpm else random.randint(bpm_min, bpm_max)
    density = req.density if req.density is not None else 0.5

    track = call_midi_generator(mood=req.mood, bpm=bpm, density=density)
    return TrackResponse(**track)
