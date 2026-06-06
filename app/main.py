from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from routers import signals, tracks, optimize
from adaptive_player import AdaptiveEngine
from storage import store_starter_song, get_starter_songs

MIDI_DIR = "adaptive_midi"
os.makedirs(MIDI_DIR, exist_ok=True)

engine = AdaptiveEngine(output_dir=MIDI_DIR)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Colocar o motor no state da app para os routers acederem
    app.state.engine = engine
    print("🎵 Adaptive Engine iniciado e ligado à API!")
    
    starter_tracks = engine.get_pre_generated_batch()
    
    # 3. Format and store them in memory so GET /tracks/starter works
    for track in starter_tracks:

        # We format the dictionary to match your StarterSong Pydantic model
        song_data = {
            "track_id": f"PRE_{track['id']}", # Make sure it's a string
            "mood": track["mood"],
            "bpm": int(track["bpm"]),
            "density": 0.5, # Default density since it isn't in the filename
            "name": track["filename"],
            "base64_file": track["base64_file"]
        }
        store_starter_song(song_data)

    yield
    print("🛑 Servidor a desligar...")

app = FastAPI(
    title="Adaptive Music API",
    description="Backend para o sistema de música adaptativa com recolha de sinais e geração de faixas.",
    version="1.0.0",
    lifespan=lifespan
)
engine = AdaptiveEngine(output_dir=MIDI_DIR)

app.state.engine = engine

# Permite chamadas do front-end durante desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve os ficheiros .mid gerados (subpastas por mood: /midi/happy/ficheiro.mid)
app.mount("/midi", StaticFiles(directory=MIDI_DIR), name="midi")


app.include_router(signals.router, prefix="/signals", tags=["Sinais"])
app.include_router(tracks.router, prefix="/tracks", tags=["Faixas"])
app.include_router(optimize.router, prefix="/optimize", tags=["Otimização Bayesiana"])

@app.get("/health", tags=["Sistema"])
def health_check():
    return {"status": "ok"}