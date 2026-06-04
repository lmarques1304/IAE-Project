from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os

from app.routers import signals, tracks
from adaptive_player import AdaptiveEngine  # <-- Importar o motor

MIDI_DIR = "adaptive_midi"
os.makedirs(MIDI_DIR, exist_ok=True)

# Instanciar o motor globalmente (mas não chamar .start() para não abrir o CLI)
engine = AdaptiveEngine(output_dir=MIDI_DIR, algo="thompson")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Colocar o motor no state da app para os routers acederem
    app.state.engine = engine
    print("🎵 Adaptive Engine iniciado e ligado à API!")
    
    # Opcional: Podes chamar o get_pre_generated_batch() aqui no futuro
    # para pré-carregar as starter songs.
    
    yield
    print("🛑 Servidor a desligar...")

app = FastAPI(
    title="Adaptive Music API",
    description="Backend para o sistema de música adaptativa com recolha de sinais e geração de faixas.",
    version="1.0.0",
    lifespan=lifespan
)
engine = AdaptiveEngine(output_dir=MIDI_DIR, algo="thompson")

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

@app.get("/health", tags=["Sistema"])
def health_check():
    return {"status": "ok"}
