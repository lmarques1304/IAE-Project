from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import signals, tracks

app = FastAPI(
    title="Adaptive Music API",
    description="Backend para o sistema de música adaptativa com recolha de sinais e geração de faixas.",
    version="1.0.0",
)

# Permite chamadas do front-end durante desenvolvimento
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signals.router, prefix="/signals", tags=["Sinais"])
app.include_router(tracks.router, prefix="/tracks", tags=["Faixas"])

@app.get("/health", tags=["Sistema"])
def health_check():
    return {"status": "ok"}