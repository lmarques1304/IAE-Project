from fastapi import APIRouter, Request # <-- Adicionar Request aqui
from models import FeedbackSignal, MoodSignal
from storage import store_feedback, store_mood_event, get_all_feedback

router = APIRouter()

@router.post("/feedback", summary="Registar feedback do utilizador")
def post_feedback(signal: FeedbackSignal, request: Request): # <-- Adicionar request: Request
    """Recebe o feedback e envia para o Bandit (Etapa 2)."""
    # Guardar na base de dados (ou memória) como já fazias
    store_feedback(signal)
    
    # Reconstruir os parâmetros (por agora density/complexity vão em default, 
    # mais tarde o frontend pode enviar tudo)
    track_params = {
        "bpm": signal.bpm,
        "density": 0.5,
        "complexity": 0.5
    }
    
    # Chamar a função nova que colaste no motor!
    engine = request.app.state.engine
    engine.process_api_feedback(
        mood=signal.mood.value, 
        track_params=track_params, 
        feedback_value=signal.feedback.value
    )

    print(signal)
    
    return {"status": "ok", "received": signal}

@router.post("/mood", summary="Registar mudança de mood")
def post_mood(signal: MoodSignal):
    """
    Regista quando o utilizador muda de mood no seletor.
    Útil para perceber padrões de uso.
    """
    store_mood_event(signal)
    return {"status": "ok", "received": signal}

@router.get("/feedback", summary="Listar todos os sinais de feedback")
def list_feedback():
    """
    Devolve todos os sinais recolhidos.
    Usado internamente pela Otimização Bayesiana (Etapa 2).
    """
    return {"count": len(get_all_feedback()), "data": get_all_feedback()}