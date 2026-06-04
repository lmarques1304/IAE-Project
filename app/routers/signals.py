from fastapi import APIRouter
from app.models import FeedbackSignal, MoodSignal
from app.storage import store_feedback, store_mood_event, get_all_feedback

router = APIRouter()

@router.post("/feedback", summary="Registar feedback do utilizador")
def post_feedback(signal: FeedbackSignal):
    """
    Recebe o feedback (👍 ou 👎) do utilizador para uma faixa.
    Usado pelo front-end do Simão após o utilizador interagir com o player.
    """
    store_feedback(signal)
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