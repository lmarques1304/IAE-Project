from typing import List
from app.models import FeedbackSignal, MoodSignal

# Armazenamento em memória por agora
# Na Etapa 2 pode ser substituído por SQLite ou PostgreSQL
_feedback_store: List[dict] = []
_mood_events: List[dict] = []


def store_feedback(signal: FeedbackSignal):
    _feedback_store.append(signal.model_dump())


def store_mood_event(signal: MoodSignal):
    _mood_events.append(signal.model_dump())


def get_all_feedback() -> List[dict]:
    return _feedback_store


def get_all_mood_events() -> List[dict]:
    return _mood_events
