from typing import List
from app.models import FeedbackSignal, MoodSignal

# Armazenamento em memória por agora
# Na Etapa 2 pode ser substituído por SQLite ou PostgreSQL
_feedback_store: List[dict] = []
_mood_events: List[dict] = []
_starter_songs: List[dict] = []

def store_feedback(signal: FeedbackSignal):
    _feedback_store.append(signal.model_dump())

def store_mood_event(signal: MoodSignal):
    _mood_events.append(signal.model_dump())

def get_all_feedback() -> List[dict]:
    return _feedback_store

def get_all_mood_events() -> List[dict]:
    return _mood_events

def store_starter_song(song: dict):
    _starter_songs.append(song)

def get_starter_songs() -> List[dict]:
    return _starter_songs