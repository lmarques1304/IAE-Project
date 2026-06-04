"""
midi_player.py
--------------
Reprodutor de ficheiros MIDI — Componente do João (J)
Projeto: Adaptive Music Experience (C2)

Funcionalidades:
  - Reprodução de ficheiros .mid via pygame.mixer
  - Controlos: play, pause, stop, skip, próxima
  - Fila de reprodução com suporte a mood
  - Registo de sinais: tempo ouvido, skip (para o backend do L)
  - Interface de linha de comandos simples para testes locais

Dependências:
  pip install pygame pretty_midi numpy
  
Nota sobre soundfonts:
  O pygame.mixer reproduz MIDI usando o sintetizador do sistema (FluidSynth
  no Linux, QuickTime no macOS, Windows GS no Windows). Para uma melhor
  qualidade de som, podes usar uma soundfont .sf2:
    → https://musescore.org/en/handbook/soundfonts-and-sfz-files
  e carregar com: pygame.mixer.music.set_soundfont("path/to/font.sf2")
"""

import os
import time
import threading
import pygame
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Estado do player
# ---------------------------------------------------------------------------

class PlayerState(Enum):
    STOPPED  = auto()
    PLAYING  = auto()
    PAUSED   = auto()


@dataclass
class TrackInfo:
    """Metadados de uma faixa na fila."""
    id: int
    path: str
    mood: str = "desconhecido"
    bpm: float = 120.0
    density: float = 0.5
    complexity: float = 0.5


@dataclass
class PlaybackSignals:
    """
    Sinais recolhidos durante a reprodução.
    Enviados ao backend (L) após cada faixa.
    """
    track_id: int
    mood: str
    listen_time: float = 0.0   # segundos ouvidos
    skipped: bool = False
    liked: bool = False
    disliked: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "mood": self.mood,
            "listen_time": round(self.listen_time, 2),
            "skipped": self.skipped,
            "liked": self.liked,
            "disliked": self.disliked,
        }


# ---------------------------------------------------------------------------
# Player principal
# ---------------------------------------------------------------------------

class MIDIPlayer:
    """
    Reprodutor de ficheiros MIDI com registo de sinais.

    Uso básico:
        player = MIDIPlayer()
        player.add_to_queue(TrackInfo(id=0, path="track.mid", mood="happy"))
        player.play()

    Com callbacks (para integrar com Streamlit/backend):
        player = MIDIPlayer(
            on_track_end=lambda signals: send_to_backend(signals),
            on_track_start=lambda track: update_ui(track),
        )
    """

    def __init__(
        self,
        on_track_end:   Optional[Callable[[PlaybackSignals], None]] = None,
        on_track_start: Optional[Callable[[TrackInfo], None]]       = None,
        volume: float = 0.8,
    ):
        self.on_track_end   = on_track_end
        self.on_track_start = on_track_start

        self._queue:        list[TrackInfo]       = []
        self._current:      Optional[TrackInfo]   = None
        self._signals:      Optional[PlaybackSignals] = None
        self._state:        PlayerState           = PlayerState.STOPPED
        self._start_time:   float                 = 0.0
        self._paused_accum: float                 = 0.0  # tempo acumulado antes de pausa
        self._lock          = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_monitor  = threading.Event()

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        pygame.mixer.music.set_volume(volume)

    # ------------------------------------------------------------------
    # Fila de reprodução
    # ------------------------------------------------------------------

    def add_to_queue(self, track: TrackInfo):
        """Adiciona uma faixa ao fim da fila."""
        with self._lock:
            self._queue.append(track)

    def add_batch(self, tracks: list[TrackInfo]):
        """Adiciona várias faixas de uma vez (ex: cold start)."""
        for t in tracks:
            self.add_to_queue(t)

    def clear_queue(self):
        with self._lock:
            self._queue.clear()

    @property
    def queue_length(self) -> int:
        return len(self._queue)

    # ------------------------------------------------------------------
    # Controlos de reprodução
    # ------------------------------------------------------------------

    def play(self):
        """Inicia ou retoma a reprodução."""
        if self._state == PlayerState.PAUSED:
            pygame.mixer.music.unpause()
            self._start_time = time.time()  # reinicia contagem do tempo paused
            self._state = PlayerState.PLAYING
            print(f"  ▶  A retomar: {self._current.path}")
            return

        if self._state == PlayerState.STOPPED:
            self._play_next()

    def pause(self):
        """Pausa a reprodução."""
        if self._state == PlayerState.PLAYING:
            pygame.mixer.music.pause()
            self._paused_accum += time.time() - self._start_time
            self._state = PlayerState.PAUSED
            print(f"  ⏸  Pausado.")

    def stop(self):
        """Para a reprodução e limpa a faixa atual."""
        self._stop_monitor.set()
        pygame.mixer.music.stop()
        self._finalise_signals(skipped=False)
        self._state = PlayerState.STOPPED
        self._current = None
        print("  ⏹  Parado.")

    def skip(self):
        """Salta para a próxima faixa e regista o skip."""
        if self._current:
            print(f"  ⏭  Skip: {self._current.path}")
        self._finalise_signals(skipped=True)
        pygame.mixer.music.stop()
        self._play_next()

    def next_track(self):
        """Alias para skip (compatível com o frontend do S)."""
        self.skip()

    def like(self):
        """Regista like na faixa atual."""
        if self._signals:
            self._signals.liked    = True
            self._signals.disliked = False
            print("  👍  Like registado.")

    def dislike(self):
        """Regista dislike na faixa atual."""
        if self._signals:
            self._signals.disliked = True
            self._signals.liked    = False
            print("  👎  Dislike registado.")

    # ------------------------------------------------------------------
    # Volume
    # ------------------------------------------------------------------

    def set_volume(self, volume: float):
        """Volume entre 0.0 e 1.0."""
        pygame.mixer.music.set_volume(max(0.0, min(1.0, volume)))

    @property
    def volume(self) -> float:
        return pygame.mixer.music.get_volume()

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    @property
    def state(self) -> PlayerState:
        return self._state

    @property
    def current_track(self) -> Optional[TrackInfo]:
        return self._current

    @property
    def listen_time(self) -> float:
        """Segundos ouvidos na faixa atual."""
        if self._state == PlayerState.PLAYING:
            return self._paused_accum + (time.time() - self._start_time)
        return self._paused_accum

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _play_next(self):
        """Carrega e toca a próxima faixa da fila."""
        with self._lock:
            if not self._queue:
                self._state = PlayerState.STOPPED
                print("  [Fim da fila]")
                return
            track = self._queue.pop(0)

        if not os.path.exists(track.path):
            print(f"  [!] Ficheiro não encontrado: {track.path}. A saltar.")
            self._play_next()
            return

        self._current      = track
        self._signals      = PlaybackSignals(track_id=track.id, mood=track.mood)
        self._start_time   = time.time()
        self._paused_accum = 0.0
        self._state        = PlayerState.PLAYING

        pygame.mixer.music.load(track.path)
        pygame.mixer.music.play()

        print(f"\n  ▶  A tocar: [{track.mood.upper()}] {os.path.basename(track.path)}"
              f"  |  BPM={track.bpm:.0f}  density={track.density:.1f}"
              f"  complexity={track.complexity:.1f}")

        if self.on_track_start:
            self.on_track_start(track)

        # Inicia thread de monitorização do fim da faixa
        self._stop_monitor.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_playback, daemon=True
        )
        self._monitor_thread.start()

    def _monitor_playback(self):
        """Thread que deteta o fim natural da faixa."""
        while not self._stop_monitor.is_set():
            if self._state == PlayerState.PLAYING:
                if not pygame.mixer.music.get_busy():
                    # Faixa acabou naturalmente
                    self._finalise_signals(skipped=False)
                    self._play_next()
                    return
            time.sleep(0.2)

    def _finalise_signals(self, skipped: bool):
        """Regista o tempo ouvido e dispara o callback."""
        if self._signals is None:
            return

        self._signals.listen_time = self.listen_time
        self._signals.skipped     = skipped

        print(f"\n  📊  Sinais → {self._signals.to_dict()}")

        if self.on_track_end:
            self.on_track_end(self._signals)

        self._signals = None

    def __del__(self):
        try:
            pygame.mixer.quit()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Interface de linha de comandos (para testes locais)
# ---------------------------------------------------------------------------

def run_cli(midi_dir: str = "."):
    """
    CLI interativa para testar o player sem Streamlit.
    Procura ficheiros .mid na pasta indicada.
    """
    # Simula o sinal callback (em produção envia para o backend do L)
    collected_signals: list[dict] = []

    def on_end(signals: PlaybackSignals):
        collected_signals.append(signals.to_dict())

    player = MIDIPlayer(on_track_end=on_end)

    # Carrega ficheiros .mid da pasta
    midi_files = sorted([
        f for f in os.listdir(midi_dir) if f.endswith(".mid")
    ])

    if not midi_files:
        print(f"[!] Nenhum ficheiro .mid encontrado em '{midi_dir}'")
        return

    for i, fname in enumerate(midi_files):
        mood = "desconhecido"
        # Tenta inferir mood a partir do nome do ficheiro
        for m in ["happy", "sad", "calm", "energetic"]:
            if m in fname:
                mood = m
                break

        player.add_to_queue(TrackInfo(
            id=i,
            path=os.path.join(midi_dir, fname),
            mood=mood,
            bpm=120.0,
        ))

    print("=" * 55)
    print("  🎵  MIDI Player — Adaptive Music Experience")
    print("=" * 55)
    print(f"  {len(midi_files)} faixa(s) na fila: {', '.join(midi_files)}\n")
    print("  Comandos:")
    print("    p  → play / pause")
    print("    s  → skip para próxima")
    print("    +  → like")
    print("    -  → dislike")
    print("    v+ → volume +10%")
    print("    v- → volume -10%")
    print("    q  → sair")
    print("-" * 55)

    player.play()

    try:
        while True:
            cmd = input("  > ").strip().lower()

            if cmd == "p":
                if player.state == PlayerState.PLAYING:
                    player.pause()
                else:
                    player.play()

            elif cmd == "s":
                player.skip()

            elif cmd == "+":
                player.like()

            elif cmd == "-":
                player.dislike()

            elif cmd == "v+":
                player.set_volume(min(1.0, player.volume + 0.1))
                print(f"  🔊  Volume: {player.volume:.0%}")

            elif cmd == "v-":
                player.set_volume(max(0.0, player.volume - 0.1))
                print(f"  🔉  Volume: {player.volume:.0%}")

            elif cmd == "q":
                break

            elif cmd == "":
                # Mostra estado atual
                t = player.current_track
                if t:
                    print(f"  Estado: {player.state.name}"
                          f" | Faixa: {os.path.basename(t.path)}"
                          f" | Ouvido: {player.listen_time:.1f}s")

    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        player.stop()
        print("\n" + "=" * 55)
        print("  Sinais recolhidos (para o backend):")
        for s in collected_signals:
            print(f"    {s}")
        print("=" * 55)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Aceita pasta como argumento opcional: python midi_player.py ./midi_files
    midi_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    run_cli(midi_dir)