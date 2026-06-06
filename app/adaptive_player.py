"""
adaptive_player.py
------------------
Motor Adaptativo — Componente do João (J)
Projeto: Adaptive Music Experience (C2)

Liga o MIDIGenerator ao MIDIPlayer com Multi-Armed Bandit (Etapa 2):
  - Seleção de parâmetros musicais via Gaussian Thompson Sampling (bandit.py)
  - Espaço contínuo: BPM, density, complexity, octave, note_duration, velocity, scale
  - Logging de todas as decisões para avaliação quantitativa
  - Ablation study integrado: adaptativo vs baseline aleatório

Corre com:
    python adaptive_player.py

Dependências:
    pip install pygame pretty_midi numpy
"""

import os
import sys
import time
import threading
import numpy as np
import base64

from midi_generator import (
    MIDIGenerator, GeneratorParams, MOOD_CONFIGS, _VARIATION_ROOTS, _MOOD_INSTRUMENTS
)
from midi_player import MIDIPlayer, TrackInfo, PlaybackSignals, PlayerState
from bandit import (
    ContinuousGaussianBandit, RandomContinuousBandit,
    BanditLogger, BanditEvaluator, MusicParams
)


# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

OUTPUT_DIR = "adaptive_midi"  # onde guardar os .mid gerados
QUEUE_MIN  = 5               # gera novas quando a fila fica abaixo disto



# ---------------------------------------------------------------------------
# Motor adaptativo principal
# ---------------------------------------------------------------------------

class AdaptiveEngine:
    """
    Liga o MIDIGenerator ao MIDIPlayer com um loop de feedback adaptativo.

    Fluxo:
      1. Gera batch inicial (cold start)
      2. Reproduz faixas uma a uma
      3. A cada like/dislike/skip → calcula recompensa → atualiza estado →
         gera nova(s) faixa(s) → adiciona à fila do player
    """

    def __init__(
        self,
        output_dir: str = OUTPUT_DIR,
        seed: int = 42,
    ):
        self.rng        = np.random.default_rng(seed)
        self.generator  = MIDIGenerator(seed=seed)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # ── Bandit (Etapa 2) ─────────────────────────────────────────
        self.bandit = ContinuousGaussianBandit(seed=seed)
        self.logger = BanditLogger(self.bandit.name)

        # Baseline paralelo para o ablation study (sempre Random)
        self.baseline_bandit = RandomContinuousBandit(seed=seed + 1)
        self.baseline_logger = BanditLogger("RandomContinuous_baseline")
        # ─────────────────────────────────────────────────────────────

        self.history: list[dict] = []
        self._track_counter      = 0
        self._last_track_params: dict = {}

        self.player = MIDIPlayer(
            on_track_end=self._on_track_end,
            on_track_start=self._on_track_start,
        )
        self._lock = threading.Lock()


    def _compute_reward(self, signals: PlaybackSignals) -> float:
        """
        Função de recompensa (versão simplificada do BO do L):
          +1.0  like explícito
          -1.0  dislike explícito
          skip com <10s ouvidos → -0.5
          skip com >10s         → +0.1  (ouviu um bocado, talvez interessante)
          fim natural           → proporcional ao tempo ouvido (0 – 0.5)
        """
        if signals.liked:
            return 1.0
        if signals.disliked:
            return -1.0
        if signals.skipped:
            return -0.5 if signals.listen_time < 10 else 0.1
        # Fim natural: recompensa pelo tempo ouvido (assumindo ~60s por faixa)
        return min(0.5, signals.listen_time / 120.0)

    def process_api_feedback(self, mood: str, track_params: dict, feedback_value: str):
        """Nova função para processar feedback vindo dos endpoints da API."""
        if feedback_value == "like":
            reward = 1.0
        elif feedback_value == "dislike":
            reward = -1.0
        elif feedback_value == "skip":
            reward = -0.5
        else:
            reward = 0.5  # Default/fim natural

        played_params = MusicParams(
            bpm        = track_params.get("bpm",        85.0),
            density    = track_params.get("density",    0.5),
            complexity = track_params.get("complexity", 0.5),
        )

        with self._lock:
            # 1. Atualizar o Bandit
            self.bandit.update(played_params, reward)

            # 2. Guardar log
            self.logger.log(self.bandit, played_params, reward,
                            getattr(self, "_last_extra", {}))
            self.history.append({
                "mood": mood, "reward": round(reward, 3),
                "api_feedback": feedback_value,
            })

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def _run_cli(self):
        print("  Comandos:")
        # ... continuação do código original ...




    def start(self):
        """Gera batch inicial e arranca a reprodução."""
        print("\n  🎵  Adaptive Music Engine — a iniciar...\n")
        self._generate_cold_start()
        self.player.play()
        self._run_cli()

    # ------------------------------------------------------------------
    # Geração de faixas
    # ------------------------------------------------------------------

    def _generate_cold_start(self):
        """Uma faixa por mood para arrancar sem dados."""
        print("  [cold start] A gerar uma faixa por mood...\n")
        for mood in ["happy", "sad", "calm", "energetic"]:
            track = self._generate_track_for_mood(mood)
            self.player.add_to_queue(track)

    def _generate_track_for_mood(self, mood: str) -> TrackInfo:
        """Gera uma faixa para o mood dado."""
        cfg            = MOOD_CONFIGS[mood]
        bpm_lo, bpm_hi = cfg["bpm_range"]
        idx            = self._track_counter
        root           = str(self.rng.choice(_VARIATION_ROOTS))
        instrument     = int(self.rng.choice(_MOOD_INSTRUMENTS[mood]))
        bpm            = float(self.rng.uniform(bpm_lo, bpm_hi))

        params = GeneratorParams(
            mood=mood,
            bpm=bpm,
            density=cfg["density"],
            complexity=cfg["complexity"],
            repetition=0.3,
            duration_bars=int(self.rng.choice([8, 12, 16])),
            root_note=root,
            instrument_program=instrument,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"{mood}_{idx:04d}_{root}_{int(bpm)}bpm.mid"
        path     = os.path.join(self.output_dir, filename)
        self.generator.generate(params, output_path=path)

        self._track_counter += 1

        with open(path, "rb") as midi_file:
            encoded_midi = base64.b64encode(midi_file.read()).decode("utf-8")

        return TrackInfo(
            id=idx, path=path, mood=mood,
            bpm=bpm, density=cfg["density"], complexity=cfg["complexity"], base64_file=encoded_midi,
        )

    def _generate_track(self, music_params: MusicParams) -> TrackInfo:
        """Gera uma faixa a partir de um MusicParams do bandit e devolve um TrackInfo."""
        p   = music_params.to_generator_dict()
        idx = self._track_counter

        root       = str(self.rng.choice(_VARIATION_ROOTS))

        mood       = "calm"   # fallback neutro para _MOOD_INSTRUMENTS
        instrument = int(self.rng.choice(_MOOD_INSTRUMENTS[mood]))

        bpm_lo, bpm_hi = p["bpm_range"]
        bpm = float(self.rng.uniform(bpm_lo, bpm_hi))

        params = GeneratorParams(
            mood=mood,
            bpm=bpm,
            density=p["density"],
            complexity=p["complexity"],
            repetition=0.3,
            duration_bars=int(self.rng.choice([8, 12, 16])),
            root_note=root,
            instrument_program=instrument,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        filename = f"track_{idx:04d}_{root}_{int(bpm)}bpm_{music_params.scale}.mid"
        path     = os.path.join(self.output_dir, filename)
        self.generator.generate(params, output_path=path)

        self._track_counter += 1

        with open(path, "rb") as midi_file:
            encoded_midi = base64.b64encode(midi_file.read()).decode("utf-8")
        print("encoded_midi", encoded_midi)

        return TrackInfo(
            id=idx, path=path, mood=mood,
            bpm=bpm, density=p["density"], complexity=p["complexity"], base64_file=encoded_midi,
        )

    def _pick_params(self) -> tuple[MusicParams, dict]:
        """Delega a seleção de parâmetros ao bandit contínuo."""
        return self.bandit.select_params()

    def _maybe_fill_queue(self):
        """Se a fila estiver curta, gera mais uma faixa adaptada."""
        if self.player.queue_length < QUEUE_MIN:
            music_params, extra = self._pick_params()
            track               = self._generate_track(music_params)
            self.player.add_to_queue(track)
            print(f"\n  🎲  [{self.bandit.name}] Nova faixa → BPM≈{music_params.bpm:.0f}"
                  f"  density={music_params.density:.2f}"
                  f"  scale={music_params.scale}"
                  f"  {os.path.basename(track.path)}")

    # ------------------------------------------------------------------
    # Callbacks do player
    # ------------------------------------------------------------------
    def get_pre_generated_batch(self) -> list[dict]:
        """
        Recolhe as músicas pré-geradas para enviar para o frontend.
        Procura estritamente na pasta 'start_tracks' relativa a este script.
        """
        tracks_data = []
        
        # 1. Descobre a pasta exata onde este script (adaptive_player.py) está
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print(base_dir)
        
        # 2. Constrói o caminho fixo para a pasta "start_tracks"
        target_dir = os.path.join(base_dir, "start_tracks")
        if not os.path.exists(target_dir):
            print(f"  [!] Diretório '{target_dir}' não encontrado.")
            return tracks_data

        track_id = 0
        
        # Percorre o diretório à procura dos ficheiros .mid
        for root_dir, _, files in os.walk(target_dir):
            for file in sorted(files): # Garante que são lidos sempre na mesma ordem
                if file.endswith(".mid"):
                    path = os.path.join(root_dir, file)
                    
                    # 1. Inferir o mood pelo nome do ficheiro (ex: calm_0002_E_91bpm.mid)
                    mood = "desconhecido"
                    for m in ["happy", "sad", "calm", "energetic"]:
                        if m in file.lower():
                            mood = m
                            break

                    # 2. Extrair o BPM pelo nome do ficheiro
                    bpm = 120.0
                    try:
                        for part in file.split("_"):
                            if "bpm" in part:
                                bpm = float(part.replace("bpm.mid", ""))
                    except ValueError:
                        pass

                    with open(path, "rb") as midi_file:
                        encoded_midi = base64.b64encode(midi_file.read()).decode("utf-8")

                    # 3. Construir o payload para o frontend
                    tracks_data.append({
                        "id": track_id,
                        "path": path,
                        "filename": file,
                        "mood": mood,
                        "bpm": bpm,
                        "base64_file": encoded_midi,
                        "source": "pre_generated"
                    })
                    track_id += 1
                        
        print(f"  [✓] Foram prontas {len(tracks_data)} músicas da pasta start_tracks.")
        return tracks_data
    
    def _on_track_start(self, track: TrackInfo):
        self._last_track_params = {
            "bpm":        track.bpm,
            "density":    track.density,
            "complexity": track.complexity,
        }
        # Pré-seleciona os próximos parâmetros e guarda o extra para o logger
        try:
            _next_params, self._last_extra = self.bandit.select_params()
        except Exception:
            self._last_extra = {}

    def _on_track_end(self, signals: PlaybackSignals):
        """Calcula recompensa, actualiza bandit, loga e gera nova faixa."""
        reward = self._compute_reward(signals)

        # Reconstrói um MusicParams mínimo a partir dos parâmetros da faixa terminada
        last = self._last_track_params
        played_params = MusicParams(
            bpm        = last.get("bpm",        85.0),
            density    = last.get("density",    0.5),
            complexity = last.get("complexity", 0.5),
        )

        with self._lock:
            # 1. Atualiza o bandit adaptativo e loga a decisão
            self.bandit.update(played_params, reward)
            self.logger.log(self.bandit, played_params, reward,
                            getattr(self, "_last_extra", {}))

            # 2. Atualiza o baseline em paralelo (mesmos parâmetros, mesma reward)
            self.baseline_bandit.update(played_params, reward)
            self.baseline_logger.log(self.baseline_bandit, played_params, reward, {})

            record = {**signals.to_dict(), "reward": round(reward, 3)}
            self.history.append(record)

        self._print_feedback_summary(signals, reward)
        self._maybe_fill_queue()


    def _compute_reward(self, signals: PlaybackSignals) -> float:
        """
        Função de recompensa (versão simplificada do BO do L):
          +1.0  like explícito
          -1.0  dislike explícito
          skip com <10s ouvidos → -0.5
          skip com >10s         → +0.1  (ouviu um bocado, talvez interessante)
          fim natural           → proporcional ao tempo ouvido (0 – 0.5)
        """
        if signals.liked:
            return 1.0
        if signals.disliked:
            return -1.0
        if signals.skipped:
            return -0.5 if signals.listen_time < 10 else 0.1
        # Fim natural: recompensa pelo tempo ouvido (assumindo ~60s por faixa)
        return min(0.5, signals.listen_time / 120.0)

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def _run_cli(self):
        print("  Comandos:")
        print("    p   → play / pause")
        print("    s   → skip")
        print("    +   → 👍 like  (gera música parecida)")
        print("    -   → 👎 dislike  (gera música diferente)")
        print("    v+  → volume +")
        print("    v-  → volume -")
        print("    st  → mostrar estado atual")
        print("    h   → histórico de feedback")
        print("    q   → sair")
        print("  " + "-" * 50)

        try:
            while True:
                cmd = input("  > ").strip().lower()

                if cmd == "p":
                    if self.player.state == PlayerState.PLAYING:
                        self.player.pause()
                    else:
                        self.player.play()

                elif cmd == "s":
                    self.player.skip()

                elif cmd == "+":
                    self.player.like()
                    # Reage imediatamente: gera já uma faixa com parâmetros semelhantes
                    music_params, _ = self._pick_params()
                    track = self._generate_track(music_params)
                    self.player.add_to_queue(track)
                    print(f"\n  ✨  Like! Nova faixa adicionada:"
                          f" {os.path.basename(track.path)}")

                elif cmd == "-":
                    self.player.dislike()
                    # Reage imediatamente: deixa o bandit explorar uma zona diferente
                    music_params, _ = self._pick_params()
                    track = self._generate_track(music_params)
                    self.player.add_to_queue(track)
                    print(f"\n  🔄  Dislike! Nova faixa adicionada:"
                          f" {os.path.basename(track.path)}")

                elif cmd == "v+":
                    self.player.set_volume(min(1.0, self.player.volume + 0.1))
                    print(f"  🔊  Volume: {self.player.volume:.0%}")

                elif cmd == "v-":
                    self.player.set_volume(max(0.0, self.player.volume - 0.1))
                    print(f"  🔉  Volume: {self.player.volume:.0%}")

                elif cmd == "st":
                    self._print_state()

                elif cmd == "h":
                    self._print_history()

                elif cmd == "q":
                    break

        except (KeyboardInterrupt, EOFError):
            pass
        finally:
            self.player.stop()
            print("\n  📊  Sessão terminada. Resumo final:")
            self._print_state()

    # ------------------------------------------------------------------
    # Display helpers
    # ------------------------------------------------------------------

    def _print_feedback_summary(self, signals: PlaybackSignals, reward: float):
        emoji = "👍" if signals.liked else ("👎" if signals.disliked else
                ("⏭" if signals.skipped else "✅"))
        stats = self.bandit.arm_stats()["params"]
        print(f"\n  {emoji}  reward={reward:+.2f}"
              f" | BPM≈{stats['bpm']['mu']:.0f}"
              f"  dens≈{stats['density']['mu']:.2f}"
              f"  cmplx≈{stats['complexity']['mu']:.2f}"
              f"  (step={self.bandit.step})")

    def _print_state(self):
        """Mostra estado do bandit — médias e incertezas por parâmetro."""
        stats = self.bandit.arm_stats()
        print(f"\n  Algoritmo: {self.bandit.name}  (step={self.bandit.step})")
        print(f"  {'PARÂMETRO':<18} {'μ (média)':>10} {'σ (incert.)':>12} {'plays':>7}")
        print("  " + "-" * 52)
        for name, s in stats["params"].items():
            print(f"  {name:<18} {s['mu']:>10.3f} {s['std']:>12.4f} {s['plays']:>7}")
        print(f"\n  Escalas (prob. preferida):")
        for scale, info in stats["scales"].items():
            bar = "█" * int(info["mean"] * 20)
            print(f"    {scale:<22} {bar:<20} {info['mean']:.2f}")
        print(f"\n  Fila: {self.player.queue_length} faixa(s)"
              f"  |  Geradas: {self._track_counter}"
              f"  |  Reward acumulada: {self.bandit.cumulative_reward:+.3f}")


    def _print_history(self):
        if not self.history:
            print("  (sem histórico ainda)")
            return
        print(f"\n  {'#':>3} {'mood':<12} {'like':>5} {'dislike':>8}"
              f" {'skip':>5} {'time':>7} {'reward':>8}")
        print("  " + "-" * 52)
        for i, h in enumerate(self.history[-15:]):  # últimas 15
            print(f"  {i+1:>3} {h['mood']:<12}"
                  f" {'✓' if h['liked'] else ' ':>5}"
                  f" {'✓' if h['disliked'] else ' ':>8}"
                  f" {'✓' if h['skipped'] else ' ':>5}"
                  f" {h['listen_time']:>6.1f}s"
                  f" {h['reward']:>+8.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = AdaptiveEngine(output_dir="adaptive_midi", seed=42)
    engine.start()

    # Ablation study automático no fim da sessão
    if len(engine.history) >= 3:
        evaluator = BanditEvaluator(engine.logger, engine.baseline_logger)
        evaluator.print_report()
        evaluator.to_json("ablation_report.json")
        engine.logger.to_json("log_gaussian_ts.json")