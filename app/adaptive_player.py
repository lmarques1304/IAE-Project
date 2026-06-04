"""
adaptive_player.py
------------------
Motor Adaptativo — Componente do João (J)
Projeto: Adaptive Music Experience (C2)

Liga o MIDIGenerator ao MIDIPlayer com Multi-Armed Bandit (Etapa 2):
  - Seleção de mood via UCB1 ou Thompson Sampling (bandit.py)
  - Parâmetros contínuos (BPM, density, complexity) adaptados por gradiente
  - Logging de todas as decisões para avaliação quantitativa
  - Ablation study integrado: adaptativo vs baseline aleatório

Corre com:
    python adaptive_player.py                  # Thompson Sampling (default)
    python adaptive_player.py --algo ucb1      # UCB1
    python adaptive_player.py --algo random    # Baseline

Dependências:
    pip install pygame pretty_midi numpy
"""

import os
import sys
import time
import threading
import numpy as np

from midi_generator import (
    MIDIGenerator, GeneratorParams, MOOD_CONFIGS, _VARIATION_ROOTS, _MOOD_INSTRUMENTS
)
from midi_player import MIDIPlayer, TrackInfo, PlaybackSignals, PlayerState
from bandit import (
    UCB1Bandit, ThompsonSamplingBandit, RandomBandit,
    BaseBandit, BanditLogger, BanditEvaluator, MOODS
)


# ---------------------------------------------------------------------------
# Configuração global
# ---------------------------------------------------------------------------

OUTPUT_DIR    = "adaptive_midi"  # onde guardar os .mid gerados
QUEUE_MIN     = 5                # gera novas quando a fila fica abaixo disto
LEARNING_RATE = 0.15             # quão depressa os parâmetros se adaptam


# ---------------------------------------------------------------------------
# Estado adaptativo por mood — parâmetros contínuos apenas
# (estatísticas de recompensa e seleção de arm passaram para bandit.py)
# ---------------------------------------------------------------------------

class MoodState:
    """
    Mantém os parâmetros contínuos ideais por mood (BPM, density, complexity).
    Atualiza-os por gradiente após cada recompensa.
    A seleção de qual mood tocar a seguir é feita pelo bandit (bandit.py).
    """

    def __init__(self, mood: str, rng: np.random.Generator):
        cfg = MOOD_CONFIGS[mood]
        bpm_lo, bpm_hi = cfg["bpm_range"]
        self.mood       = mood
        self.rng        = rng
        self.bpm        = float(np.mean([bpm_lo, bpm_hi]))
        self.density    = cfg["density"]
        self.complexity = cfg["complexity"]
        self.repetition = 0.3
        self._bpm_lo    = bpm_lo
        self._bpm_hi    = bpm_hi

    def update_params(self, reward: float, track_params: dict):
        """
        Ajusta os parâmetros na direção da faixa que gerou esta recompensa.
        reward > 0 → aproxima parâmetros dos da faixa
        reward < 0 → afasta parâmetros dos da faixa
        """
        lr = LEARNING_RATE * abs(reward)
        self.bpm = float(np.clip(
            self.bpm + lr * (track_params["bpm"] - self.bpm),
            self._bpm_lo, self._bpm_hi
        ))
        self.density = float(np.clip(
            self.density + lr * (track_params["density"] - self.density),
            0.1, 1.0
        ))
        self.complexity = float(np.clip(
            self.complexity + lr * (track_params["complexity"] - self.complexity),
            0.0, 1.0
        ))

    def sample_params(self) -> dict:
        """Amostra parâmetros com ruído gaussiano para garantir diversidade."""
        noise = lambda v, s: float(np.clip(self.rng.normal(v, s), 0.05, 1.0))
        return {
            "bpm":        float(np.clip(self.rng.normal(self.bpm, 8.0),
                                        self._bpm_lo, self._bpm_hi)),
            "density":    noise(self.density,    0.08),
            "complexity": noise(self.complexity, 0.08),
            "repetition": float(np.clip(self.rng.normal(self.repetition, 0.05),
                                        0.1, 0.7)),
        }


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
        algo: str = "thompson",   # "thompson" | "ucb1" | "random"
        seed: int = 42,
    ):
        self.rng        = np.random.default_rng(seed)
        self.generator  = MIDIGenerator(seed=seed)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Parâmetros contínuos por mood (BPM, density, complexity)
        self.mood_states: dict[str, MoodState] = {
            m: MoodState(m, self.rng) for m in MOODS
        }

        # ── Bandit (Etapa 2) ─────────────────────────────────────────
        self.bandit: BaseBandit = self._build_bandit(algo, seed)
        self.logger             = BanditLogger(self.bandit.name)

        # Baseline paralelo para o ablation study (sempre Random)
        self.baseline_bandit = RandomBandit(MOODS, seed=seed + 1)
        self.baseline_logger = BanditLogger("Random_baseline")
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
        # Mapear o feedback do frontend para a recompensa do Bandit
        if feedback_value == "like":
            reward = 1.0
        elif feedback_value == "dislike":
            reward = -1.0
        elif feedback_value == "skip":
            reward = -0.5
        else:
            reward = 0.5 # Default/fim natural
            
        with self._lock:
            # 1. Atualizar o Bandit
            self.bandit.update(mood, reward)
            
            # 2. Atualizar parâmetros contínuos
            self.mood_states[mood].update_params(reward, track_params)
            
            # 3. Guardar log
            self.logger.log(self.bandit, mood, reward, getattr(self, "_last_extra", {}))
            self.history.append({"mood": mood, "reward": round(reward, 3), "api_feedback": feedback_value})

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------

    def _run_cli(self):
        print("  Comandos:")
        # ... continuação do código original ...

    @staticmethod
    def _build_bandit(algo: str, seed: int) -> BaseBandit:
        algo = algo.lower()
        if algo == "ucb1":
            return UCB1Bandit(MOODS, c=1.5, seed=seed)
        if algo == "random":
            return RandomBandit(MOODS, seed=seed)
        return ThompsonSamplingBandit(MOODS, seed=seed)  # default

    # ------------------------------------------------------------------
    # Arranque
    # ------------------------------------------------------------------

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
        for mood in MOODS:
            track = self._generate_track(mood)
            self.player.add_to_queue(track)

    def _generate_track(self, mood: str) -> TrackInfo:
        """Gera uma faixa para o mood dado e devolve um TrackInfo."""
        state  = self.mood_states[mood]
        p      = state.sample_params()
        idx    = self._track_counter

        root       = str(self.rng.choice(_VARIATION_ROOTS))
        instrument = int(self.rng.choice(_MOOD_INSTRUMENTS[mood]))

        params = GeneratorParams(
            mood=mood,
            bpm=p["bpm"],
            density=p["density"],
            complexity=p["complexity"],
            repetition=p["repetition"],
            duration_bars=int(self.rng.choice([8, 12, 16])),
            root_note=root,
            instrument_program=instrument,
        )

        mood_dir = self.output_dir
        os.makedirs(mood_dir, exist_ok=True)
        filename = f"{mood}_{idx:04d}_{root}_{int(p['bpm'])}bpm.mid"
        path     = os.path.join(mood_dir, filename)
        self.generator.generate(params, output_path=path)

        self._track_counter += 1
        return TrackInfo(
            id=idx, path=path, mood=mood,
            bpm=p["bpm"], density=p["density"], complexity=p["complexity"],
        )

    def _pick_mood(self) -> tuple[str, dict]:
        """Delega a seleção de mood ao bandit configurado."""
        return self.bandit.select_arm()

    def _maybe_fill_queue(self):
        """Se a fila estiver curta, gera mais uma faixa adaptada."""
        if self.player.queue_length < QUEUE_MIN:
            mood, extra = self._pick_mood()
            track       = self._generate_track(mood)
            self.player.add_to_queue(track)
            print(f"\n  🎲  [{self.bandit.name}] Nova faixa → [{mood.upper()}]"
                  f" {os.path.basename(track.path)}")

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

                    # 3. Construir o payload para o frontend
                    tracks_data.append({
                        "id": track_id,
                        "path": path,
                        "filename": file,
                        "mood": mood,
                        "bpm": bpm,
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
        # Pré-seleciona o próximo mood e guarda o extra para o logger
        try:
            _next_mood, self._last_extra = self.bandit.select_arm()
        except Exception:
            self._last_extra = {}

    def _on_track_end(self, signals: PlaybackSignals):
        """Calcula recompensa, actualiza bandit + parâmetros, loga e gera nova faixa."""
        reward = self._compute_reward(signals)
        mood   = signals.mood

        with self._lock:
            # 1. Atualiza o bandit adaptativo e loga a decisão
            self.bandit.update(mood, reward)
            mood_extra = {}  # extra vem do select_arm anterior (guardado no _last_extra)
            self.logger.log(self.bandit, mood, reward,
                            getattr(self, "_last_extra", {}))

            # 2. Atualiza o baseline em paralelo (mesmo mood, mesma reward)
            self.baseline_bandit.update(mood, reward)
            self.baseline_logger.log(self.baseline_bandit, mood, reward, {})

            # 3. Ajusta parâmetros contínuos do mood
            self.mood_states[mood].update_params(reward, self._last_track_params)

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
                    # Reage imediatamente: gera já uma faixa do mesmo mood
                    current = self.player.current_track
                    if current:
                        mood  = current.mood
                        track = self._generate_track(mood)
                        self.player.add_to_queue(track)
                        print(f"\n  ✨  Like! Nova faixa [{mood.upper()}] adicionada:"
                              f" {os.path.basename(track.path)}")

                elif cmd == "-":
                    self.player.dislike()
                    # Reage imediatamente: gera uma faixa de mood diferente
                    current = self.player.current_track

                    if current:
                        other_moods = [m for m in MOODS if m != current.mood]
                        mood, _extra = self._pick_mood()
                        track = self._generate_track(mood)
                        self.player.add_to_queue(track)
                        
                        print(f"\n  🔄  Dislike! Nova faixa [{mood.upper()}] adicionada:"
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
        state = self.mood_states[signals.mood]
        arm   = self.bandit.arms[signals.mood]
        print(f"\n  {emoji}  reward={reward:+.2f}"
              f" | [{signals.mood}] BPM≈{state.bpm:.0f}"
              f"  dens≈{state.density:.2f}"
              f"  cmplx≈{state.complexity:.2f}"
              f"  (plays={arm.plays}  μ={arm.mean_reward:+.3f})")

    def _print_state(self):
        """Mostra estado do bandit + parâmetros contínuos por mood."""
        print(f"\n  Algoritmo: {self.bandit.name}")
        print(f"  {'MOOD':<12} {'plays':>6} {'μ reward':>9} {'BPM':>7}"
              f" {'density':>8} {'complexity':>11} {'α':>6} {'β':>6}")
        print("  " + "-" * 70)
        for mood in MOODS:
            arm   = self.bandit.arms[mood]
            state = self.mood_states[mood]
            print(f"  {mood:<12} {arm.plays:>6} {arm.mean_reward:>+9.3f}"
                  f" {state.bpm:>7.1f} {state.density:>8.3f}"
                  f" {state.complexity:>11.3f} {arm.alpha:>6.1f} {arm.beta:>6.1f}")
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
    algo = "thompson"
    if "--algo" in sys.argv:
        idx  = sys.argv.index("--algo")
        algo = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else "thompson"

    engine = AdaptiveEngine(output_dir="adaptive_midi", algo=algo, seed=42)
    engine.start()

    # Ablation study automático no fim da sessão
    if len(engine.history) >= 3:
        evaluator = BanditEvaluator(engine.logger, engine.baseline_logger)
        evaluator.print_report()
        evaluator.to_json("ablation_report.json")
        engine.logger.to_json(f"log_{algo}.json")