"""
bandit.py
---------
Multi-Armed Bandit — espaço de parâmetros contínuos
Projeto: Adaptive Music Experience (C2)

Implementa:
  - ContinuousGaussianBandit — Thompson Sampling sobre parâmetros contínuos
        Cada parâmetro (bpm, density, …) tem uma distribuição Normal independente
        que é actualizada com cada reward via Bayesian update (Normal-Normal).
  - RandomContinuousBandit   — baseline uniforme nos mesmos ranges (sem aprendizagem)
  - BanditLogger             — logging quantitativo de todas as decisões
  - BanditEvaluator          — ablation study: adaptativo vs baseline

Integração com adaptive_player.py:
    from bandit import ContinuousGaussianBandit, BanditLogger, MusicParams

    bandit = ContinuousGaussianBandit()
    logger = BanditLogger("GaussianTS")

    params, extra = bandit.select_params()      # escolhe parâmetros
    reward = compute_reward(signals)
    bandit.update(params, reward)
    logger.log(bandit, params, reward, extra)

Dependências:
    pip install numpy
"""

import json
import time
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Definição do espaço de parâmetros
# ---------------------------------------------------------------------------

def _infer_mood(bpm: float, density: float, complexity: float, scale: str) -> str:
    """
    Mapeia os parâmetros escolhidos pelo bandit num mood legível.

    Regras (por ordem de prioridade):
      energetic — bpm alto E densidade alta
      happy     — bpm moderado-alto, escala maior/mixolídio
      sad       — escala menor/dórico OU bpm baixo + densidade baixa
      calm      — tudo o resto (bpm baixo, pouca densidade)
    """
    is_minor_scale = scale in ("minor", "dorian")
    is_major_scale = scale in ("major", "mixolydian", "pentatonic_major")

    if bpm >= 115 and density >= 0.55:
        return "energetic"
    if bpm >= 95 and density >= 0.4 and is_major_scale:
        return "happy"
    if is_minor_scale or (bpm < 85 and density < 0.35):
        return "sad"
    return "calm"


@dataclass
class MusicParams:
    """
    Parâmetros contínuos de geração musical escolhidos pelo bandit.
    Todos os valores são escalares prontos a passar ao gerador.
    """
    scale:               str   = "major"
    bpm:                 float = 85.0
    density:             float = 0.3
    complexity:          float = 0.2
    octave:              float = 4.5    # valor contínuo; arredonda-se ao usar
    note_duration:       float = 0.7
    velocity:            float = 60.0

    @property
    def mood(self) -> str:
        """Mood inferido automaticamente a partir dos parâmetros escolhidos."""
        return _infer_mood(self.bpm, self.density, self.complexity, self.scale)

    def to_dict(self) -> dict:
        return {
            "mood":          self.mood,
            "scale":         self.scale,
            "bpm":           round(self.bpm, 1),
            "density":       round(self.density, 3),
            "complexity":    round(self.complexity, 3),
            "octave":        round(self.octave, 2),
            "note_duration": round(self.note_duration, 3),
            "velocity":      round(self.velocity, 1),
        }

    def to_generator_dict(self) -> dict:
        """Formato pronto a passar ao engine de geração MIDI."""
        return {
            "scale":               self.scale,
            "bpm_range":           (max(60, self.bpm - 5), self.bpm + 5),
            "density":             round(self.density, 3),
            "complexity":          round(self.complexity, 3),
            "octave_range":        (max(2, int(self.octave)), min(7, int(self.octave) + 1)),
            "note_duration_range": (
                max(0.1, self.note_duration - 0.15),
                self.note_duration + 0.15,
            ),
            "velocity_range":      (
                max(20, int(self.velocity) - 10),
                min(127, int(self.velocity) + 10),
            ),
        }


# ---------------------------------------------------------------------------
# Espaço de busca — ranges e escala de cada parâmetro
# ---------------------------------------------------------------------------

# Cada entrada: (min, max, prior_mean, prior_std)
# prior_mean = centro do range; prior_std = range/4 (cobre ~95% do espaço)
PARAM_SPACE: dict[str, tuple[float, float, float, float]] = {
    "bpm":           (70.0,  140.0,  100.0,  17.5),
    "density":       (0.1,   0.9,    0.5,    0.2),
    "complexity":    (0.1,   0.9,    0.5,    0.2),
    "octave":        (3.0,   6.0,    4.5,    0.75),
    "note_duration": (0.15,  1.2,    0.65,   0.26),
    "velocity":      (30.0,  110.0,  70.0,   20.0),
}

SCALES = ["major", "minor", "dorian", "mixolydian", "pentatonic_major", "pentatonic_minor"]


# ---------------------------------------------------------------------------
# Estrutura de dados
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """Registo imutável de uma decisão do bandit."""
    step:              int
    timestamp:         float
    algorithm:         str
    params:            dict
    reward:            float
    cumulative_reward: float
    arm_stats:         dict
    extra:             dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Braço Gaussiano — um por parâmetro contínuo
# ---------------------------------------------------------------------------

class GaussianArm:
    """
    Representa a distribuição de um parâmetro contínuo.

    Usa um Bayesian update Normal-Normal (variância conhecida):
        posterior_mean = (prior_mean/prior_var + obs_sum/noise_var)
                         / (1/prior_var + n/noise_var)
        posterior_var  = 1 / (1/prior_var + n/noise_var)

    Interpretação:
      - mu  → estimativa actual do valor óptimo do parâmetro
      - var → incerteza; encolhe à medida que acumulamos dados
      - noise_var → variância do reward (quanto o utilizador é inconsistente)
    """

    NOISE_VAR = 0.15   # variância assumida do reward — ajusta se necessário

    def __init__(self, name: str, lo: float, hi: float, prior_mean: float, prior_std: float):
        self.name      = name
        self.lo        = lo
        self.hi        = hi
        self.mu        = prior_mean
        self.var       = prior_std ** 2
        self._obs_sum  = 0.0   # Σ (reward * param_value) — suficiente estatística
        self._n        = 0

    def sample(self, rng: np.random.Generator) -> float:
        """Thompson Sampling: amostra da posterior e clipa ao range."""
        value = float(rng.normal(self.mu, np.sqrt(self.var)))
        return float(np.clip(value, self.lo, self.hi))

    def update(self, value: float, reward: float):
        """
        Bayesian update: a observação é (value, reward).
        Interpretamos reward como sinal de quão bom foi este valor.
        Actualizamos a média posterior na direcção de 'value' ponderada pelo reward.

        Formulação simplificada mas eficaz:
          - reward > 0 → puxa a média em direcção ao valor usado
          - reward < 0 → afasta a média do valor usado
          - magnitude  → velocidade de aprendizagem
        """
        self._n       += 1
        self._obs_sum += reward * value

        # Actualização da variância posterior
        precision_prior = 1.0 / self.var
        precision_data  = self._n / self.NOISE_VAR
        self.var        = 1.0 / (precision_prior + precision_data)

        # Actualização da média posterior
        prior_contrib = precision_prior * self.mu
        data_contrib  = (self._obs_sum / self.NOISE_VAR)
        self.mu       = self.var * (prior_contrib + data_contrib)
        self.mu       = float(np.clip(self.mu, self.lo, self.hi))

    def to_dict(self) -> dict:
        return {
            "name":  self.name,
            "mu":    round(self.mu,   4),
            "std":   round(float(np.sqrt(self.var)), 4),
            "plays": self._n,
        }


# ---------------------------------------------------------------------------
# Braço discreto para a escala musical
# ---------------------------------------------------------------------------

class ScaleArm:
    """
    Thompson Sampling Beta para a escala (variável categórica).
    Cada escala tem o seu par (alpha, beta) — idêntico ao BanditArm original.
    """

    def __init__(self, scales: list[str]):
        self.scales = scales
        self.alpha  = {s: 1.0 for s in scales}
        self.beta   = {s: 1.0 for s in scales}

    def sample(self, rng: np.random.Generator) -> str:
        samples = {s: float(rng.beta(self.alpha[s], self.beta[s])) for s in self.scales}
        return max(samples, key=samples.get)

    def update(self, scale: str, reward: float):
        r_norm = (reward + 1.0) / 2.0       # [-1,1] → [0,1]
        self.alpha[scale] += r_norm
        self.beta[scale]  += (1.0 - r_norm)

    def to_dict(self) -> dict:
        return {
            s: {
                "alpha": round(self.alpha[s], 3),
                "beta":  round(self.beta[s],  3),
                "mean":  round(self.alpha[s] / (self.alpha[s] + self.beta[s]), 3),
            }
            for s in self.scales
        }


# ---------------------------------------------------------------------------
# Bandit principal — espaço contínuo
# ---------------------------------------------------------------------------

class ContinuousGaussianBandit:
    """
    Gaussian Thompson Sampling sobre o espaço de parâmetros musicais.

    Cada parâmetro numérico tem um GaussianArm independente.
    A escala musical tem um ScaleArm (Beta por categoria).

    Após cada faixa:
        params, extra = bandit.select_params()
        # ... gera e reproduz a faixa ...
        bandit.update(params, reward)

    O bandit converge gradualmente para a zona do espaço que maximiza o reward,
    mantendo exploração via amostras da posterior.
    """

    name = "GaussianTS"

    def __init__(self, seed: Optional[int] = None):
        self.rng    = np.random.default_rng(seed)
        self.step   = 0
        self.cumulative_reward = 0.0

        # Um GaussianArm por parâmetro contínuo
        self.param_arms: dict[str, GaussianArm] = {
            name: GaussianArm(name, lo, hi, mu, std)
            for name, (lo, hi, mu, std) in PARAM_SPACE.items()
        }

        # ScaleArm para a variável categórica
        self.scale_arm = ScaleArm(SCALES)

    # ------------------------------------------------------------------
    # Interface principal
    # ------------------------------------------------------------------

    def select_params(self) -> tuple[MusicParams, dict]:
        """
        Amostra um conjunto de parâmetros musicais da posterior actual.
        Devolve (MusicParams, info_extra) — o extra é logado para análise.
        """
        sampled: dict[str, float] = {
            name: arm.sample(self.rng)
            for name, arm in self.param_arms.items()
        }
        scale = self.scale_arm.sample(self.rng)

        params = MusicParams(
            scale         = scale,
            bpm           = sampled["bpm"],
            density       = sampled["density"],
            complexity    = sampled["complexity"],
            octave        = sampled["octave"],
            note_duration = sampled["note_duration"],
            velocity      = sampled["velocity"],
        )

        extra = {
            "inferred_mood":   params.mood,
            "posterior_means": {n: round(a.mu, 4) for n, a in self.param_arms.items()},
            "posterior_stds":  {n: round(float(np.sqrt(a.var)), 4) for n, a in self.param_arms.items()},
            "scale_probs":     {
                s: round(self.scale_arm.alpha[s] /
                         (self.scale_arm.alpha[s] + self.scale_arm.beta[s]), 3)
                for s in SCALES
            },
        }
        return params, extra

    def update(self, params: MusicParams, reward: float):
        """Regista o reward e actualiza todas as distribuições posteriores."""
        self.param_arms["bpm"].update(params.bpm, reward)
        self.param_arms["density"].update(params.density, reward)
        self.param_arms["complexity"].update(params.complexity, reward)
        self.param_arms["octave"].update(params.octave, reward)
        self.param_arms["note_duration"].update(params.note_duration, reward)
        self.param_arms["velocity"].update(params.velocity, reward)
        self.scale_arm.update(params.scale, reward)

        self.cumulative_reward += reward
        self.step              += 1

    def arm_stats(self) -> dict:
        return {
            "params": {n: a.to_dict() for n, a in self.param_arms.items()},
            "scales": self.scale_arm.to_dict(),
        }

    def reset(self):
        self.__init__(seed=None)

    # ------------------------------------------------------------------
    # Exploração forçada (cold start)
    # ------------------------------------------------------------------

    def cold_start_params(self, n: int = 8) -> list[MusicParams]:
        """
        Gera n conjuntos de parâmetros diversificados para cold start.
        Usa Latin Hypercube Sampling para cobrir o espaço uniformemente
        sem depender de feedback.
        """
        result: list[MusicParams] = []
        for i in range(n):
            sampled = {}
            for name, (lo, hi, _, _) in PARAM_SPACE.items():
                lo_slice = lo + (hi - lo) * i / n
                hi_slice = lo + (hi - lo) * (i + 1) / n
                sampled[name] = float(self.rng.uniform(lo_slice, hi_slice))
            scale = SCALES[i % len(SCALES)]
            result.append(MusicParams(
                scale         = scale,
                bpm           = sampled["bpm"],
                density       = sampled["density"],
                complexity    = sampled["complexity"],
                octave        = sampled["octave"],
                note_duration = sampled["note_duration"],
                velocity      = sampled["velocity"],
            ))
        return result


# ---------------------------------------------------------------------------
# Baseline aleatório (para ablation study)
# ---------------------------------------------------------------------------

class RandomContinuousBandit:
    """
    Baseline — amostragem uniforme aleatória dentro dos ranges.
    Não aprende. Comparado com ContinuousGaussianBandit no ablation study.
    """

    name = "RandomContinuous"

    def __init__(self, seed: Optional[int] = None):
        self.rng               = np.random.default_rng(seed)
        self.step              = 0
        self.cumulative_reward = 0.0

    def select_params(self) -> tuple[MusicParams, dict]:
        sampled = {
            name: float(self.rng.uniform(lo, hi))
            for name, (lo, hi, _, _) in PARAM_SPACE.items()
        }
        scale = str(self.rng.choice(SCALES))
        params = MusicParams(scale=scale, **sampled)
        return params, {"strategy": "uniform_random"}

    def update(self, params: MusicParams, reward: float):
        self.cumulative_reward += reward
        self.step              += 1

    def arm_stats(self) -> dict:
        return {}

    def reset(self):
        self.__init__(seed=None)


# ---------------------------------------------------------------------------
# Logger de decisões (compatível com ambos os bandits)
# ---------------------------------------------------------------------------

class BanditLogger:
    """
    Regista todas as decisões do bandit para avaliação quantitativa.

    Uso:
        logger = BanditLogger("GaussianTS")
        params, extra = bandit.select_params()
        # ... reward ...
        logger.log(bandit, params, reward, extra)
        logger.to_json("session_log.json")
    """

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.decisions: list[Decision] = []

    def log(
        self,
        bandit,
        params: MusicParams,
        reward: float,
        extra: dict,
    ) -> Decision:
        d = Decision(
            step=bandit.step,
            timestamp=time.time(),
            algorithm=self.algorithm_name,
            params=params.to_dict(),
            reward=reward,
            cumulative_reward=bandit.cumulative_reward,
            arm_stats=bandit.arm_stats(),
            extra=extra,
        )
        self.decisions.append(d)
        return d

    # ------------------------------------------------------------------
    # Análise
    # ------------------------------------------------------------------

    def cumulative_rewards(self) -> list[float]:
        return [d.cumulative_reward for d in self.decisions]

    def rewards_per_step(self) -> list[float]:
        return [d.reward for d in self.decisions]

    def mean_params_over_time(self) -> dict[str, list[float]]:
        """Evolução de cada parâmetro ao longo dos steps — útil para visualização."""
        result: dict[str, list[float]] = {}
        for d in self.decisions:
            for k, v in d.params.items():
                if isinstance(v, (int, float)):
                    result.setdefault(k, []).append(v)
        return result

    def summary(self) -> dict:
        if not self.decisions:
            return {"algorithm": self.algorithm_name, "total_steps": 0}
        total    = len(self.decisions)
        last     = self.decisions[-1]
        tail     = self.decisions[-10:]
        avg_tail: dict[str, float] = {}
        for k in last.params:
            vals = [d.params[k] for d in tail if isinstance(d.params.get(k), (int, float))]
            if vals:
                avg_tail[k] = round(sum(vals) / len(vals), 3)

        return {
            "algorithm":             self.algorithm_name,
            "total_steps":           total,
            "total_reward":          round(last.cumulative_reward, 4),
            "mean_reward_per_step":  round(last.cumulative_reward / total, 4),
            "converged_params":      avg_tail,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  [{s['algorithm']}]  steps={s['total_steps']}"
              f"  reward={s.get('total_reward', 0):+.3f}"
              f"  média/step={s.get('mean_reward_per_step', 0):+.4f}")
        print("  Parâmetros convergidos (últimas 10 decisões):")
        for k, v in s.get("converged_params", {}).items():
            print(f"    {k:<18} {v}")

    # ------------------------------------------------------------------
    # Persistência
    # ------------------------------------------------------------------

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self.decisions], f, indent=2)
        print(f"  [✓] Log guardado → {path}")

    @classmethod
    def from_json(cls, path: str) -> "BanditLogger":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            raise ValueError("Ficheiro de log vazio.")
        logger = cls(data[0]["algorithm"])
        for d in data:
            logger.decisions.append(Decision(**d))
        return logger


# ---------------------------------------------------------------------------
# Avaliador — Ablation Study
# ---------------------------------------------------------------------------

class BanditEvaluator:
    """
    Compara ContinuousGaussianBandit vs RandomContinuousBandit.
    Interface idêntica à versão anterior para não quebrar o código do L.
    """

    def __init__(
        self,
        adaptive_logger: BanditLogger,
        baseline_logger: BanditLogger,
    ):
        self.adaptive = adaptive_logger
        self.baseline = baseline_logger

    def compare(self) -> dict:
        a = self.adaptive.summary()
        b = self.baseline.summary()

        a_reward = a.get("total_reward", 0.0)
        b_reward = b.get("total_reward", 0.0)
        improvement_pct = (a_reward - b_reward) / (abs(b_reward) + 1e-9) * 100

        return {
            "adaptive":        a,
            "baseline":        b,
            "improvement_pct": round(improvement_pct, 2),
            "winner":          a["algorithm"] if a_reward > b_reward else b["algorithm"],
        }

    def print_report(self):
        report = self.compare()
        print("\n" + "=" * 60)
        print("  📊  ABLATION STUDY — Gaussian TS vs Random")
        print("=" * 60)

        for label, key in [("Adaptativo (GaussianTS)", "adaptive"), ("Baseline (Random)", "baseline")]:
            s = report[key]
            print(f"\n  {label}:")
            print(f"    Total reward:  {s.get('total_reward', 0):+.4f}")
            print(f"    Média/step:    {s.get('mean_reward_per_step', 0):+.4f}")
            print(f"    Convergência:")
            for k, v in s.get("converged_params", {}).items():
                print(f"      {k:<18} {v}")

        print(f"\n  Melhoria: {report['improvement_pct']:+.1f}%")
        print(f"  Vencedor: {report['winner']}")
        print("=" * 60)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.compare(), f, indent=2)
        print(f"  [✓] Relatório guardado → {path}")