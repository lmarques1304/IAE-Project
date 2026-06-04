"""
bandit.py
---------
Multi-Armed Bandit
Projeto: Adaptive Music Experience (C2)

Implementa:
  - UCB1   — exploração baseada em incerteza (Upper Confidence Bound)
  - Thompson Sampling — exploração Bayesiana via distribuição Beta
  - RandomBandit      — baseline uniforme (sem aprendizagem)
  - BanditLogger      — logging quantitativo de todas as decisões
  - BanditEvaluator   — ablation study: adaptativo vs baseline

Integração com adaptive_player.py:
    from bandit import ThompsonSamplingBandit, BanditLogger, MOODS

    bandit = ThompsonSamplingBandit(arms=MOODS)
    logger = BanditLogger("thompson")

    mood, extra = bandit.select_arm()           # escolhe mood
    reward = compute_reward(signals)
    bandit.update(mood, reward)
    logger.log(bandit, mood, reward, extra)

Dependências:
    pip install numpy
"""

import math
import json
import time
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Constantes partilhadas
# ---------------------------------------------------------------------------

MOODS = ["happy", "sad", "calm", "energetic"]


# ---------------------------------------------------------------------------
# Estrutura de dados
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    """Registo imutável de uma decisão do bandit (para logging e avaliação)."""
    step:               int
    timestamp:          float
    algorithm:          str
    chosen_arm:         str    # mood escolhido
    reward:             float  # recompensa recebida
    cumulative_reward:  float
    arm_stats:          dict   # snapshot das estatísticas de cada arm
    extra:              dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Braço individual (mood)
# ---------------------------------------------------------------------------

class BanditArm:
    """
    Estatísticas de um braço (= um mood).

    Mantém:
      - plays / total_reward   → para UCB1 e médias
      - alpha / beta           → para Thompson Sampling (distribuição Beta)

    Os priors alpha=1, beta=1 equivalem a uma distribuição uniforme no arranque,
    garantindo que todos os braços são explorados antes de qualquer feedback.
    """

    def __init__(self, name: str):
        self.name         = name
        self.plays        = 0
        self.total_reward = 0.0
        self.alpha        = 1.0   # prior Beta — "1 sucesso imaginário"
        self.beta         = 1.0   # prior Beta — "1 falha imaginária"

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.plays if self.plays > 0 else 0.0

    @property
    def uncertainty(self) -> float:
        """Variância da Beta — mede o quanto ainda não sabemos sobre este braço."""
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def update(self, reward: float):
        """
        Atualiza com recompensa em [-1, 1].
        Normaliza para [0, 1] antes de actualizar a Beta.
        """
        self.plays        += 1
        self.total_reward += reward
        r_norm   = (reward + 1.0) / 2.0          # [-1,1] → [0,1]
        self.alpha += r_norm
        self.beta  += (1.0 - r_norm)

    def to_dict(self) -> dict:
        return {
            "name":        self.name,
            "plays":       self.plays,
            "mean_reward": round(self.mean_reward, 4),
            "alpha":       round(self.alpha, 3),
            "beta":        round(self.beta,  3),
            "uncertainty": round(self.uncertainty, 5),
        }


# ---------------------------------------------------------------------------
# Interface base
# ---------------------------------------------------------------------------

class BaseBandit(ABC):
    """Interface comum — todos os bandits partilham select_arm / update / reset."""

    name: str = "base"

    def __init__(self, arms: list[str], seed: Optional[int] = None):
        self.arms: dict[str, BanditArm] = {a: BanditArm(a) for a in arms}
        self.rng               = np.random.default_rng(seed)
        self.step              = 0
        self.cumulative_reward = 0.0

    @abstractmethod
    def select_arm(self) -> tuple[str, dict]:
        """
        Escolhe um braço segundo a estratégia do algoritmo.
        Devolve (arm_name, info_extra) — o extra é logado para análise.
        """
        ...

    def update(self, arm_name: str, reward: float):
        """Regista a recompensa e avança o contador de passos."""
        self.arms[arm_name].update(reward)
        self.cumulative_reward += reward
        self.step              += 1

    def arm_stats(self) -> dict:
        return {n: arm.to_dict() for n, arm in self.arms.items()}

    def reset(self):
        for arm in self.arms.values():
            arm.plays        = 0
            arm.total_reward = 0.0
            arm.alpha        = 1.0
            arm.beta         = 1.0
        self.step              = 0
        self.cumulative_reward = 0.0


# ---------------------------------------------------------------------------
# UCB1
# ---------------------------------------------------------------------------

class UCB1Bandit(BaseBandit):
    """
    UCB1 — Upper Confidence Bound.

    Fórmula por braço i:
        score(i) = mean_reward(i) + c * sqrt( ln(t) / plays(i) )

    onde t = total de plays e c controla o trade-off exploração/exploitação.

    c > 1  → mais aventureiro (explora braços menos visitados)
    c < 1  → mais conservador (explora mais o que já conhece)
    """

    name = "UCB1"

    def __init__(self, arms: list[str], c: float = 1.5, seed: Optional[int] = None):
        super().__init__(arms, seed)
        self.c = c

    def select_arm(self) -> tuple[str, dict]:
        total = self.step + 1   # +1 evita log(0) no primeiro passo
        scores: dict[str, float] = {}

        for name, arm in self.arms.items():
            if arm.plays == 0:
                scores[name] = float("inf")  # garante que todos são visitados primeiro
            else:
                exploit = arm.mean_reward
                explore = self.c * math.sqrt(math.log(total) / arm.plays)
                scores[name] = exploit + explore

        chosen = max(scores, key=scores.get)
        extra  = {
            "ucb_scores": {
                k: round(v, 4) if v != float("inf") else "∞"
                for k, v in scores.items()
            }
        }
        return chosen, extra


# ---------------------------------------------------------------------------
# Thompson Sampling
# ---------------------------------------------------------------------------

class ThompsonSamplingBandit(BaseBandit):
    """
    Thompson Sampling com distribuição Beta.

    Para cada braço i, amostra θ_i ~ Beta(alpha_i, beta_i).
    Escolhe o braço com maior θ_i.

    Vantagem sobre UCB1: a exploração é proporcional à incerteza real
    estimada, não a uma fórmula fixa. Funciona muito bem com poucas amostras
    — ideal para o cold start deste projeto.

    A distribuição Beta começa em Beta(1,1) = uniforme → nenhum mood é
    favorecido antes de haver dados.
    """

    name = "ThompsonSampling"

    def select_arm(self) -> tuple[str, dict]:
        samples: dict[str, float] = {}
        for name, arm in self.arms.items():
            samples[name] = float(self.rng.beta(arm.alpha, arm.beta))

        chosen = max(samples, key=samples.get)
        extra  = {"ts_samples": {k: round(v, 4) for k, v in samples.items()}}
        return chosen, extra


# ---------------------------------------------------------------------------
# Baseline aleatório
# ---------------------------------------------------------------------------

class RandomBandit(BaseBandit):
    """
    Baseline — seleção uniforme aleatória.
    Não aprende. Serve para o ablation study:
    compara-se a recompensa acumulada do adaptativo vs este baseline.
    """

    name = "Random"

    def select_arm(self) -> tuple[str, dict]:
        chosen = str(self.rng.choice(list(self.arms.keys())))
        return chosen, {"strategy": "uniform_random"}


# ---------------------------------------------------------------------------
# Logger de decisões
# ---------------------------------------------------------------------------

class BanditLogger:
    """
    Regista todas as decisões do bandit para avaliação quantitativa (Etapa 2).

    Uso:
        logger = BanditLogger("ThompsonSampling")
        mood, extra = bandit.select_arm()
        # ... reproduz música, obtém reward ...
        logger.log(bandit, mood, reward, extra)
        logger.to_json("session_log.json")
    """

    def __init__(self, algorithm_name: str):
        self.algorithm_name = algorithm_name
        self.decisions: list[Decision] = []

    def log(
        self,
        bandit: BaseBandit,
        chosen_arm: str,
        reward: float,
        extra: dict,
    ) -> Decision:
        d = Decision(
            step=bandit.step,
            timestamp=time.time(),
            algorithm=self.algorithm_name,
            chosen_arm=chosen_arm,
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

    def arm_selection_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.decisions:
            counts[d.chosen_arm] = counts.get(d.chosen_arm, 0) + 1
        return counts

    def arm_selection_pct(self) -> dict[str, float]:
        counts = self.arm_selection_counts()
        total  = sum(counts.values()) or 1
        return {k: round(v / total * 100, 1) for k, v in counts.items()}

    def summary(self) -> dict:
        if not self.decisions:
            return {"algorithm": self.algorithm_name, "total_steps": 0}
        total = len(self.decisions)
        return {
            "algorithm":             self.algorithm_name,
            "total_steps":           total,
            "total_reward":          round(self.decisions[-1].cumulative_reward, 4),
            "mean_reward_per_step":  round(
                self.decisions[-1].cumulative_reward / total, 4
            ),
            "arm_selections_pct":    self.arm_selection_pct(),
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  [{s['algorithm']}]  steps={s['total_steps']}"
              f"  reward={s.get('total_reward', 0):+.3f}"
              f"  média/step={s.get('mean_reward_per_step', 0):+.4f}")
        for mood, pct in s.get("arm_selections_pct", {}).items():
            bar = "█" * int(pct / 5)
            print(f"    {mood:<12} {bar:<20} {pct:.1f}%")

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
    Compara dois algoritmos lado a lado.
    Serve para o ablation study da Etapa 2 (toggle Bandit on/off).

    Uso:
        evaluator = BanditEvaluator(ts_logger, random_logger)
        evaluator.print_report()
        evaluator.to_json("ablation_report.json")
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
        improvement_pct = (
            (a_reward - b_reward) / (abs(b_reward) + 1e-9) * 100
        )

        return {
            "adaptive":        a,
            "baseline":        b,
            "improvement_pct": round(improvement_pct, 2),
            "winner":          (
                a["algorithm"] if a_reward > b_reward else b["algorithm"]
            ),
        }

    def print_report(self):
        report = self.compare()
        print("\n" + "=" * 60)
        print("  📊  ABLATION STUDY — Adaptive vs Baseline")
        print("=" * 60)

        for label, key in [("Adaptativo", "adaptive"), ("Baseline (Random)", "baseline")]:
            s = report[key]
            print(f"\n  {label}  [{s['algorithm']}]:")
            print(f"    Total reward:   {s.get('total_reward', 0):+.4f}")
            print(f"    Média/step:     {s.get('mean_reward_per_step', 0):+.4f}")
            print(f"    Seleções:")
            for mood, pct in s.get("arm_selections_pct", {}).items():
                bar = "█" * int(pct / 5)
                print(f"      {mood:<12} {bar:<20} {pct:.1f}%")

        print(f"\n  Melhoria adaptativo vs baseline: "
              f"{report['improvement_pct']:+.1f}%")
        print(f"  Vencedor: {report['winner']}")
        print("=" * 60)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.compare(), f, indent=2)
        print(f"  [✓] Relatório guardado → {path}")


# ---------------------------------------------------------------------------
# Demo rápida (teste sem áudio)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    print("=== Teste do Bandit (simulação sem áudio) ===\n")

    # Simula preferências do utilizador: happy e energetic têm reward maior
    TRUE_REWARDS = {
        "happy":     0.6,
        "sad":    -0.2,
        "calm":      0.1,
        "energetic": 0.8,
    }

    def fake_reward(mood: str) -> float:
        base  = TRUE_REWARDS[mood]
        noise = random.gauss(0, 0.2)
        return max(-1.0, min(1.0, base + noise))

    N_STEPS = 40

    # Corre UCB1, Thompson Sampling e Random em paralelo
    bandits  = [UCB1Bandit(MOODS, c=1.5, seed=0),
                ThompsonSamplingBandit(MOODS, seed=1),
                RandomBandit(MOODS, seed=2)]
    loggers  = [BanditLogger(b.name) for b in bandits]

    for step in range(N_STEPS):
        for bandit, logger in zip(bandits, loggers):
            mood, extra = bandit.select_arm()
            reward      = fake_reward(mood)
            bandit.update(mood, reward)
            logger.log(bandit, mood, reward, extra)

    print("  Resultados após", N_STEPS, "passos:\n")
    for logger in loggers:
        logger.print_summary()

    # Ablation study: Thompson vs Random
    print()
    evaluator = BanditEvaluator(loggers[1], loggers[2])
    evaluator.print_report()

    # Guarda logs
    for logger in loggers:
        logger.to_json(f"log_{logger.algorithm_name.lower()}.json")