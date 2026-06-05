import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from typing import Optional
from storage import get_all_feedback

MOOD_BOUNDS = {
    "happy": {"bpm": (120, 160), "density": (0.1, 1.0)},
    "calm": {"bpm": (60,  90),  "density": (0.1, 1.0)},
    "sad": {"bpm": (50,  80),  "density": (0.1, 1.0)},
    "energetic": {"bpm": (140, 180), "density": (0.1, 1.0)},
}

MOOD_DEFAULTS = {
    "happy": {"bpm": 140, "density": 0.7},
    "calm": {"bpm": 75,  "density": 0.4},
    "sad": {"bpm": 65,  "density": 0.3},
    "energetic": {"bpm": 160, "density": 0.8},
}

MIN_POINTS = 3   # Mínimo de pontos de feedback antes de ligar o GP

REWARD_MAP = {
    "like": 1.0,
    "dislike": -1.0,
    "skip": -0.5,
}

def _feedback_to_reward(feedback: str) -> float:
    return REWARD_MAP.get(feedback, 0.0)

def _normalise(bpm: float, density: float, mood: str):
    bounds = MOOD_BOUNDS[mood]
    bpm_lo, bpm_hi = bounds["bpm"]
    bpm_norm = (bpm - bpm_lo) / (bpm_hi - bpm_lo)
    density_norm = (density - 0.1) / 0.9
    return np.array([bpm_norm, density_norm])

def _denormalise(x_norm: np.ndarray, mood: str) -> dict:
    bounds = MOOD_BOUNDS[mood]
    bpm_lo, bpm_hi = bounds["bpm"]
    bpm = float(np.clip(x_norm[0] * (bpm_hi - bpm_lo) + bpm_lo, bpm_lo, bpm_hi))
    density = float(np.clip(x_norm[1] * 0.9 + 0.1, 0.1, 1.0))
    return {"bpm": round(bpm), "density": round(density, 3)}

def _expected_improvement(
    X_candidate: np.ndarray,
    gp: GaussianProcessRegressor,
    y_best: float,
    xi: float = 0.01,
) -> np.ndarray:

    mu, sigma = gp.predict(X_candidate, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * norm.cdf(z) + sigma * norm.pdf(z)
    return -ei  # negativo para minimizar

class BayesianOptimizer:
    def __init__(self):
        # Um GP por mood — cada um aprende de forma independente
        self._gp_cache: dict[str, GaussianProcessRegressor] = {}

    def _build_gp(self) -> GaussianProcessRegressor:
        """ Kernel Matern 5/2 + ruído branco. Matern é mais robusto que RBF para funções não suaves (preferências humanas). """
        kernel = Matern(length_scale=0.5, nu=2.5) + WhiteKernel(noise_level=0.1)
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
        )

    def _get_training_data(self, mood: str):
        """
        Lê o histórico de feedback e devolve (X, y) para o mood pedido.
        X: array (n, 2) com [bpm_norm, density_norm]
        y: array (n,)   com recompensas em [-1, 1]
        """
        all_feedback = get_all_feedback()
        mood_feedback = [f for f in all_feedback if f["mood"] == mood]

        if len(mood_feedback) < MIN_POINTS:
            return None, None

        X, y = [], []
        for f in mood_feedback:
            bpm = f.get("bpm", MOOD_DEFAULTS[mood]["bpm"])
            density = f.get("density", MOOD_DEFAULTS[mood]["density"])
            reward = _feedback_to_reward(f["feedback"])
            X.append(_normalise(bpm, density, mood))
            y.append(reward)

        return np.array(X), np.array(y)

    def suggest(self, mood: str) -> dict:
        """
        Sugere os melhores parâmetros (BPM, density) para o mood dado.
        Devolve sempre um dict com: { "bpm": int, "density": float, "source": "gp" | "default" }
        source="default" significa que há poucos dados e usámos os valores pré-definidos do mood.
        """
        if mood not in MOOD_BOUNDS:
            mood = "calm"  # fallback seguro

        X, y = self._get_training_data(mood)

        if X is None:
            defaults = MOOD_DEFAULTS[mood].copy()
            defaults["source"] = "default"
            defaults["reason"] = f"Menos de {MIN_POINTS} pontos de feedback para este mood"
            return defaults

        gp = self._build_gp()
        gp.fit(X, y)
        self._gp_cache[mood] = gp

        y_best = float(np.max(y))
        rng = np.random.default_rng(seed=42)
        X_candidates = rng.uniform(0, 1, size=(200, 2))
        ei_values = _expected_improvement(X_candidates, gp, y_best)
        best_idx = int(np.argmin(ei_values))
        x0 = X_candidates[best_idx]
        result = minimize(
            fun=lambda x: float(_expected_improvement(x.reshape(1, -1), gp, y_best)),
            x0=x0,
            bounds=[(0, 1), (0, 1)],
            method="L-BFGS-B",
        )

        best_x = result.x if result.success else x0
        params = _denormalise(best_x, mood)
        params["source"] = "gp"
        params["n_observations"] = len(y)
        params["y_best"] = round(y_best, 3)
        return params

    def status(self) -> dict:
        """ Resumo do estado atual do otimizador (para debug/monitoring). Devolve quantos feedbacks temos por mood e se o GP está ativo."""
        all_feedback = get_all_feedback()
        summary = {}
        for mood in MOOD_BOUNDS:
            mood_data = [f for f in all_feedback if f["mood"] == mood]
            summary[mood] = {
                "n_feedback": len(mood_data),
                "gp_active": mood in self._gp_cache,
                "ready": len(mood_data) >= MIN_POINTS,
            }
        return summary

optimizer = BayesianOptimizer()