from fastapi import APIRouter
from bayesian_optimizer import optimizer

router = APIRouter()

@router.post("/{mood}", summary="Sugerir parâmetros otimizados para um mood")
def suggest_params(mood: str):
    """ Usa a Otimização Bayesiana para sugerir os melhores parâmetros (BPM, density) para o mood dado, com base no histórico de feedback. """
    params = optimizer.suggest(mood)
    return params

@router.get("/status", summary="Estado do otimizador por mood")
def optimizer_status():
    """ Mostra quantos pontos de feedback existem por mood e se o GP está ativo. Útil para debug e para o relatório do ablation study. """
    return optimizer.status()
