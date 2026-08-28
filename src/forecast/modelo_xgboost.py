"""XGBoost — candidato independiente en `comparar_modelos.CANDIDATOS`.

Ver `_modelo_arboles.py` por la construcción de features y la estrategia
de pronóstico directo (un estimador por paso de horizonte). Hiperparámetros
fijos, sin autotuning; `random_state` fijo para que el backtest sea
reproducible.
"""

from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from ._modelo_arboles import pronosticar_directo
from .benchmark import pronosticar_seasonal_naive

SEMILLA_ALEATORIA = 42
N_ESTIMATORS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.1


def _crear_estimador() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=SEMILLA_ALEATORIA,
    )


def _ajustar_xgboost(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, Optional[str]]:
    """Ajusta XGBoost y devuelve (forecast, fallback, motivo_fallback)."""
    try:
        return pronosticar_directo(serie, horizonte, _crear_estimador), False, None
    except ValueError as error:
        motivo = f"{type(error).__name__}: {error}"
        return pronosticar_seasonal_naive(serie, horizonte), True, motivo


def pronosticar_xgboost(serie: pd.Series, horizonte: int) -> np.ndarray:
    forecast, _fallback, _motivo = _ajustar_xgboost(serie, horizonte)
    return forecast
