"""Wrapper compartido de fallback — usado por modelo_ets.py, modelo_sarima.py
y modelo_prophet.py: los tres repetían exactamente la misma forma
(ajustar, si falla con una excepción conocida caer al benchmark Seasonal
Naive y registrar el motivo). XGBoost y Random Forest no lo usan: ya
tienen su propio seam en `_modelo_arboles.py` (su único modo de fallo
conocido es el `ValueError` de la precondición de filas mínimas, no una
librería externa que ajustar), forzarlos acá sería una capa sin necesidad.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .benchmark import pronosticar_seasonal_naive


def ajustar_con_fallback(
    serie: pd.Series,
    horizonte: int,
    ajustar: Callable[[pd.Series, int], np.ndarray],
    excepciones: tuple[type[Exception], ...],
) -> tuple[np.ndarray, bool, Optional[str]]:
    """Corre `ajustar(serie, horizonte)`. Si lanza una de `excepciones`,
    cae a `pronosticar_seasonal_naive` y devuelve el motivo. Cualquier
    otra excepción se propaga: sería un error de programación, no algo
    que el fallback deba ocultar."""
    try:
        return ajustar(serie, horizonte), False, None
    except excepciones as error:
        motivo = f"{type(error).__name__}: {error}"
        return pronosticar_seasonal_naive(serie, horizonte), True, motivo
