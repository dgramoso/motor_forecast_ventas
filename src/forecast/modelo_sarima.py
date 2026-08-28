"""SARIMA (statsmodels SARIMAX) — candidato independiente en
`comparar_modelos.CANDIDATOS`, no forma parte del router de `modelo.py`.

El orden (p,d,q)(P,D,Q) se elige por grilla de AIC, recalculada en cada
ventana de entrenamiento recibida — nunca con información posterior al
origen del pronóstico, mismo criterio que `benchmark.estimar_tendencia`.
No se usa `pmdarima` (evita una dependencia con historial de
incompatibilidad con numpy/statsmodels recientes): la grilla es chica a
propósito y alcanza con lo que ya trae `statsmodels`.

Igual que `modelo_ets.py`, si el ajuste falla por datos degenerados cae
al benchmark Seasonal Naive.
"""

import itertools
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from ._ajuste_con_fallback import ajustar_con_fallback
from .benchmark import PERIODO_ESTACIONAL

_ORDENES_P_Q = (0, 1, 2)
_ORDENES_D = (0, 1)
_ORDENES_ESTACIONALES = (0, 1)

# Mismas excepciones que `modelo_ets._EXCEPCIONES_AJUSTE_ETS`: condición
# de datos degenerada (poca historia, ventana casi constante, etc.), no
# bug de programación.
_EXCEPCIONES_AJUSTE_SARIMA = (ValueError, np.linalg.LinAlgError)


def _mejor_orden_por_aic(serie: pd.Series) -> tuple[tuple[int, int, int], tuple[int, int, int, int]]:
    mejor_aic = np.inf
    mejor_orden = (0, 1, 0)
    mejor_orden_estacional = (0, 0, 0, PERIODO_ESTACIONAL)

    combinaciones = itertools.product(
        _ORDENES_P_Q, _ORDENES_D, _ORDENES_P_Q,
        _ORDENES_ESTACIONALES, _ORDENES_ESTACIONALES, _ORDENES_ESTACIONALES,
    )
    for p, d, q, P, D, Q in combinaciones:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                resultado = SARIMAX(
                    serie,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, PERIODO_ESTACIONAL),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
        except _EXCEPCIONES_AJUSTE_SARIMA:
            continue

        if resultado.aic < mejor_aic:
            mejor_aic = resultado.aic
            mejor_orden = (p, d, q)
            mejor_orden_estacional = (P, D, Q, PERIODO_ESTACIONAL)

    return mejor_orden, mejor_orden_estacional


def _ajustar_sarimax(serie: pd.Series, horizonte: int) -> np.ndarray:
    orden, orden_estacional = _mejor_orden_por_aic(serie)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = SARIMAX(
            serie,
            order=orden,
            seasonal_order=orden_estacional,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)
    return modelo.forecast(horizonte).to_numpy()


def _ajustar_sarima(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, Optional[str]]:
    """Ajusta SARIMA y devuelve (forecast, fallback, motivo_fallback)."""
    return ajustar_con_fallback(serie, horizonte, _ajustar_sarimax, _EXCEPCIONES_AJUSTE_SARIMA)


def pronosticar_sarima(serie: pd.Series, horizonte: int) -> np.ndarray:
    forecast, _fallback, _motivo = _ajustar_sarima(serie, horizonte)
    return forecast
