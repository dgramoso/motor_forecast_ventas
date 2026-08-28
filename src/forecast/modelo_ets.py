"""Modelo estadístico clásico por SKU: Holt-Winters (ETS), aditivo.

Trend y seasonal se activan según lo que muestre el propio período de
entrenamiento (misma lógica que el benchmark, sin mirar el futuro). Si
el ajuste falla — típico en ventanas casi constantes o con muchos ceros,
como el SKU intermitente — cae al benchmark Seasonal Naive.
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from .benchmark import PERIODO_ESTACIONAL, pronosticar_seasonal_naive, tiene_tendencia


def pronosticar_ets(serie: pd.Series, horizonte: int) -> np.ndarray:
    usar_estacionalidad = len(serie) >= 2 * PERIODO_ESTACIONAL
    trend = "add" if tiene_tendencia(serie) else None
    seasonal = "add" if usar_estacionalidad else None

    try:
        modelo = ExponentialSmoothing(
            serie,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=PERIODO_ESTACIONAL if usar_estacionalidad else None,
            initialization_method="estimated",
        ).fit()
        return modelo.forecast(horizonte).to_numpy()
    except Exception:
        return pronosticar_seasonal_naive(serie, horizonte)
