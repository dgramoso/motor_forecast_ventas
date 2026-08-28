"""Prophet — candidato independiente en `comparar_modelos.CANDIDATOS`.

Config para datos mensuales (Prophet por default asume datos diarios):
solo estacionalidad anual, sin semanal/diaria; `growth="linear"` sin cap
(no hay techo de demanda definido en la spec); `seasonality_mode="additive"`,
coherente con que `modelo_ets.py` tampoco usa variantes multiplicativas en
el MVP.

Igual que `modelo_ets.py` / `modelo_sarima.py`, si el ajuste falla cae al
benchmark Seasonal Naive.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from .benchmark import pronosticar_seasonal_naive

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

# Prophet no lanza una excepción homogénea documentada ante datos
# degenerados (muy poca historia, serie constante); en la práctica falla
# con RuntimeError (optimización de Stan) o ValueError (validación de
# entrada). Cualquier otra excepción se deja propagar.
_EXCEPCIONES_AJUSTE_PROPHET = (ValueError, RuntimeError)


def _ajustar_prophet(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, Optional[str]]:
    """Ajusta Prophet y devuelve (forecast, fallback, motivo_fallback)."""
    datos = pd.DataFrame({"ds": serie.index, "y": serie.to_numpy(dtype=float)})

    try:
        modelo = Prophet(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode="additive",
        )
        modelo.fit(datos)

        fechas_futuras = pd.date_range(
            start=serie.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq="MS"
        )
        prediccion = modelo.predict(pd.DataFrame({"ds": fechas_futuras}))
        return prediccion["yhat"].to_numpy(), False, None
    except _EXCEPCIONES_AJUSTE_PROPHET as error:
        motivo = f"{type(error).__name__}: {error}"
        return pronosticar_seasonal_naive(serie, horizonte), True, motivo


def pronosticar_prophet(serie: pd.Series, horizonte: int) -> np.ndarray:
    forecast, _fallback, _motivo = _ajustar_prophet(serie, horizonte)
    return forecast
