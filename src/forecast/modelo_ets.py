"""Modelo estadístico clásico por SKU: Holt-Winters (ETS), aditivo.

Trend y seasonal se activan según lo que muestre el propio período de
entrenamiento (sin mirar el futuro):

    n < 2*PERIODO_ESTACIONAL (24) -> sin estacionalidad -> ETS(A,N,N) o ETS(A,A,N)
    n >= 2*PERIODO_ESTACIONAL     -> con estacionalidad  -> ETS(A,N,A) o ETS(A,A,A)

En ambos casos, el componente de tendencia ("N" o "A" en la posición del
medio) se activa según `benchmark.tiene_tendencia` — la misma regresión
OLS-con-dummies-mensuales que decide el drift del benchmark, pero acá solo
decide si activar el trend de Holt-Winters. ETS estima su propia pendiente
de tendencia de forma independiente: nunca toma la pendiente OLS como
valor, solo la usa como señal de encendido/apagado.

Todavía no hay variantes multiplicativas ni damped trend — quedan fuera
del alcance del MVP.

Si el ajuste falla por una condición de datos degenerada — típico en
ventanas casi constantes o con muy poca historia, ver `_EXCEPCIONES_AJUSTE_ETS`
más abajo — cae al benchmark Seasonal Naive. Ese fallback queda registrado:
`_ajustar_ets` devuelve si hubo fallback y el motivo, para que quien
necesite auditar la corrida (backtest o pipeline) pueda usarlo sin que
`pronosticar_ets()` deje de devolver solo el array de pronóstico.
"""

from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from ._ajuste_con_fallback import ajustar_con_fallback
from .benchmark import PERIODO_ESTACIONAL, tiene_tendencia

# Todas las excepciones que ExponentialSmoothing puede lanzar por una
# condición de datos degenerada (poca historia, ventana casi constante,
# parámetros de suavizado fuera de rango, etc.) son ValueError en esta
# versión de statsmodels — verificado contra el código fuente de
# statsmodels.tsa.holtwinters. Se agrega LinAlgError de forma defensiva:
# no se lo pudo disparar en la práctica, pero es un fallo numérico conocido
# del optimizador (matriz singular en el ajuste) y cae dentro de la misma
# categoría de "problema de convergencia/datos", no de bug de programación.
# Cualquier otra excepción (TypeError, AttributeError, etc.) se deja
# propagar: sería un error de programación, no algo que el fallback deba
# ocultar.
_EXCEPCIONES_AJUSTE_ETS = (ValueError, np.linalg.LinAlgError)


def _ajustar_holt_winters(serie: pd.Series, horizonte: int) -> np.ndarray:
    usar_estacionalidad = len(serie) >= 2 * PERIODO_ESTACIONAL
    trend = "add" if tiene_tendencia(serie) else None
    seasonal = "add" if usar_estacionalidad else None

    modelo = ExponentialSmoothing(
        serie,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=PERIODO_ESTACIONAL if usar_estacionalidad else None,
        initialization_method="estimated",
    ).fit()
    return modelo.forecast(horizonte).to_numpy()


def _ajustar_ets(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, Optional[str]]:
    """Ajusta ETS y devuelve (forecast, fallback, motivo_fallback).
    `fallback=True` implica que el forecast en realidad vino de
    `pronosticar_seasonal_naive` (fallback), no de Holt-Winters."""
    return ajustar_con_fallback(serie, horizonte, _ajustar_holt_winters, _EXCEPCIONES_AJUSTE_ETS)


def pronosticar_ets(serie: pd.Series, horizonte: int) -> np.ndarray:
    forecast, _fallback, _motivo = _ajustar_ets(serie, horizonte)
    return forecast
