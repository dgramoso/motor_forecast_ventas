"""Benchmark Seasonal Naive + drift condicional — piso de comparación del
MVP (spec.md:19).

    seasonal_naive_h = y[t + h - periodo]
    forecast_h = seasonal_naive_h + beta * h   si p_value(beta) < 0.05
    forecast_h = seasonal_naive_h              si no

`beta` es la pendiente temporal de una regresión OLS que controla por
estacionalidad mensual (ver `estimar_tendencia`), no la diferencia simple
entre el primer y el último valor de la serie — eso confundía outliers
puntuales con tendencia real. La decisión de aplicar drift y su magnitud
salen de la misma regresión (una sola vez por llamada), para que sean
estadísticamente coherentes y no se recalculen dos veces.

`estimar_tendencia` también la usa `modelo_ets.py` para decidir si activar
el componente de tendencia de Holt-Winters — la pendiente OLS solo decide
ahí; ETS estima su propia tendencia de forma independiente, nunca toma
`beta` como valor.

Todo se recalcula sobre la serie recibida en cada llamada (típicamente
`entrenamiento = serie.iloc[:origen]` dentro del backtest walk-forward),
así que nunca se usa información posterior al origen del pronóstico.
"""

from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress

PERIODO_ESTACIONAL = 12
UMBRAL_P_VALOR_TENDENCIA = 0.05

# Con dummies mensuales el modelo estima 13 parámetros (intercepto +
# tendencia + 11 dummies). Por debajo de dos ciclos estacionales completos
# no hay grados de libertad suficientes para una estimación confiable, así
# que se cae a una regresión lineal simple (y_t = alpha + beta*t). Es una
# limitación conocida del MVP con ventanas cortas (p.ej. origen=24 del
# backtest walk-forward, límite del `ventana_minima` de comparar_modelos.py):
# no se aplica ninguna corrección automática, solo el fallback a la
# regresión simple.
MIN_OBSERVACIONES_TENDENCIA_ESTACIONAL = 2 * PERIODO_ESTACIONAL


class ResultadoTendencia(NamedTuple):
    """Salida de `estimar_tendencia`. Se calcula una sola vez por serie y
    se reutiliza tanto para la decisión (`tiene_tendencia`) como para la
    magnitud del drift (`pendiente`), en vez de correr la regresión dos
    veces con el mismo dato de entrada."""

    pendiente: float
    p_valor: float
    intercepto: float
    tiene_tendencia: bool


def estimar_tendencia(serie: pd.Series) -> ResultadoTendencia:
    """Pendiente temporal controlando por estacionalidad mensual:

        y_t = alpha + beta*t + gamma_feb*D_feb + ... + gamma_dic*D_dic + eps_t

        H0: beta = 0   vs.   H1: beta != 0

    Enero queda como categoría de referencia (dummies para febrero a
    diciembre). `tiene_tendencia` es True cuando p_value(beta) <
    `UMBRAL_P_VALOR_TENDENCIA`. Ignora observaciones NaN.

    Con menos de `MIN_OBSERVACIONES_TENDENCIA_ESTACIONAL` puntos válidos,
    o si el índice no trae información de mes (no es `DatetimeIndex`), cae
    a una regresión simple y_t = alpha + beta*t (mismo criterio de
    significancia, sin controlar por estacionalidad).
    """
    y_completo = serie.to_numpy(dtype=float)
    validos = ~np.isnan(y_completo)
    t = np.arange(len(y_completo))[validos]
    y = y_completo[validos]

    if len(y) < 2:
        intercepto = float(y[0]) if len(y) else 0.0
        return ResultadoTendencia(pendiente=0.0, p_valor=1.0, intercepto=intercepto, tiene_tendencia=False)

    tiene_meses = isinstance(serie.index, pd.DatetimeIndex)
    if len(y) < MIN_OBSERVACIONES_TENDENCIA_ESTACIONAL or not tiene_meses:
        resultado = linregress(t, y)
        pendiente, p_valor, intercepto = float(resultado.slope), float(resultado.pvalue), float(resultado.intercept)
    else:
        meses = serie.index.month.to_numpy()[validos]
        dummies = pd.get_dummies(meses, prefix="mes", drop_first=True, dtype=float).to_numpy()
        x = np.column_stack([np.ones_like(t, dtype=float), t, dummies])

        modelo = sm.OLS(y, x).fit()
        pendiente, p_valor, intercepto = float(modelo.params[1]), float(modelo.pvalues[1]), float(modelo.params[0])

    return ResultadoTendencia(
        pendiente=pendiente,
        p_valor=p_valor,
        intercepto=intercepto,
        tiene_tendencia=p_valor < UMBRAL_P_VALOR_TENDENCIA,
    )


def tiene_tendencia(serie: pd.Series) -> bool:
    """Wrapper compatible con el uso anterior (solo bool) — ver
    `estimar_tendencia` si además hace falta la pendiente o el p-value."""
    return estimar_tendencia(serie).tiene_tendencia


def pronosticar_seasonal_naive(
    serie: pd.Series, horizonte: int, periodo: int = PERIODO_ESTACIONAL
) -> np.ndarray:
    """Seasonal naive, con drift automático (pendiente OLS) si la serie
    tiene tendencia estadísticamente significativa una vez controlada la
    estacionalidad mensual."""
    valores = serie.to_numpy(dtype=float)
    if len(valores) < periodo:
        raise ValueError(f"Se necesitan al menos {periodo} períodos de histórico")

    base = np.array([valores[-periodo + (h % periodo)] for h in range(horizonte)])

    resultado = estimar_tendencia(serie)
    if resultado.tiene_tendencia:
        base = base + resultado.pendiente * np.arange(1, horizonte + 1)

    return base
