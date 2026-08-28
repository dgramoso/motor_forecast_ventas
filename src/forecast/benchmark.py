"""Benchmark Seasonal Naive — piso de comparación del MVP (spec.md:19).

Con drift si la serie tiene tendencia, sin drift si no (decisión tomada
solo con los datos disponibles hasta el origen del pronóstico, para no
filtrar información futura en el backtest).
"""

import numpy as np
import pandas as pd
from scipy.stats import linregress

PERIODO_ESTACIONAL = 12
UMBRAL_P_VALOR_TENDENCIA = 0.05


def tiene_tendencia(serie: pd.Series) -> bool:
    t = np.arange(len(serie))
    resultado = linregress(t, serie.to_numpy())
    return resultado.pvalue < UMBRAL_P_VALOR_TENDENCIA


def pronosticar_seasonal_naive(
    serie: pd.Series, horizonte: int, periodo: int = PERIODO_ESTACIONAL
) -> np.ndarray:
    """Seasonal naive, con drift automático si la serie tiene tendencia."""
    valores = serie.to_numpy(dtype=float)
    if len(valores) < periodo:
        raise ValueError(f"Se necesitan al menos {periodo} períodos de histórico")

    base = np.array([valores[-periodo + (h % periodo)] for h in range(horizonte)])

    if tiene_tendencia(serie):
        drift = (valores[-1] - valores[0]) / (len(valores) - 1)
        base = base + drift * np.arange(1, horizonte + 1)

    return base
