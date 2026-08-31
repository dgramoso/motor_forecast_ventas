"""TSB (Teunter-Syntetos-Babai) para demanda intermitente — SKU-003 mostró
que Holt-Winters no sirve cuando la mayoría de los períodos tienen
demanda cero (ver comparación benchmark vs. ETS).

A diferencia de Croston clásico, TSB actualiza la probabilidad de
ocurrencia en cada período (no solo cuando hay demanda), lo que evita
que el pronóstico quede "viejo" tras una racha larga de ceros.

TSB es un candidato independiente en `comparar_modelos.CANDIDATOS`, igual
que ETS — cuál conviene para cada SKU lo decide el backtest walk-forward,
no una regla fija de intermitencia (ver `diagnostico_demanda.py`, que
clasifica el patrón de demanda como diagnóstico, sin forzar el modelo
ganador).
"""

from typing import Optional

import numpy as np
import pandas as pd

ALPHA = 0.1  # suavizado del tamaño de la demanda
BETA = 0.1  # suavizado de la probabilidad de ocurrencia


def pronosticar_tsb(
    serie: pd.Series, horizonte: int, alpha: float = ALPHA, beta: float = BETA
) -> np.ndarray:
    valores = serie.to_numpy(dtype=float)
    demandas_no_cero = valores[valores > 0]

    z_hat = demandas_no_cero[0] if len(demandas_no_cero) else 0.0
    p_hat = (valores > 0).mean()

    for d in valores:
        ocurre = d > 0
        if ocurre:
            z_hat = alpha * d + (1 - alpha) * z_hat
        p_hat = beta * ocurre + (1 - beta) * p_hat

    nivel = z_hat * p_hat
    return np.full(horizonte, nivel)


def _ajustar_tsb(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, Optional[str]]:
    """Mismo contrato que `_ajustar_ets` (forecast, fallback, motivo) para
    que `comparar_modelos.CANDIDATOS_CON_METADATA` trate a TSB igual que a
    los demás candidatos. TSB nunca cae en fallback: no ajusta ninguna
    librería externa que pueda fallar por datos degenerados, solo
    suavizado exponencial simple sobre la serie recibida."""
    return pronosticar_tsb(serie, horizonte), False, None
