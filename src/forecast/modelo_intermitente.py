"""TSB (Teunter-Syntetos-Babai) para demanda intermitente — SKU-003 mostró
que Holt-Winters no sirve cuando la mayoría de los períodos tienen
demanda cero (ver comparación benchmark vs. ETS).

A diferencia de Croston clásico, TSB actualiza la probabilidad de
ocurrencia en cada período (no solo cuando hay demanda), lo que evita
que el pronóstico quede "viejo" tras una racha larga de ceros.
"""

import numpy as np
import pandas as pd

UMBRAL_PROPORCION_CERO = 0.3
ALPHA = 0.1  # suavizado del tamaño de la demanda
BETA = 0.1  # suavizado de la probabilidad de ocurrencia


def es_intermitente(serie: pd.Series, umbral: float = UMBRAL_PROPORCION_CERO) -> bool:
    return (serie == 0).mean() >= umbral


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
