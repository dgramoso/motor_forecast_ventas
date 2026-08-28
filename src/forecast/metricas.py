"""WAPE, Bias y MAE — ver spec.md:31-36 para la justificación de por qué
estas tres y no MAPE/R².
"""

import numpy as np


def wape(real: np.ndarray, pronostico: np.ndarray) -> float:
    """np.nan si la ventana tiene demanda real total cero (indefinido, no error del modelo)."""
    real, pronostico = np.asarray(real, dtype=float), np.asarray(pronostico, dtype=float)
    denominador = np.sum(np.abs(real))
    if denominador == 0:
        return np.nan
    return np.sum(np.abs(real - pronostico)) / denominador


def bias(real: np.ndarray, pronostico: np.ndarray) -> float:
    """np.nan si la ventana tiene demanda real total cero (indefinido, no error del modelo)."""
    real, pronostico = np.asarray(real, dtype=float), np.asarray(pronostico, dtype=float)
    denominador = np.sum(np.abs(real))
    if denominador == 0:
        return np.nan
    return np.sum(pronostico - real) / denominador


def mae(real: np.ndarray, pronostico: np.ndarray) -> float:
    real, pronostico = np.asarray(real, dtype=float), np.asarray(pronostico, dtype=float)
    return np.mean(np.abs(real - pronostico))
