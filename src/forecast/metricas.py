"""WAPE, Bias y MAE — ver spec.md:31-36 para la justificación de por qué
estas tres y no MAPE/R². MASE se agrega para series intermitentes, donde
WAPE puede quedar indefinido (demanda real total cero en la ventana) y
MASE sigue siendo calculable mientras el histórico de entrenamiento no
sea degenerado.
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


def mase(
    real: np.ndarray,
    pronostico: np.ndarray,
    historico_entrenamiento: np.ndarray,
    periodo_estacional: int = 1,
) -> float:
    """Mean Absolute Scaled Error (Hyndman-Koehler): MAE del pronóstico
    escalado por el MAE de un naive estacional in-sample sobre el
    histórico de entrenamiento (no sobre la ventana de test) — a
    diferencia de WAPE/bias/MAE, necesita ese histórico para calcular el
    denominador, no solo `real`/`pronostico`.

    `nan` si el histórico no alcanza para estimar el denominador (menos
    de `periodo_estacional + 1` observaciones — cae a naive de un paso,
    `periodo_estacional=1`, si ni así alcanza) o si el naive in-sample
    tiene error cero (serie de entrenamiento perfectamente
    estacional/constante, no hay variación que escalar)."""
    real, pronostico = np.asarray(real, dtype=float), np.asarray(pronostico, dtype=float)
    historico = np.asarray(historico_entrenamiento, dtype=float)

    m = periodo_estacional
    if len(historico) <= m:
        m = 1
    if len(historico) <= m:
        return np.nan

    error_naive_in_sample = np.abs(historico[m:] - historico[:-m])
    denominador = error_naive_in_sample.mean()
    if denominador == 0:
        return np.nan

    return float(np.mean(np.abs(real - pronostico)) / denominador)
