"""Backtest walk-forward (rolling), sin split único ni k-fold —
ver spec.md:19 sobre por qué evitar leakage temporal.

En cada origen se entrena solo con el histórico disponible hasta ese
punto y se pronostica el horizonte siguiente. Sirve tanto para el
benchmark como para cualquier modelo que respete la firma
`función_pronostico(serie_entrenamiento, horizonte) -> array`.
"""

from typing import Callable

import numpy as np
import pandas as pd

from .benchmark import PERIODO_ESTACIONAL
from .metricas import bias, mae, mase, wape

FuncionPronostico = Callable[[pd.Series, int], np.ndarray]


def backtest_walk_forward(
    serie: pd.Series,
    funcion_pronostico: FuncionPronostico,
    horizonte: int,
    ventana_minima: int,
) -> pd.DataFrame:
    filas = []
    ultimo_origen = len(serie) - horizonte
    for origen in range(ventana_minima, ultimo_origen + 1):
        entrenamiento = serie.iloc[:origen]
        real = serie.iloc[origen : origen + horizonte].to_numpy()
        pronostico = funcion_pronostico(entrenamiento, horizonte)

        filas.append(
            {
                "fecha_origen": serie.index[origen - 1],
                "wape": wape(real, pronostico),
                "bias": bias(real, pronostico),
                "mae": mae(real, pronostico),
                "mase": mase(real, pronostico, entrenamiento.to_numpy(), PERIODO_ESTACIONAL),
            }
        )

    return pd.DataFrame(filas)
