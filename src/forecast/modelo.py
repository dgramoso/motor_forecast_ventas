"""Router del modelo: TSB para SKUs intermitentes, ETS para el resto.

Un solo algoritmo no le sirve a todos los SKUs (ver hallazgo de
SKU-003) — esto es la segmentación mínima necesaria, no una elección
arbitraria.
"""

import numpy as np
import pandas as pd

from .modelo_ets import pronosticar_ets
from .modelo_intermitente import es_intermitente, pronosticar_tsb


def pronosticar_modelo(serie: pd.Series, horizonte: int) -> np.ndarray:
    if es_intermitente(serie):
        return pronosticar_tsb(serie, horizonte)
    return pronosticar_ets(serie, horizonte)
