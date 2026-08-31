"""Diagnóstico del patrón de demanda — ADI, CV² y clasificación SBC
(Syntetos, Boylan y Croston 2005), calculados exclusivamente sobre la
serie recibida (nunca con información futura: quien llama pasa la
ventana de entrenamiento o el histórico completo, según corresponda).

Es diagnóstico y priorización de candidatos, no determina el modelo
ganador — eso lo sigue decidiendo el backtest walk-forward en
comparar_modelos.py (ver docs/adr/, decisión de no usar reglas rígidas
tipo "si intermitente, TSB").
"""

import numpy as np
import pandas as pd

UMBRAL_ADI = 1.32
UMBRAL_CV2 = 0.49


def tasa_de_ceros(serie: pd.Series) -> float:
    """% de períodos con demanda cero. `nan` si la serie está vacía."""
    if len(serie) == 0:
        return np.nan
    return float((serie == 0).mean())


def adi(serie: pd.Series) -> float:
    """Average demand interval: cantidad de períodos por cada período con
    demanda positiva. `inf` si la serie no tuvo ningún período con
    demanda positiva — no hay intervalo promedio que calcular."""
    n_positivos = int((serie > 0).sum())
    if n_positivos == 0:
        return np.inf
    return len(serie) / n_positivos


def cv2(serie: pd.Series) -> float:
    """Coeficiente de variación al cuadrado de la demanda, sobre los
    períodos con demanda positiva únicamente. `0.0` (no `nan`) cuando hay
    menos de 2 observaciones positivas — no hay variación que estimar con
    un solo punto — o cuando la media positiva da cero, algo que no
    debería ocurrir por construcción pero se cubre para no dividir por
    cero."""
    positivos = serie[serie > 0].to_numpy(dtype=float)
    if len(positivos) < 2:
        return 0.0
    media = positivos.mean()
    if media == 0:
        return 0.0
    return float((positivos.std() / media) ** 2)


def clasificar_demanda(serie: pd.Series, umbral_adi: float = UMBRAL_ADI, umbral_cv2: float = UMBRAL_CV2) -> str:
    """Clasificación SBC sobre ADI y CV² — diagnóstico y priorización de
    candidatos, no una regla que fuerce el modelo ganador:

        "sin_demanda"  -> serie completamente cero (ADI indefinido)
        "regular"      -> ADI bajo, CV² bajo
        "intermitente" -> ADI alto, CV² bajo
        "erratica"     -> ADI bajo, CV² alto
        "lumpy"        -> ADI alto, CV² alto
    """
    valor_adi = adi(serie)
    if np.isinf(valor_adi):
        return "sin_demanda"

    valor_cv2 = cv2(serie)
    if valor_adi < umbral_adi:
        return "erratica" if valor_cv2 >= umbral_cv2 else "regular"
    return "lumpy" if valor_cv2 >= umbral_cv2 else "intermitente"
