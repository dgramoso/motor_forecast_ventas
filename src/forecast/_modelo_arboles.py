"""Utilidades compartidas por los modelos basados en árboles (XGBoost,
Random Forest) — construcción de features y pronóstico directo por
horizonte. No es un candidato en sí mismo: lo consumen `modelo_xgboost.py`
y `modelo_random_forest.py`.

Pronóstico directo, no recursivo: un estimador independiente por paso de
horizonte, para que el error de un paso no contamine los siguientes.

El lag=12 (mismo mes del año anterior) y su media móvil consumen 12
observaciones de historia antes de dejar una sola fila utilizable — con
poca historia (`len(serie) < 2*PERIODO_ESTACIONAL`) eso fuerza fallback
casi siempre, aunque el modelo sí podría entrenar con lags más cortos.
Mismo criterio que `modelo_ets.py` para prender/apagar estacionalidad
según el largo de la serie: con suficiente historia se usa el set
completo (lags 1,2,3,12 + medias móviles 3,12); si no, el set corto
(lags 1,2,3 + media móvil 3) — más pobre, pero entrenable. En ambos casos
se agrega el mes calendario del período a pronosticar como dummy (mismo
patrón que `benchmark.estimar_tendencia`).
"""

from typing import Callable

import numpy as np
import pandas as pd

from .benchmark import PERIODO_ESTACIONAL

LAGS_COMPLETOS = (1, 2, 3, PERIODO_ESTACIONAL)
LAGS_CORTOS = (1, 2, 3)
VENTANAS_MEDIA_MOVIL_COMPLETAS = (3, PERIODO_ESTACIONAL)
VENTANAS_MEDIA_MOVIL_CORTAS = (3,)

# Con lag máximo 12, hace falta esta cantidad de filas utilizables después
# de construir las features para no ajustar sobre un puñado de puntos.
# A diferencia de statsmodels, scikit-learn/xgboost no fallan solos ante
# datos escasos, así que la precondición se chequea a mano.
MIN_FILAS_ENTRENAMIENTO = 12


def _elegir_lags(serie: pd.Series) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if len(serie) >= 2 * PERIODO_ESTACIONAL:
        return LAGS_COMPLETOS, VENTANAS_MEDIA_MOVIL_COMPLETAS
    return LAGS_CORTOS, VENTANAS_MEDIA_MOVIL_CORTAS


def _construir_features(serie: pd.Series) -> pd.DataFrame:
    """Una fila por posición `i` de la serie, con lags/medias móviles
    calculados con datos hasta `i` inclusive, más el mes calendario de
    `i` como dummy. Las posiciones sin lag/media móvil completos (no hay
    suficiente historia previa) se descartan."""
    lags, ventanas_media_movil = _elegir_lags(serie)

    features = pd.DataFrame({f"lag_{lag}": serie.shift(lag - 1) for lag in lags})
    for ventana in ventanas_media_movil:
        features[f"media_movil_{ventana}"] = serie.rolling(ventana).mean()
    features["_posicion"] = np.arange(len(serie))
    features = features.dropna()
    features["_posicion"] = features["_posicion"].astype(int)

    meses = serie.index[features["_posicion"].to_numpy()].month
    dummies_mes = pd.get_dummies(meses, prefix="mes", drop_first=True, dtype=float)
    dummies_mes.index = features.index
    return pd.concat([features, dummies_mes], axis=1)


def pronosticar_directo(
    serie: pd.Series, horizonte: int, crear_estimador: Callable[[], object]
) -> np.ndarray:
    """`crear_estimador` construye (sin ajustar) un estimador nuevo por
    cada paso de horizonte — no se reusa estado entre pasos.

    Lanza `ValueError` si no hay suficientes filas utilizables tras
    construir las features (ver `MIN_FILAS_ENTRENAMIENTO`), para que quien
    llama pueda capturarla y hacer fallback."""
    valores = serie.to_numpy(dtype=float)
    features = _construir_features(serie)

    filas_utilizables_peor_caso = len(features) - horizonte
    if filas_utilizables_peor_caso < MIN_FILAS_ENTRENAMIENTO:
        raise ValueError(
            "Historial insuficiente para features de árboles: "
            f"{max(filas_utilizables_peor_caso, 0)} filas utilizables, "
            f"se necesitan al menos {MIN_FILAS_ENTRENAMIENTO}"
        )

    columnas_x = [c for c in features.columns if c != "_posicion"]
    forecast = np.empty(horizonte)

    for h in range(1, horizonte + 1):
        entrenamiento = features[features["_posicion"] + h < len(valores)]
        x_entrenamiento = entrenamiento[columnas_x]
        y_entrenamiento = valores[entrenamiento["_posicion"].to_numpy() + h]

        estimador = crear_estimador()
        estimador.fit(x_entrenamiento, y_entrenamiento)

        x_origen = features.iloc[[-1]][columnas_x]
        forecast[h - 1] = estimador.predict(x_origen)[0]

    return forecast
