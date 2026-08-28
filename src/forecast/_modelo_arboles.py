"""Utilidades compartidas por los modelos basados en árboles (XGBoost,
Random Forest) — construcción de features y pronóstico directo por
horizonte. No es un candidato en sí mismo: lo consumen `modelo_xgboost.py`
y `modelo_random_forest.py`.

Pronóstico directo, no recursivo: un estimador independiente por paso de
horizonte, para que el error de un paso no contamine los siguientes.
Todos ven el mismo set de features — lags 1, 2, 3, 12 (últimos 3 meses +
mismo mes del año anterior, mismo `PERIODO_ESTACIONAL=12` que el resto
del proyecto), medias móviles de 3 y 12 meses, y el mes calendario del
período a pronosticar como dummy (mismo patrón que
`benchmark.estimar_tendencia`) — pero cada uno con el target desplazado
`h` pasos.
"""

from typing import Callable

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 12)
VENTANAS_MEDIA_MOVIL = (3, 12)

# Con lag máximo 12, hace falta esta cantidad de filas utilizables después
# de construir las features para no ajustar sobre un puñado de puntos.
# A diferencia de statsmodels, scikit-learn/xgboost no fallan solos ante
# datos escasos, así que la precondición se chequea a mano.
MIN_FILAS_ENTRENAMIENTO = 12

_LAG_MAXIMO = max(LAGS)


def _construir_features(serie: pd.Series) -> pd.DataFrame:
    """Una fila por posición `i` de la serie, con lags/medias móviles
    calculados con datos hasta `i` inclusive, más el mes calendario de
    `i` como dummy. Las primeras `_LAG_MAXIMO - 1` posiciones no tienen
    lag completo y se descartan."""
    valores = serie.to_numpy(dtype=float)
    n = len(valores)

    filas = []
    for i in range(_LAG_MAXIMO - 1, n):
        fila = {f"lag_{lag}": valores[i - lag + 1] for lag in LAGS}
        for ventana in VENTANAS_MEDIA_MOVIL:
            fila[f"media_movil_{ventana}"] = valores[i - ventana + 1 : i + 1].mean()
        fila["_posicion"] = i
        filas.append(fila)

    features = pd.DataFrame(filas)
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
