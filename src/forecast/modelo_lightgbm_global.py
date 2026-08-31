"""LightGBM global — UN modelo por paso de horizonte, entrenado con las
observaciones de TODAS las SKUs apiladas (ver sección 3 del pedido: nada
de un modelo por SKU, eso no escala a 50.000 SKUs). Misma estrategia
directa que XGBoost/Random Forest (`_modelo_arboles.py`): un estimador
por paso, no recursivo — pero acá "por paso" es un LightGBM entrenado una
sola vez con todo el dataset, no un estimador por serie.

`incluir_sku_id`: comparación Modelo A (sin identidad de SKU) vs. Modelo
B (sección 8 del pedido) — cuál conviene se decide por backtest, no acá.
Cuando se usa, el SKU entra como categórica nativa de LightGBM (no un
entero continuo) — hay que pasar el mismo `dataset` completo a
`entrenar_lightgbm_global` y a `pronosticar_lightgbm_global` para que las
categorías de `sku_id` coincidan entre entrenamiento y predicción.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

SEMILLA_ALEATORIA = 42
N_ESTIMATORS = 200
MAX_DEPTH = 3
LEARNING_RATE = 0.1


def _crear_estimador() -> LGBMRegressor:
    return LGBMRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        random_state=SEMILLA_ALEATORIA,
        verbosity=-1,
    )


def _columnas_features(dataset: pd.DataFrame) -> list[str]:
    return [c for c in dataset.columns if c.startswith(("lag_", "rolling_")) or c == "mes_objetivo"]


def _matriz_x(dataset: pd.DataFrame, incluir_sku_id: bool) -> pd.DataFrame:
    """`sku_id` como categórica de pandas con las categorías de TODO
    `dataset` (no solo del subconjunto de filas de `x`), para que
    entrenamiento y predicción usen la misma codificación."""
    x = dataset[_columnas_features(dataset)].copy()
    if incluir_sku_id:
        x["sku_id"] = pd.Categorical(dataset["sku_id"], categories=sorted(dataset["sku_id"].unique()))
    return x


def entrenar_lightgbm_global(
    dataset: pd.DataFrame, horizonte: int, incluir_sku_id: bool = False
) -> dict[int, LGBMRegressor]:
    """Un LightGBM por paso de horizonte, cada uno entrenado con las filas
    de todas las SKUs para ese paso. `dataset` es la salida de
    `features_lightgbm.construir_dataset_supervisado`. Filas con `target`
    `NaN` (esa fecha futura todavía no ocurrió) se excluyen del
    entrenamiento."""
    modelos = {}
    for paso in range(1, horizonte + 1):
        entrenamiento = dataset[(dataset["paso_horizonte"] == paso) & dataset["target"].notna()]
        x = _matriz_x(entrenamiento, incluir_sku_id)

        estimador = _crear_estimador()
        estimador.fit(x, entrenamiento["target"])
        modelos[paso] = estimador

    return modelos


def pronosticar_lightgbm_global(
    modelos: dict[int, LGBMRegressor], dataset: pd.DataFrame, incluir_sku_id: bool = False
) -> pd.DataFrame:
    """Pronóstico para el último origen disponible de cada SKU en
    `dataset`, usando los modelos ya entrenados (no reentrena nada).
    Devuelve `sku_id`, `paso_horizonte`, `unidades_pronosticadas` — nunca
    negativo (igual que el resto de los candidatos, ver
    `comparar_modelos._sin_negativos`)."""
    ultimo_origen = dataset.groupby("sku_id")["fecha"].transform("max") == dataset["fecha"]

    filas = []
    for paso, estimador in modelos.items():
        bloque = dataset[ultimo_origen & (dataset["paso_horizonte"] == paso)]
        x = _matriz_x(bloque, incluir_sku_id)
        pronostico = np.maximum(estimador.predict(x), 0.0)
        filas.append(
            pd.DataFrame(
                {
                    "sku_id": bloque["sku_id"].to_numpy(),
                    "paso_horizonte": paso,
                    "unidades_pronosticadas": pronostico,
                }
            )
        )

    return pd.concat(filas, ignore_index=True).sort_values(["sku_id", "paso_horizonte"]).reset_index(drop=True)
