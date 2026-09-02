"""Backtest walk-forward del candidato LightGBM global — `comparar_modelos_sku`
evalúa una serie a la vez y no sirve acá: LightGBM entrena UN modelo con
el histórico de TODAS las SKUs hasta cada origen, no un modelo por SKU
(ver modelo_lightgbm_global.py y la sección 3 del pedido de arquitectura
a escala: nada de 50.000 modelos independientes). Este módulo reentrena
el modelo global una vez por origen del walk-forward, no una vez por SKU,
y evalúa a todas las SKUs de una sola pasada en cada origen.

Asume que todas las SKUs comparten el mismo calendario — walk-forward
por fecha, no por posición individual de cada serie. Es una
simplificación conocida (spec.md no define altas/bajas de SKU en el
tiempo); con SKUs de distinta antigüedad habría que revisar esta
suposición, no está resuelto acá.

Si el ajuste global falla en algún origen (p.ej. sin filas de
entrenamiento por falta de historia), cae a Seasonal Naive para todas
las SKUs en ese origen — el fallback es a nivel del modelo global, no
por SKU, porque la falla también lo es.
"""

from typing import Optional

import numpy as np
import pandas as pd

from .benchmark import PERIODO_ESTACIONAL, pronosticar_seasonal_naive
from .comparar_modelos import HORIZONTE, VENTANA_MINIMA, comparar_modelos
from .features_lightgbm import LAGS, VENTANAS_ROLLING, construir_dataset_supervisado
from .metricas import bias, mae, mase, wape
from .modelo_lightgbm_global import entrenar_lightgbm_global, pronosticar_lightgbm_global

NOMBRE_CANDIDATO = "lightgbm_global"


def _pronosticar_origen(
    historico: pd.DataFrame,
    horizonte: int,
    incluir_sku_id: bool,
    lags: tuple[int, ...],
    ventanas_rolling: tuple[int, ...],
) -> tuple[pd.DataFrame, bool, Optional[str]]:
    """Entrena y predice para un origen. Devuelve una tabla indexada por
    `sku_id` con una columna por paso de horizonte (1..horizonte). Si
    falla (p.ej. sin filas de entrenamiento por falta de historia), cae a
    Seasonal Naive por SKU y lo marca como fallback."""
    try:
        dataset = construir_dataset_supervisado(historico, horizonte, lags, ventanas_rolling)
        modelos = entrenar_lightgbm_global(dataset, horizonte, incluir_sku_id)
        pronostico_largo = pronosticar_lightgbm_global(modelos, dataset, incluir_sku_id)
        pronostico = pronostico_largo.pivot(index="sku_id", columns="paso_horizonte", values="unidades_pronosticadas")
        return pronostico, False, None
    except (ValueError, LookupError) as error:
        motivo = f"{type(error).__name__}: {error}"
        fallback = {
            sku_id: pronosticar_seasonal_naive(
                grupo.sort_values("fecha").set_index("fecha")["unidades_vendidas"].asfreq("MS"), horizonte
            )
            for sku_id, grupo in historico.groupby("sku_id")
        }
        pronostico = pd.DataFrame(fallback).T
        pronostico.columns = range(1, horizonte + 1)
        return pronostico, True, motivo


def _iterar_predicciones_por_sku(
    ventas: pd.DataFrame,
    horizonte: int,
    ventana_minima: int,
    incluir_sku_id: bool,
    lags: tuple[int, ...],
    ventanas_rolling: tuple[int, ...],
):
    """Un origen del walk-forward por vez: reentrena el modelo global una
    sola vez (no por SKU) y va cediendo `(sku_id, real, pronostico,
    fallback)` para cada SKU con historia en ese origen. Generador
    compartido por `backtest_lightgbm_global` (agrega a métricas) y
    `recolectar_predicciones_lightgbm_global` (conserva los arrays crudos
    para `ensemble.py`) — evita correr el walk-forward dos veces."""
    fechas = sorted(ventas["fecha"].unique())
    ultimo_origen = len(fechas) - horizonte

    for origen_idx in range(ventana_minima, ultimo_origen + 1):
        fecha_origen = fechas[origen_idx - 1]
        historico = ventas[ventas["fecha"] <= fecha_origen]
        pronostico, fallback, _motivo = _pronosticar_origen(
            historico, horizonte, incluir_sku_id, lags, ventanas_rolling
        )

        fechas_horizonte = fechas[origen_idx : origen_idx + horizonte]
        for sku_id in ventas["sku_id"].unique():
            if sku_id not in pronostico.index:
                continue  # SKU sin historia todavía en este origen

            real = (
                ventas[(ventas["sku_id"] == sku_id) & (ventas["fecha"].isin(fechas_horizonte))]
                .sort_values("fecha")["unidades_vendidas"]
                .to_numpy()
            )
            pron = pronostico.loc[sku_id, range(1, horizonte + 1)].to_numpy(dtype=float)
            yield sku_id, real, pron, fallback, historico


def backtest_lightgbm_global(
    ventas: pd.DataFrame,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    incluir_sku_id: bool = False,
    lags: tuple[int, ...] = LAGS,
    ventanas_rolling: tuple[int, ...] = VENTANAS_ROLLING,
) -> pd.DataFrame:
    """Una fila por SKU, con las mismas columnas que produce
    `comparar_modelos_sku` para cada candidato — para poder concatenar
    con `comparar_modelos(ventas)` antes de seleccionar el mejor
    candidato (ver `comparar_modelos_con_lightgbm_global`).
    `candidato` queda fijo en `"lightgbm_global"`."""
    metricas_por_sku: dict = {sku_id: [] for sku_id in ventas["sku_id"].unique()}
    fallbacks_por_sku: dict = {sku_id: [] for sku_id in ventas["sku_id"].unique()}

    for sku_id, real, pron, fallback, historico in _iterar_predicciones_por_sku(
        ventas, horizonte, ventana_minima, incluir_sku_id, lags, ventanas_rolling
    ):
        entrenamiento_sku = (
            historico[historico["sku_id"] == sku_id].sort_values("fecha")["unidades_vendidas"].to_numpy()
        )
        metricas_por_sku[sku_id].append(
            {
                "wape": wape(real, pron),
                "bias": bias(real, pron),
                "mae": mae(real, pron),
                "mase": mase(real, pron, entrenamiento_sku, PERIODO_ESTACIONAL),
            }
        )
        fallbacks_por_sku[sku_id].append(fallback)

    filas = []
    for sku_id, resultados in metricas_por_sku.items():
        tabla = pd.DataFrame(resultados)
        fallbacks = fallbacks_por_sku[sku_id]
        filas.append(
            {
                "sku_id": sku_id,
                "candidato": NOMBRE_CANDIDATO,
                "n_ventanas": len(tabla),
                "wape_indefinido": int(tabla["wape"].isna().sum()) if len(tabla) else 0,
                "wape_medio": tabla["wape"].mean() if len(tabla) else np.nan,
                "bias_medio": tabla["bias"].mean() if len(tabla) else np.nan,
                "mae_medio": tabla["mae"].mean() if len(tabla) else np.nan,
                "mase_medio": tabla["mase"].mean() if len(tabla) else np.nan,
                "tasa_fallback_backtest": float(np.mean(fallbacks)) if fallbacks else 0.0,
            }
        )

    return pd.DataFrame(filas)


def recolectar_predicciones_lightgbm_global(
    ventas: pd.DataFrame,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    incluir_sku_id: bool = False,
    lags: tuple[int, ...] = LAGS,
    ventanas_rolling: tuple[int, ...] = VENTANAS_ROLLING,
) -> dict[str, tuple[list[np.ndarray], list[np.ndarray]]]:
    """Predicciones out-of-sample crudas del walk-forward, por SKU:
    `{sku_id: (reales, pronosticos)}` — las usa `ensemble.py` para ajustar
    pesos con las mismas ventanas que ya evaluó el backtest, sin
    reentrenar de nuevo."""
    reales_por_sku: dict = {sku_id: [] for sku_id in ventas["sku_id"].unique()}
    pronosticos_por_sku: dict = {sku_id: [] for sku_id in ventas["sku_id"].unique()}

    for sku_id, real, pron, _fallback, _historico in _iterar_predicciones_por_sku(
        ventas, horizonte, ventana_minima, incluir_sku_id, lags, ventanas_rolling
    ):
        reales_por_sku[sku_id].append(real)
        pronosticos_por_sku[sku_id].append(pron)

    return {sku_id: (reales_por_sku[sku_id], pronosticos_por_sku[sku_id]) for sku_id in ventas["sku_id"].unique()}


def comparar_modelos_con_lightgbm_global(
    ventas: pd.DataFrame,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    incluir_sku_id: bool = False,
) -> pd.DataFrame:
    """`comparar_modelos(ventas)` (candidatos por-SKU: benchmark, ETS,
    TSB, XGBoost, Prophet, Random Forest) más el candidato LightGBM
    global, en una sola tabla — lista para `seleccionar_modelo.py` sin
    cambiarlo."""
    tabla_por_sku = comparar_modelos(ventas, horizonte, ventana_minima)
    tabla_global = backtest_lightgbm_global(ventas, horizonte, ventana_minima, incluir_sku_id)
    return pd.concat([tabla_por_sku, tabla_global], ignore_index=True)
