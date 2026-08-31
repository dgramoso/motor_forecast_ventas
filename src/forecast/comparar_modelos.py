"""Corre el backtest walk-forward de benchmark vs. modelo por SKU y
devuelve la tabla comparativa — reusable por la selección del mejor
modelo (ver .scratch/motor-forecast-pipeline/issues/04-*) y por la
integración end-to-end.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.datos.cargar_datos import cargar_ventas, serie_por_sku
from src.forecast.backtest import backtest_walk_forward
from src.forecast.benchmark import pronosticar_seasonal_naive
from src.forecast.modelo_ets import _ajustar_ets, pronosticar_ets
from src.forecast.modelo_intermitente import _ajustar_tsb, pronosticar_tsb
from src.forecast.modelo_prophet import _ajustar_prophet, pronosticar_prophet
from src.forecast.modelo_random_forest import _ajustar_random_forest, pronosticar_random_forest
from src.forecast.modelo_xgboost import _ajustar_xgboost, pronosticar_xgboost

HORIZONTE = 3
VENTANA_MINIMA = 24


def _sin_negativos(funcion):
    """Las unidades vendidas/pronosticadas nunca son negativas — un
    modelo puede extrapolar por debajo de cero (tendencia decreciente,
    poca historia), pero eso no es una cantidad física válida. Se aplica
    acá, una sola vez para los 6 candidatos, en vez de duplicarlo en cada
    modelo_X.py."""

    def envoltorio(serie, horizonte):
        return np.maximum(funcion(serie, horizonte), 0.0)

    return envoltorio


def _sin_negativos_con_metadata(funcion):
    def envoltorio(serie, horizonte):
        forecast, fallback, motivo = funcion(serie, horizonte)
        return np.maximum(forecast, 0.0), fallback, motivo

    return envoltorio


# Candidatos que compiten por el título de "mejor modelo" — el benchmark
# participa en el mismo ranking, no es un caso especial (spec.md:19,
# decisión del ticket 01). ETS y TSB compiten como candidatos
# independientes: cuál conviene para cada SKU lo decide el backtest, no
# una regla fija de intermitencia (ver docs/adr/0002-ets-tsb-por-backtest.md
# y diagnostico_demanda.py, que clasifica el patrón de demanda solo como
# diagnóstico). SARIMA no compite (ver docs/adr/0001-no-sarima.md).
CANDIDATOS = {
    nombre: _sin_negativos(funcion)
    for nombre, funcion in {
        "benchmark": pronosticar_seasonal_naive,
        "ets": pronosticar_ets,
        "tsb": pronosticar_tsb,
        "xgboost": pronosticar_xgboost,
        "prophet": pronosticar_prophet,
        "random_forest": pronosticar_random_forest,
    }.items()
}


def _ajustar_benchmark(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, Optional[str]]:
    """El benchmark es el destino del fallback de los demás — nunca
    cae en fallback él mismo."""
    return pronosticar_seasonal_naive(serie, horizonte), False, None


# Igual que CANDIDATOS, pero cada función devuelve también si hubo
# fallback y el motivo (ver CONTEXT.md, "Fallback" / "Motivo de
# fallback") — lo consumen comparar_modelos_sku (para la tasa de
# fallback del backtest) y pronosticar_futuro_sku (para saber si lo que
# se sirve es el candidato real o el benchmark disfrazado). CANDIDATOS
# se mantiene como el contrato público simple, sin metadata.
CANDIDATOS_CON_METADATA = {
    nombre: _sin_negativos_con_metadata(funcion)
    for nombre, funcion in {
        "benchmark": _ajustar_benchmark,
        "ets": _ajustar_ets,
        "tsb": _ajustar_tsb,
        "xgboost": _ajustar_xgboost,
        "prophet": _ajustar_prophet,
        "random_forest": _ajustar_random_forest,
    }.items()
}


def comparar_modelos_sku(
    serie: pd.Series,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    candidatos: dict = CANDIDATOS_CON_METADATA,
) -> pd.DataFrame:
    """Una fila por candidato, con sus métricas agregadas del backtest y
    su tasa de fallback (ver CONTEXT.md, "Tasa de fallback"). `candidatos`
    es un seam de inyección — el default es `CANDIDATOS_CON_METADATA`;
    los tests pasan un dict propio con doubles rápidos en vez de correr
    los 6 modelos reales (Prophet incluido)."""
    filas = []
    for nombre, ajustar_con_metadata in candidatos.items():
        fallbacks = []

        def funcion_pronostico(entrenamiento, h, _ajustar=ajustar_con_metadata, _fallbacks=fallbacks):
            forecast, fallback, _motivo = _ajustar(entrenamiento, h)
            _fallbacks.append(fallback)
            return forecast

        resultados = backtest_walk_forward(serie, funcion_pronostico, horizonte, ventana_minima)
        filas.append(
            {
                "candidato": nombre,
                "n_ventanas": len(resultados),
                "wape_indefinido": resultados["wape"].isna().sum(),
                "wape_medio": resultados["wape"].mean(),
                "bias_medio": resultados["bias"].mean(),
                "mae_medio": resultados["mae"].mean(),
                "mase_medio": resultados["mase"].mean(),
                "tasa_fallback_backtest": float(np.mean(fallbacks)) if fallbacks else 0.0,
            }
        )
    return pd.DataFrame(filas)


def comparar_modelos(
    ventas: pd.DataFrame,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    candidatos: dict = CANDIDATOS_CON_METADATA,
) -> pd.DataFrame:
    """Igual que `comparar_modelos_sku`, para todos los SKUs de `ventas`."""
    tablas = []
    for sku_id in ventas["sku_id"].unique():
        serie = serie_por_sku(ventas, sku_id)
        tabla_sku = comparar_modelos_sku(serie, horizonte, ventana_minima, candidatos)
        tabla_sku.insert(0, "sku_id", sku_id)
        tablas.append(tabla_sku)
    return pd.concat(tablas, ignore_index=True)
