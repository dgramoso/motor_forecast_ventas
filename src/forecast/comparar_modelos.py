"""Corre el backtest walk-forward de benchmark vs. modelo por SKU y
devuelve la tabla comparativa — reusable por la selección del mejor
modelo (ver .scratch/motor-forecast-pipeline/issues/04-*) y por la
integración end-to-end.
"""

import pandas as pd

from src.datos.cargar_datos import cargar_ventas, serie_por_sku
from src.forecast.backtest import backtest_walk_forward
from src.forecast.benchmark import pronosticar_seasonal_naive
from src.forecast.modelo import pronosticar_modelo
from src.forecast.modelo_prophet import pronosticar_prophet
from src.forecast.modelo_random_forest import pronosticar_random_forest
from src.forecast.modelo_sarima import pronosticar_sarima
from src.forecast.modelo_xgboost import pronosticar_xgboost

HORIZONTE = 3
VENTANA_MINIMA = 24

# Candidatos que compiten por el título de "mejor modelo" — el benchmark
# participa en el mismo ranking, no es un caso especial (spec.md:19,
# decisión del ticket 01). "ets_tsb" es el router de modelo.py (ETS o TSB
# según intermitencia, ver modelo.py); se llama así y no "modelo" para no
# quedar ambiguo junto a los demás candidatos, que también son modelos
# (ver issue 08).
CANDIDATOS = {
    "benchmark": pronosticar_seasonal_naive,
    "ets_tsb": pronosticar_modelo,
    "sarima": pronosticar_sarima,
    "xgboost": pronosticar_xgboost,
    "prophet": pronosticar_prophet,
    "random_forest": pronosticar_random_forest,
}


def comparar_modelos_sku(
    serie: pd.Series, horizonte: int = HORIZONTE, ventana_minima: int = VENTANA_MINIMA
) -> pd.DataFrame:
    """Una fila por candidato, con sus métricas agregadas del backtest."""
    filas = []
    for nombre, funcion_pronostico in CANDIDATOS.items():
        resultados = backtest_walk_forward(serie, funcion_pronostico, horizonte, ventana_minima)
        filas.append(
            {
                "candidato": nombre,
                "n_ventanas": len(resultados),
                "wape_indefinido": resultados["wape"].isna().sum(),
                "wape_medio": resultados["wape"].mean(),
                "bias_medio": resultados["bias"].mean(),
                "mae_medio": resultados["mae"].mean(),
            }
        )
    return pd.DataFrame(filas)


def comparar_modelos(
    ventas: pd.DataFrame, horizonte: int = HORIZONTE, ventana_minima: int = VENTANA_MINIMA
) -> pd.DataFrame:
    """Igual que `comparar_modelos_sku`, para todos los SKUs de `ventas`."""
    tablas = []
    for sku_id in ventas["sku_id"].unique():
        serie = serie_por_sku(ventas, sku_id)
        tabla_sku = comparar_modelos_sku(serie, horizonte, ventana_minima)
        tabla_sku.insert(0, "sku_id", sku_id)
        tablas.append(tabla_sku)
    return pd.concat(tablas, ignore_index=True)
