"""Selección del mejor modelo por SKU — criterio decidido en
.scratch/motor-forecast-pipeline/issues/01-criterio-seleccion-mejor-modelo.md:
menor WAPE medio del backtest: el benchmark compite en el mismo ranking
que el modelo (gana por default si nadie le gana), desempate por menor
|Bias medio|.
"""

import pandas as pd

DECIMALES_EMPATE = 3


def seleccionar_mejor_modelo_sku(tabla_comparativa_sku: pd.DataFrame) -> dict:
    """`tabla_comparativa_sku` es la salida de `comparar_modelos_sku` para un SKU."""
    tabla = tabla_comparativa_sku.copy()
    tabla["wape_redondeado"] = tabla["wape_medio"].round(DECIMALES_EMPATE)
    tabla["bias_abs"] = tabla["bias_medio"].abs()

    ganador = tabla.sort_values(["wape_redondeado", "bias_abs"]).iloc[0]

    return {
        "candidato": ganador["candidato"],
        "wape_medio": ganador["wape_medio"],
        "bias_medio": ganador["bias_medio"],
        "mae_medio": ganador["mae_medio"],
    }


def seleccionar_mejor_modelo(tabla_comparativa: pd.DataFrame) -> pd.DataFrame:
    """`tabla_comparativa` es la salida de `comparar_modelos` (todos los SKUs)."""
    filas = []
    for sku_id, tabla_sku in tabla_comparativa.groupby("sku_id"):
        seleccion = seleccionar_mejor_modelo_sku(tabla_sku)
        seleccion["sku_id"] = sku_id
        filas.append(seleccion)

    columnas = ["sku_id", "candidato", "wape_medio", "bias_medio", "mae_medio"]
    return pd.DataFrame(filas)[columnas]
