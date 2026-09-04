"""Selección del mejor modelo por SKU — criterio decidido en
.scratch/motor-forecast-pipeline/issues/01-criterio-seleccion-mejor-modelo.md:
menor WAPE medio del backtest: el benchmark compite en el mismo ranking
que el modelo (gana por default si nadie le gana), desempate por menor
|Bias medio|.
"""

import pandas as pd

DECIMALES_EMPATE = 3


def seleccionar_mejor_modelo_sku(tabla_comparativa_sku: pd.DataFrame) -> dict:
    """`tabla_comparativa_sku` es la salida de `comparar_modelos_sku` para un SKU.

    Si el `wape_medio` del ganador es `NaN`, el SKU no tuvo ninguna
    ventana del backtest con demanda real distinta de cero — el WAPE
    queda indefinido para los 6 candidatos por igual (ver
    metricas.wape), así que no hubo comparación real entre ellos.
    `sort_values` deja ganar al primero del diccionario `CANDIDATOS` en
    ese caso (orden estable, todos empatados en NaN/NaN) — no es una
    elección con fundamento, así que queda marcado explícitamente en
    `sin_datos_suficientes` en vez de reportarse como un ganador más."""
    tabla = tabla_comparativa_sku.copy()
    tabla["wape_redondeado"] = tabla["wape_medio"].round(DECIMALES_EMPATE)
    tabla["bias_abs"] = tabla["bias_medio"].abs()

    ganador = tabla.sort_values(["wape_redondeado", "bias_abs"]).iloc[0]

    resultado = {
        "candidato": ganador["candidato"],
        "wape_medio": ganador["wape_medio"],
        "bias_medio": ganador["bias_medio"],
        "mae_medio": ganador["mae_medio"],
        "mase_medio": ganador["mase_medio"],
        "tasa_fallback_backtest": ganador["tasa_fallback_backtest"],
        "sin_datos_suficientes": bool(pd.isna(ganador["wape_medio"])),
    }
    # Si el ganador trae sus propios parámetros (p.ej. los pesos del
    # candidato "ensemble", columnas `peso_*` de comparar_modelos_con_ensemble),
    # se propagan acá — quedan en la corrida persistida (guardar_corrida)
    # en vez de perderse en la selección. Ausentes para el resto de los
    # candidatos, que no tienen parámetros propios que registrar.
    resultado.update(
        {columna: ganador[columna] for columna in tabla.columns if columna.startswith("peso_") and pd.notna(ganador[columna])}
    )
    return resultado


def seleccionar_mejor_modelo(tabla_comparativa: pd.DataFrame) -> pd.DataFrame:
    """`tabla_comparativa` es la salida de `comparar_modelos` (todos los SKUs)."""
    filas = []
    for sku_id, tabla_sku in tabla_comparativa.groupby("sku_id"):
        seleccion = seleccionar_mejor_modelo_sku(tabla_sku)
        seleccion["sku_id"] = sku_id
        filas.append(seleccion)

    columnas_base = [
        "sku_id",
        "candidato",
        "wape_medio",
        "bias_medio",
        "mae_medio",
        "mase_medio",
        "tasa_fallback_backtest",
        "sin_datos_suficientes",
    ]
    # Columnas `peso_*` sólo si algún SKU ganó con un candidato que las
    # trae (ver seleccionar_mejor_modelo_sku) — no todas las corridas
    # tienen un ganador "ensemble".
    columnas_peso = sorted({columna for fila in filas for columna in fila if columna.startswith("peso_")})
    return pd.DataFrame(filas)[columnas_base + columnas_peso]
