"""Corre el backtest walk-forward de benchmark vs. modelo ETS sobre
los 5 SKUs sintéticos y muestra el resumen comparativo por SKU.
"""

import pandas as pd

from src.datos.cargar_datos import cargar_ventas, serie_por_sku
from src.forecast.backtest import backtest_walk_forward
from src.forecast.benchmark import pronosticar_seasonal_naive, tiene_tendencia
from src.forecast.modelo import pronosticar_modelo

HORIZONTE = 3
VENTANA_MINIMA = 24


def _resumir(resultados: pd.DataFrame) -> dict:
    return {
        "n_ventanas": len(resultados),
        "wape_indefinido": resultados["wape"].isna().sum(),
        "wape_medio": resultados["wape"].mean(),
        "bias_medio": resultados["bias"].mean(),
        "mae_medio": resultados["mae"].mean(),
    }


def main() -> None:
    ventas = cargar_ventas()
    filas = []

    for sku_id in ventas["sku_id"].unique():
        serie = serie_por_sku(ventas, sku_id)

        benchmark = backtest_walk_forward(serie, pronosticar_seasonal_naive, HORIZONTE, VENTANA_MINIMA)
        modelo = backtest_walk_forward(serie, pronosticar_modelo, HORIZONTE, VENTANA_MINIMA)

        fila = {"sku_id": sku_id, "tendencia": tiene_tendencia(serie)}
        for nombre, valor in _resumir(benchmark).items():
            fila[f"benchmark_{nombre}"] = valor
        for nombre, valor in _resumir(modelo).items():
            fila[f"modelo_{nombre}"] = valor
        fila["modelo_le_gana_a_benchmark_wape"] = fila["modelo_wape_medio"] < fila["benchmark_wape_medio"]
        filas.append(fila)

    tabla = pd.DataFrame(filas).round(3)
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
