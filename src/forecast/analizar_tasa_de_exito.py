"""Calcula, por SKU, el % de ventanas del backtest donde el modelo le
gana al benchmark en WAPE — el dato que falta para cerrar spec.md:34
(qué % de ventanas define "consistentemente").
"""

import pandas as pd

from src.datos.cargar_datos import cargar_ventas, serie_por_sku
from src.forecast.backtest import backtest_walk_forward
from src.forecast.benchmark import pronosticar_seasonal_naive
from src.forecast.modelo import pronosticar_modelo

HORIZONTE = 3
VENTANA_MINIMA = 24


def main() -> None:
    ventas = cargar_ventas()
    filas = []

    for sku_id in ventas["sku_id"].unique():
        serie = serie_por_sku(ventas, sku_id)

        benchmark = backtest_walk_forward(serie, pronosticar_seasonal_naive, HORIZONTE, VENTANA_MINIMA)
        modelo = backtest_walk_forward(serie, pronosticar_modelo, HORIZONTE, VENTANA_MINIMA)

        comparacion = pd.DataFrame(
            {
                "wape_benchmark": benchmark["wape"],
                "wape_modelo": modelo["wape"],
            }
        ).dropna()

        gana_modelo = comparacion["wape_modelo"] < comparacion["wape_benchmark"]

        filas.append(
            {
                "sku_id": sku_id,
                "ventanas_comparables": len(comparacion),
                "pct_ventanas_gana_modelo": gana_modelo.mean(),
            }
        )

    tabla = pd.DataFrame(filas).round(3)
    print(tabla.to_string(index=False))
    print(f"\nPromedio simple entre SKUs: {tabla['pct_ventanas_gana_modelo'].mean():.3f}")


if __name__ == "__main__":
    main()
