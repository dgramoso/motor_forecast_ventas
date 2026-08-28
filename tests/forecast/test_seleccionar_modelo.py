"""Tests de seleccionar_modelo.py: criterio de selección (menor WAPE
medio, desempate por |Bias medio|) y el caso límite donde ningún
candidato tiene un WAPE definido para el SKU (todas las ventanas del
backtest sin demanda real) — ver CONTEXT.md, "SKU sin datos suficientes
para comparar"."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.seleccionar_modelo import seleccionar_mejor_modelo_sku


def _tabla(filas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(filas)


class TestCriterioDeSeleccion(unittest.TestCase):
    def test_gana_el_candidato_con_menor_wape_medio(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": 0.5, "bias_medio": 0.1, "mae_medio": 10.0, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": 0.3, "bias_medio": -0.2, "mae_medio": 8.0, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertEqual(seleccion["candidato"], "ets_tsb")
        self.assertFalse(seleccion["sin_datos_suficientes"])

    def test_desempata_por_menor_bias_absoluto(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": 0.300, "bias_medio": 0.05, "mae_medio": 10.0, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": 0.3001, "bias_medio": -0.20, "mae_medio": 8.0, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertEqual(seleccion["candidato"], "benchmark")


class TestSinDatosSuficientes(unittest.TestCase):
    def test_todos_nan_marca_sin_datos_suficientes(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": np.nan, "bias_medio": np.nan, "mae_medio": 0.0, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": np.nan, "bias_medio": np.nan, "mae_medio": 0.0, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertTrue(seleccion["sin_datos_suficientes"])
        # Gana el primero del dict (orden estable) — no hay fundamento
        # real, pero el resultado queda marcado en vez de escondido.
        self.assertEqual(seleccion["candidato"], "benchmark")

    def test_un_solo_candidato_definido_no_marca_sin_datos_suficientes(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": np.nan, "bias_medio": np.nan, "mae_medio": 0.0, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": 0.4, "bias_medio": 0.1, "mae_medio": 5.0, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertFalse(seleccion["sin_datos_suficientes"])
        self.assertEqual(seleccion["candidato"], "ets_tsb")


if __name__ == "__main__":
    unittest.main()
