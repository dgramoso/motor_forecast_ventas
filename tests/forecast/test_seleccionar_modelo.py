"""Tests de seleccionar_modelo.py: criterio de selección (menor WAPE
medio, desempate por |Bias medio|) y el caso límite donde ningún
candidato tiene un WAPE definido para el SKU (todas las ventanas del
backtest sin demanda real) — ver CONTEXT.md, "SKU sin datos suficientes
para comparar"."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.seleccionar_modelo import seleccionar_mejor_modelo, seleccionar_mejor_modelo_sku


def _tabla(filas: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(filas)


class TestCriterioDeSeleccion(unittest.TestCase):
    def test_gana_el_candidato_con_menor_wape_medio(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": 0.5, "bias_medio": 0.1, "mae_medio": 10.0, "mase_medio": 1.2, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": 0.3, "bias_medio": -0.2, "mae_medio": 8.0, "mase_medio": 0.9, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertEqual(seleccion["candidato"], "ets_tsb")
        self.assertFalse(seleccion["sin_datos_suficientes"])

    def test_desempata_por_menor_bias_absoluto(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": 0.300, "bias_medio": 0.05, "mae_medio": 10.0, "mase_medio": 1.0, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": 0.3001, "bias_medio": -0.20, "mae_medio": 8.0, "mase_medio": 0.95, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertEqual(seleccion["candidato"], "benchmark")


class TestSinDatosSuficientes(unittest.TestCase):
    def test_todos_nan_marca_sin_datos_suficientes(self):
        tabla = _tabla(
            [
                {"candidato": "benchmark", "wape_medio": np.nan, "bias_medio": np.nan, "mae_medio": 0.0, "mase_medio": np.nan, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": np.nan, "bias_medio": np.nan, "mae_medio": 0.0, "mase_medio": np.nan, "tasa_fallback_backtest": 0.0},
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
                {"candidato": "benchmark", "wape_medio": np.nan, "bias_medio": np.nan, "mae_medio": 0.0, "mase_medio": np.nan, "tasa_fallback_backtest": 0.0},
                {"candidato": "ets_tsb", "wape_medio": 0.4, "bias_medio": 0.1, "mae_medio": 5.0, "mase_medio": 1.1, "tasa_fallback_backtest": 0.0},
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertFalse(seleccion["sin_datos_suficientes"])
        self.assertEqual(seleccion["candidato"], "ets_tsb")


class TestParametrosPropiosDelGanador(unittest.TestCase):
    """Cuando el ganador trae columnas `peso_*` (el candidato "ensemble",
    ver ensemble_backtest.py), seleccionar_mejor_modelo_sku las propaga
    en vez de perderlas — son los parámetros del modelo que se sirvió,
    necesarios para reproducir el pronóstico desde lo persistido (ver
    CONTEXT.md / guardar_corrida)."""

    def test_propaga_los_pesos_del_ganador_ensemble(self):
        tabla = _tabla(
            [
                {
                    "candidato": "benchmark", "wape_medio": 0.5, "bias_medio": 0.1, "mae_medio": 10.0,
                    "mase_medio": 1.2, "tasa_fallback_backtest": 0.0,
                    "peso_ets": np.nan, "peso_tsb": np.nan, "peso_lightgbm_global": np.nan,
                },
                {
                    "candidato": "ensemble", "wape_medio": 0.1, "bias_medio": 0.0, "mae_medio": 2.0,
                    "mase_medio": 0.5, "tasa_fallback_backtest": 0.0,
                    "peso_ets": 0.3, "peso_tsb": 0.5, "peso_lightgbm_global": 0.2,
                },
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertEqual(seleccion["candidato"], "ensemble")
        self.assertEqual(seleccion["peso_ets"], 0.3)
        self.assertEqual(seleccion["peso_tsb"], 0.5)
        self.assertEqual(seleccion["peso_lightgbm_global"], 0.2)

    def test_no_agrega_pesos_si_el_ganador_no_es_ensemble(self):
        tabla = _tabla(
            [
                {
                    "candidato": "benchmark", "wape_medio": 0.1, "bias_medio": 0.0, "mae_medio": 2.0,
                    "mase_medio": 0.5, "tasa_fallback_backtest": 0.0,
                    "peso_ets": np.nan, "peso_tsb": np.nan, "peso_lightgbm_global": np.nan,
                },
                {
                    "candidato": "ensemble", "wape_medio": 0.5, "bias_medio": 0.1, "mae_medio": 10.0,
                    "mase_medio": 1.2, "tasa_fallback_backtest": 0.0,
                    "peso_ets": 0.3, "peso_tsb": 0.5, "peso_lightgbm_global": 0.2,
                },
            ]
        )

        seleccion = seleccionar_mejor_modelo_sku(tabla)

        self.assertEqual(seleccion["candidato"], "benchmark")
        self.assertNotIn("peso_ets", seleccion)

    def test_seleccionar_mejor_modelo_no_pierde_los_pesos_en_la_tabla_final(self):
        tabla = pd.concat(
            [
                _tabla(
                    [
                        {
                            "sku_id": "SKU-0", "candidato": "ensemble", "wape_medio": 0.1, "bias_medio": 0.0,
                            "mae_medio": 2.0, "mase_medio": 0.5, "tasa_fallback_backtest": 0.0,
                            "peso_ets": 0.3, "peso_tsb": 0.5, "peso_lightgbm_global": 0.2,
                        }
                    ]
                ),
                _tabla(
                    [
                        {
                            "sku_id": "SKU-1", "candidato": "benchmark", "wape_medio": 0.2, "bias_medio": 0.0,
                            "mae_medio": 3.0, "mase_medio": 0.6, "tasa_fallback_backtest": 0.0,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

        selecciones = seleccionar_mejor_modelo(tabla)

        self.assertIn("peso_ets", selecciones.columns)
        fila_sku0 = selecciones[selecciones["sku_id"] == "SKU-0"].iloc[0]
        fila_sku1 = selecciones[selecciones["sku_id"] == "SKU-1"].iloc[0]
        self.assertEqual(fila_sku0["peso_ets"], 0.3)
        self.assertTrue(pd.isna(fila_sku1["peso_ets"]))


if __name__ == "__main__":
    unittest.main()
