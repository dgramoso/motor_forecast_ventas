"""Tests de ensemble.py: combinación por pesos fijos y ajuste de pesos
óptimos sobre predicciones out-of-sample ya calculadas."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.backtest import backtest_walk_forward
from src.forecast.comparar_modelos_global import recolectar_predicciones_lightgbm_global
from src.forecast.ensemble import PESOS_A, PESOS_B, PESOS_C, combinar_pronosticos, optimizar_pesos
from src.forecast.modelo_ets import pronosticar_ets
from src.forecast.modelo_intermitente import pronosticar_tsb


class TestCombinarPronosticos(unittest.TestCase):
    def test_combinacion_lineal_simple(self):
        pronosticos = {"ets": np.array([10.0, 20.0]), "tsb": np.array([30.0, 40.0])}
        pesos = {"ets": 0.5, "tsb": 0.5}

        combinado = combinar_pronosticos(pronosticos, pesos)

        np.testing.assert_allclose(combinado, [20.0, 30.0])

    def test_pesos_a_b_c_son_validos(self):
        pronosticos = {"ets": np.array([10.0]), "tsb": np.array([10.0]), "lightgbm_global": np.array([10.0])}
        for pesos in (PESOS_A, PESOS_B, PESOS_C):
            combinado = combinar_pronosticos(pronosticos, pesos)
            self.assertAlmostEqual(combinado[0], 10.0, places=3)

    def test_falla_si_los_modelos_no_coinciden(self):
        pronosticos = {"ets": np.array([10.0])}
        pesos = {"ets": 0.5, "tsb": 0.5}
        with self.assertRaises(ValueError):
            combinar_pronosticos(pronosticos, pesos)

    def test_falla_si_los_pesos_no_suman_uno(self):
        pronosticos = {"ets": np.array([10.0]), "tsb": np.array([10.0])}
        pesos = {"ets": 0.5, "tsb": 0.6}
        with self.assertRaises(ValueError):
            combinar_pronosticos(pronosticos, pesos)

    def test_falla_si_algun_peso_es_negativo(self):
        pronosticos = {"ets": np.array([10.0]), "tsb": np.array([10.0])}
        pesos = {"ets": 1.5, "tsb": -0.5}
        with self.assertRaises(ValueError):
            combinar_pronosticos(pronosticos, pesos)

    def test_nunca_negativo(self):
        pronosticos = {"ets": np.array([-100.0]), "tsb": np.array([10.0])}
        pesos = {"ets": 0.9, "tsb": 0.1}

        combinado = combinar_pronosticos(pronosticos, pesos)

        self.assertTrue((combinado >= 0).all())


class TestOptimizarPesos(unittest.TestCase):
    def test_pesos_suman_uno_y_son_no_negativos(self):
        reales = [np.array([10.0, 20.0]), np.array([15.0, 25.0])]
        pronosticos = {
            "ets": [np.array([9.0, 19.0]), np.array([16.0, 24.0])],
            "tsb": [np.array([50.0, 5.0]), np.array([1.0, 60.0])],
        }

        pesos = optimizar_pesos(reales, pronosticos)

        self.assertAlmostEqual(sum(pesos.values()), 1.0, places=6)
        self.assertTrue(all(peso >= 0 for peso in pesos.values()))

    def test_favorece_al_modelo_que_predice_casi_perfecto(self):
        rng = np.random.default_rng(0)
        reales = [rng.normal(100, 5, 3) for _ in range(15)]
        pronosticos = {
            "bueno": [real + rng.normal(0, 0.5, 3) for real in reales],
            "malo": [rng.normal(500, 100, 3) for _ in reales],
        }

        pesos = optimizar_pesos(reales, pronosticos)

        self.assertGreater(pesos["bueno"], pesos["malo"])
        self.assertGreater(pesos["bueno"], 0.8)

    def test_falla_si_las_longitudes_no_coinciden(self):
        reales = [np.array([10.0])]
        pronosticos = {"ets": [np.array([10.0]), np.array([11.0])]}

        with self.assertRaises(ValueError):
            optimizar_pesos(reales, pronosticos)

    def test_falla_si_no_hay_ventanas(self):
        with self.assertRaises(ValueError):
            optimizar_pesos([], {"ets": []})


def _ventas_multi_sku(n_skus: int, n_meses: int, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2020-01-01", periods=n_meses, freq="MS")
    filas = []
    for i in range(n_skus):
        nivel = 50 + i * 10
        valores = np.maximum(nivel + rng.normal(0, 5, n_meses), 0)
        for fecha, valor in zip(fechas, valores):
            filas.append({"sku_id": f"SKU-{i}", "fecha": fecha, "unidades_vendidas": valor})
    return pd.DataFrame(filas)


class TestFlujoCompletoEtsTsbLightgbm(unittest.TestCase):
    """Ejercita el ejemplo concreto de la sección 15 del pedido: ajustar
    pesos de ensemble para ETS + TSB + LightGBM global a partir de las
    predicciones out-of-sample que YA calculó el backtest — sin
    reentrenar nada de nuevo acá."""

    def test_ajusta_pesos_con_predicciones_reales_del_backtest(self):
        horizonte, ventana_minima = 2, 15
        ventas = _ventas_multi_sku(3, 30, semilla=7)
        sku_id = "SKU-0"
        serie = ventas[ventas["sku_id"] == sku_id].set_index("fecha")["unidades_vendidas"].asfreq("MS")

        backtest_ets = backtest_walk_forward(serie, pronosticar_ets, horizonte, ventana_minima)
        backtest_tsb = backtest_walk_forward(serie, pronosticar_tsb, horizonte, ventana_minima)
        predicciones_lgbm = recolectar_predicciones_lightgbm_global(ventas, horizonte, ventana_minima)
        reales_lgbm, pronosticos_lgbm = predicciones_lgbm[sku_id]

        # Mismo número de orígenes en los tres — mismo calendario
        # compartido, mismo horizonte/ventana_minima.
        self.assertEqual(len(backtest_ets), len(reales_lgbm))
        self.assertEqual(len(backtest_tsb), len(reales_lgbm))

        pronosticos_por_modelo = {
            "ets": list(backtest_ets["pronostico"]),
            "tsb": list(backtest_tsb["pronostico"]),
            "lightgbm_global": pronosticos_lgbm,
        }
        pesos = optimizar_pesos(reales_lgbm, pronosticos_por_modelo)

        self.assertEqual(set(pesos), {"ets", "tsb", "lightgbm_global"})
        self.assertAlmostEqual(sum(pesos.values()), 1.0, places=6)

        combinado = combinar_pronosticos(
            {
                "ets": backtest_ets["pronostico"].iloc[-1],
                "tsb": backtest_tsb["pronostico"].iloc[-1],
                "lightgbm_global": pronosticos_lgbm[-1],
            },
            pesos,
        )
        self.assertEqual(len(combinado), horizonte)
        self.assertTrue((combinado >= 0).all())


if __name__ == "__main__":
    unittest.main()
