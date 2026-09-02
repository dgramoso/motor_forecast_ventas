"""Tests de ensemble_informativo.py: evaluación y pronóstico futuro del
ensemble ETS+TSB+LightGBM global como vista comparativa, sin tocar
`seleccionar_modelo.py` (ver el docstring del módulo sobre por qué no
compite)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.ensemble_informativo import (
    MODELOS_ENSEMBLE,
    NOMBRE_ENSEMBLE,
    evaluar_ensemble_informativo,
    pronosticar_futuro_ensemble,
)


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


class TestEvaluarEnsembleInformativo(unittest.TestCase):
    def test_una_fila_por_sku_con_pesos_validos(self):
        ventas = _ventas_multi_sku(3, 30, semilla=1)
        evaluacion = evaluar_ensemble_informativo(ventas, horizonte=2, ventana_minima=24)

        self.assertEqual(set(evaluacion["sku_id"]), {"SKU-0", "SKU-1", "SKU-2"})
        self.assertTrue((evaluacion["candidato"] == NOMBRE_ENSEMBLE).all())
        for fila in evaluacion.itertuples():
            self.assertFalse(np.isnan(fila.wape_medio))
            for nombre in MODELOS_ENSEMBLE:
                peso = getattr(fila, f"peso_{nombre}")
                self.assertGreaterEqual(peso, 0.0)
        pesos_totales = evaluacion[[f"peso_{n}" for n in MODELOS_ENSEMBLE]].sum(axis=1)
        np.testing.assert_allclose(pesos_totales, 1.0, atol=1e-6)

    def test_sin_datos_suficientes_queda_en_nan(self):
        # 30 meses con ventana_minima=29 y horizonte=2 no deja ni una sola
        # ventana de backtest (último origen = 30 - 2 = 28 < 29).
        ventas = _ventas_multi_sku(1, 30, semilla=2)
        evaluacion = evaluar_ensemble_informativo(ventas, horizonte=2, ventana_minima=29)

        self.assertEqual(len(evaluacion), 1)
        self.assertEqual(evaluacion.iloc[0]["n_ventanas"], 0)
        self.assertTrue(np.isnan(evaluacion.iloc[0]["wape_medio"]))
        for nombre in MODELOS_ENSEMBLE:
            self.assertTrue(np.isnan(evaluacion.iloc[0][f"peso_{nombre}"]))


class TestPronosticarFuturoEnsemble(unittest.TestCase):
    def test_combina_los_tres_modelos_para_skus_con_pesos_validos(self):
        ventas = _ventas_multi_sku(2, 30, semilla=3)
        evaluacion = evaluar_ensemble_informativo(ventas, horizonte=2, ventana_minima=24)

        pronostico = pronosticar_futuro_ensemble(ventas, evaluacion, horizonte=2)

        self.assertEqual(set(pronostico["sku_id"]), {"SKU-0", "SKU-1"})
        self.assertTrue((pronostico["candidato"] == NOMBRE_ENSEMBLE).all())
        self.assertEqual(len(pronostico), 2 * 2)
        self.assertTrue((pronostico["unidades_pronosticadas"] >= 0).all())

    def test_excluye_skus_sin_datos_suficientes(self):
        ventas = _ventas_multi_sku(1, 30, semilla=4)
        evaluacion = evaluar_ensemble_informativo(ventas, horizonte=2, ventana_minima=29)

        pronostico = pronosticar_futuro_ensemble(ventas, evaluacion, horizonte=2)

        self.assertTrue(pronostico.empty)


if __name__ == "__main__":
    unittest.main()
