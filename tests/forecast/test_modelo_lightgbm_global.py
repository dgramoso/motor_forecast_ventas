"""Tests de modelo_lightgbm_global.py: un solo modelo entrenado con todas
las SKUs (no uno por SKU), predicción multi-step y la comparación
Modelo A (sin identidad de SKU) / Modelo B (con `sku_id` categórico)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.features_lightgbm import construir_dataset_supervisado
from src.forecast.modelo_lightgbm_global import entrenar_lightgbm_global, pronosticar_lightgbm_global


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


class TestEntrenarYPronosticar(unittest.TestCase):
    def test_un_modelo_por_paso_de_horizonte(self):
        dataset = construir_dataset_supervisado(
            _ventas_multi_sku(4, 30), horizonte=3, lags=(1, 2), ventanas_rolling=(3,)
        )
        modelos = entrenar_lightgbm_global(dataset, horizonte=3)

        self.assertEqual(set(modelos.keys()), {1, 2, 3})

    def test_una_fila_de_pronostico_por_sku_y_paso(self):
        dataset = construir_dataset_supervisado(
            _ventas_multi_sku(3, 24), horizonte=2, lags=(1, 2), ventanas_rolling=(3,)
        )
        modelos = entrenar_lightgbm_global(dataset, horizonte=2)
        pronostico = pronosticar_lightgbm_global(modelos, dataset)

        self.assertEqual(len(pronostico), 3 * 2)
        self.assertEqual(set(pronostico["sku_id"]), {"SKU-0", "SKU-1", "SKU-2"})

    def test_sin_negativos(self):
        dataset = construir_dataset_supervisado(
            _ventas_multi_sku(3, 24, semilla=1), horizonte=2, lags=(1, 2), ventanas_rolling=(3,)
        )
        modelos = entrenar_lightgbm_global(dataset, horizonte=2)
        pronostico = pronosticar_lightgbm_global(modelos, dataset)

        self.assertTrue((pronostico["unidades_pronosticadas"] >= 0).all())

    def test_sin_nan_ni_inf(self):
        dataset = construir_dataset_supervisado(
            _ventas_multi_sku(3, 24, semilla=2), horizonte=2, lags=(1, 2), ventanas_rolling=(3,)
        )
        modelos = entrenar_lightgbm_global(dataset, horizonte=2)
        pronostico = pronosticar_lightgbm_global(modelos, dataset)

        self.assertFalse(pronostico["unidades_pronosticadas"].isna().any())
        self.assertTrue(np.isfinite(pronostico["unidades_pronosticadas"]).all())

    def test_entrenamiento_excluye_filas_sin_target(self):
        # Con 1 solo SKU y horizonte=1, la última fila del dataset tiene
        # target NaN (ver test_features_lightgbm) — no debería reventar
        # el fit ni contaminar el modelo con NaN como target.
        dataset = construir_dataset_supervisado(
            _ventas_multi_sku(1, 24, semilla=3), horizonte=1, lags=(1, 2), ventanas_rolling=(3,)
        )
        modelos = entrenar_lightgbm_global(dataset, horizonte=1)

        self.assertEqual(len(modelos), 1)


class TestIdentidadSKU(unittest.TestCase):
    def test_modelo_b_con_sku_id_categorico_entrena_y_predice(self):
        dataset = construir_dataset_supervisado(
            _ventas_multi_sku(4, 24, semilla=4), horizonte=1, lags=(1, 2), ventanas_rolling=(3,)
        )
        modelos = entrenar_lightgbm_global(dataset, horizonte=1, incluir_sku_id=True)
        pronostico = pronosticar_lightgbm_global(modelos, dataset, incluir_sku_id=True)

        self.assertEqual(len(pronostico), 4)
        self.assertTrue((pronostico["unidades_pronosticadas"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
