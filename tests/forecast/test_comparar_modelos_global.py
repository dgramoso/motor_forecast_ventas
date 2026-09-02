"""Tests de comparar_modelos_global.py: el backtest de LightGBM global
entrena UN modelo por origen (no uno por SKU) y produce una fila por SKU
compatible con la tabla de `comparar_modelos`, más el fallback a Seasonal
Naive cuando el ajuste global falla en algún origen."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos import _ajustar_benchmark, comparar_modelos_sku
from src.forecast.comparar_modelos_global import backtest_y_predicciones_lightgbm_global
from src.forecast.modelo_ets import _ajustar_ets
from src.forecast.modelo_intermitente import _ajustar_tsb
from src.forecast.seleccionar_modelo import seleccionar_mejor_modelo_sku


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


class TestBacktestLightgbmGlobal(unittest.TestCase):
    def test_una_fila_por_sku_con_candidato_lightgbm_global(self):
        ventas = _ventas_multi_sku(3, 30)
        tabla = backtest_y_predicciones_lightgbm_global(ventas, horizonte=2, ventana_minima=15)[0]

        self.assertEqual(set(tabla["sku_id"]), {"SKU-0", "SKU-1", "SKU-2"})
        self.assertTrue((tabla["candidato"] == "lightgbm_global").all())

    def test_produce_metricas_definidas_con_datos_normales(self):
        ventas = _ventas_multi_sku(3, 30, semilla=1)
        tabla = backtest_y_predicciones_lightgbm_global(ventas, horizonte=2, ventana_minima=15)[0]

        self.assertTrue((tabla["n_ventanas"] > 0).all())
        self.assertFalse(tabla["wape_medio"].isna().any())
        self.assertEqual((tabla["tasa_fallback_backtest"] == 0.0).sum(), len(tabla))

    def test_cae_a_fallback_si_no_alcanza_la_historia_para_los_lags(self):
        # lag=20 exige más historia de la que tiene el primer origen
        # (ventana_minima=15) -> el ajuste global falla ahí y debe caer a
        # Seasonal Naive (que sí alcanza con 15 >= 12 meses), no reventar
        # el backtest.
        ventas = _ventas_multi_sku(2, 25, semilla=2)
        tabla = backtest_y_predicciones_lightgbm_global(ventas, horizonte=2, ventana_minima=15, lags=(1, 2, 3, 20))[0]

        self.assertTrue((tabla["tasa_fallback_backtest"] > 0.0).any())
        self.assertFalse(tabla["wape_medio"].isna().any())

    def test_incluir_sku_id_no_rompe_el_backtest(self):
        ventas = _ventas_multi_sku(3, 30, semilla=3)
        tabla = backtest_y_predicciones_lightgbm_global(ventas, horizonte=2, ventana_minima=15, incluir_sku_id=True)[0]

        self.assertEqual(len(tabla), 3)
        self.assertFalse(tabla["wape_medio"].isna().any())


class TestComparacionCombinada(unittest.TestCase):
    def test_se_puede_seleccionar_el_mejor_candidato_con_lightgbm_incluido(self):
        # Candidatos livianos (sin Prophet/XGBoost/Random Forest) para que
        # el test sea rápido — lo que se valida acá es que la tabla del
        # candidato global se integra con seleccionar_modelo.py sin
        # cambiarlo, no la performance relativa de cada modelo.
        candidatos_livianos = {"benchmark": _ajustar_benchmark, "ets": _ajustar_ets, "tsb": _ajustar_tsb}

        ventas = _ventas_multi_sku(2, 30, semilla=4)
        tablas_por_sku = []
        for sku_id, grupo in ventas.groupby("sku_id"):
            serie = grupo.sort_values("fecha").set_index("fecha")["unidades_vendidas"].asfreq("MS")
            tabla_sku = comparar_modelos_sku(serie, horizonte=2, ventana_minima=15, candidatos=candidatos_livianos)
            tabla_sku.insert(0, "sku_id", sku_id)
            tablas_por_sku.append(tabla_sku)

        tabla_global = backtest_y_predicciones_lightgbm_global(ventas, horizonte=2, ventana_minima=15)[0]
        tabla_combinada = pd.concat(tablas_por_sku + [tabla_global], ignore_index=True)

        for sku_id, tabla_sku in tabla_combinada.groupby("sku_id"):
            seleccion = seleccionar_mejor_modelo_sku(tabla_sku)
            self.assertIn(seleccion["candidato"], {"benchmark", "ets", "tsb", "lightgbm_global"})


if __name__ == "__main__":
    unittest.main()
