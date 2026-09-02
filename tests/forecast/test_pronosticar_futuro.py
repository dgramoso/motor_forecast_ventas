"""Tests de pronosticar_futuro.py: servir el pronóstico final, incluido
el candidato "lightgbm_global" (Fase 6) — se entrena UNA vez para todas
las SKUs que lo ganaron, no una por una, y cae a Seasonal Naive si el
ajuste global falla."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos import _ajustar_benchmark
from src.forecast.comparar_modelos_global import NOMBRE_CANDIDATO
from src.forecast.ensemble_backtest import NOMBRE_ENSEMBLE, comparar_modelos_con_ensemble
from src.forecast.modelo_ets import _ajustar_ets
from src.forecast.modelo_intermitente import _ajustar_tsb
from src.forecast.pronosticar_futuro import pronosticar_futuro, pronosticar_futuro_ensemble, pronosticar_futuro_lightgbm_global

# Candidatos livianos (sin Prophet/XGBoost/Random Forest) para que los
# tests de comparar_modelos_con_ensemble sean rápidos.
_CANDIDATOS_LIVIANOS = {"benchmark": _ajustar_benchmark, "ets": _ajustar_ets, "tsb": _ajustar_tsb}


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


class TestPronosticarFuturoLightgbmGlobal(unittest.TestCase):
    def test_una_fila_por_sku_y_paso_de_horizonte(self):
        ventas = _ventas_multi_sku(3, 30)
        pronostico = pronosticar_futuro_lightgbm_global(ventas, skus=["SKU-0", "SKU-2"], horizonte=2)

        self.assertEqual(len(pronostico), 2 * 2)
        self.assertEqual(set(pronostico["sku_id"]), {"SKU-0", "SKU-2"})
        self.assertTrue((pronostico["candidato"] == NOMBRE_CANDIDATO).all())

    def test_no_reentrena_una_vez_por_sku_sino_una_vez_en_total(self):
        # No hay forma directa de "contar" entrenamientos desde afuera sin
        # mockear — lo que sí se puede verificar es que sirve más de una
        # SKU sin reventar y en tiempo acotado (regresión de performance
        # burda: si volviera a un modelo por SKU, esto seguiría pasando
        # igual con pocas SKUs, así que el valor real de este test es
        # documentar la intención, no medir tiempo).
        ventas = _ventas_multi_sku(5, 30, semilla=1)
        pronostico = pronosticar_futuro_lightgbm_global(ventas, skus=[f"SKU-{i}" for i in range(5)], horizonte=1)

        self.assertEqual(len(pronostico), 5)
        self.assertFalse(pronostico["fallback"].any())

    def test_incluye_diagnostico_de_demanda(self):
        ventas = _ventas_multi_sku(2, 30, semilla=2)
        pronostico = pronosticar_futuro_lightgbm_global(ventas, skus=["SKU-0"], horizonte=1)

        for columna in ("tasa_de_ceros", "adi", "cv2", "clase_demanda", "observaciones_entrenamiento"):
            self.assertIn(columna, pronostico.columns)

    def test_sin_negativos(self):
        ventas = _ventas_multi_sku(3, 30, semilla=3)
        pronostico = pronosticar_futuro_lightgbm_global(ventas, skus=["SKU-0", "SKU-1"], horizonte=2)

        self.assertTrue((pronostico["unidades_pronosticadas"] >= 0).all())

    def test_cae_a_fallback_si_el_ajuste_global_falla(self):
        # lag=20 exige más historia de la que hay (15 meses) -> el ajuste
        # global falla para todas las SKUs, pero Seasonal Naive sí
        # alcanza (15 >= 12 períodos) -> debe caer a ese fallback, no
        # reventar.
        ventas = _ventas_multi_sku(2, 15, semilla=4)
        pronostico = pronosticar_futuro_lightgbm_global(
            ventas, skus=["SKU-0", "SKU-1"], horizonte=1, lags=(1, 2, 3, 20)
        )

        self.assertTrue(pronostico["fallback"].all())


class TestPronosticarFuturoIntegrado(unittest.TestCase):
    def test_mezcla_candidatos_por_sku_y_lightgbm_global(self):
        ventas = _ventas_multi_sku(2, 30, semilla=5)
        tabla_comparativa = pd.DataFrame(
            [
                {
                    "sku_id": "SKU-0",
                    "candidato": "benchmark",
                    "wape_medio": 0.1,
                    "bias_medio": 0.0,
                    "mae_medio": 1.0,
                    "mase_medio": 0.9,
                    "tasa_fallback_backtest": 0.0,
                },
                {
                    "sku_id": "SKU-1",
                    "candidato": NOMBRE_CANDIDATO,
                    "wape_medio": 0.05,
                    "bias_medio": 0.0,
                    "mae_medio": 0.5,
                    "mase_medio": 0.5,
                    "tasa_fallback_backtest": 0.0,
                },
            ]
        )

        pronostico = pronosticar_futuro(ventas, tabla_comparativa, horizonte=2)

        self.assertEqual(set(pronostico["sku_id"]), {"SKU-0", "SKU-1"})
        candidato_sku0 = pronostico.loc[pronostico["sku_id"] == "SKU-0", "candidato"].unique()
        candidato_sku1 = pronostico.loc[pronostico["sku_id"] == "SKU-1", "candidato"].unique()
        self.assertEqual(list(candidato_sku0), ["benchmark"])
        self.assertEqual(list(candidato_sku1), [NOMBRE_CANDIDATO])


class TestPronosticarFuturoEnsemble(unittest.TestCase):
    def test_sirve_ensemble_cuando_gana_la_seleccion(self):
        ventas = _ventas_multi_sku(2, 40, semilla=6)
        tabla_comparativa = comparar_modelos_con_ensemble(
            ventas, horizonte=2, ventana_minima=20, candidatos=_CANDIDATOS_LIVIANOS
        )

        # Fuerza a "ensemble" a ganar en SKU-0 sin depender del backtest real.
        tabla_comparativa.loc[
            (tabla_comparativa["sku_id"] == "SKU-0") & (tabla_comparativa["candidato"] != NOMBRE_ENSEMBLE),
            "wape_medio",
        ] = 999.0
        tabla_comparativa.loc[
            (tabla_comparativa["sku_id"] == "SKU-0") & (tabla_comparativa["candidato"] == NOMBRE_ENSEMBLE),
            "wape_medio",
        ] = 0.001

        pronostico = pronosticar_futuro(ventas, tabla_comparativa, horizonte=2)

        candidato_sku0 = pronostico.loc[pronostico["sku_id"] == "SKU-0", "candidato"].unique()
        self.assertEqual(list(candidato_sku0), [NOMBRE_ENSEMBLE])
        self.assertTrue((pronostico.loc[pronostico["sku_id"] == "SKU-0", "unidades_pronosticadas"] >= 0).all())

    def test_combina_los_tres_modelos_con_los_pesos_de_produccion(self):
        ventas = _ventas_multi_sku(1, 40, semilla=7)
        tabla_comparativa = comparar_modelos_con_ensemble(
            ventas, horizonte=2, ventana_minima=20, candidatos=_CANDIDATOS_LIVIANOS
        )

        pronostico = pronosticar_futuro_ensemble(ventas, ["SKU-0"], tabla_comparativa, horizonte=2)

        self.assertEqual(len(pronostico), 2)
        self.assertTrue((pronostico["candidato"] == NOMBRE_ENSEMBLE).all())
        self.assertFalse(pronostico["fallback"].any())


if __name__ == "__main__":
    unittest.main()
