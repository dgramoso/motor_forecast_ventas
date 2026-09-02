"""Tests de ensemble_backtest.py: el candidato "ensemble" compite en
seleccionar_modelo.py con pesos ajustados por walk-forward anidado (ver
el docstring del módulo sobre por qué anidado y no un solo ajuste
global)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos import _ajustar_benchmark
from src.forecast.ensemble_backtest import (
    MIN_VENTANAS_AJUSTE_PESOS,
    MODELOS_ENSEMBLE,
    NOMBRE_ENSEMBLE,
    comparar_modelos_con_ensemble,
    evaluar_ensemble,
    evaluar_ensemble_por_sku,
)
from src.forecast.modelo_ets import _ajustar_ets
from src.forecast.modelo_intermitente import _ajustar_tsb

# Candidatos livianos (sin Prophet/XGBoost/Random Forest) para que los
# tests de comparar_modelos_con_ensemble sean rápidos — ver el mismo
# criterio en test_comparar_modelos_global.py::TestComparacionCombinada.
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


class TestEvaluarEnsemblePorSku(unittest.TestCase):
    def test_pesos_de_produccion_suman_uno(self):
        rng = np.random.default_rng(0)
        n = 15
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {
            "ets": [r + rng.normal(0, 1, 2) for r in reales],
            "tsb": [rng.normal(500, 100, 2) for _ in range(n)],
            "lightgbm_global": [r + rng.normal(0, 2, 2) for r in reales],
        }

        resultado = evaluar_ensemble_por_sku(reales, pronosticos)

        pesos = [resultado[f"peso_{nombre}"] for nombre in MODELOS_ENSEMBLE]
        self.assertAlmostEqual(sum(pesos), 1.0, places=6)
        self.assertTrue(all(p >= 0 for p in pesos))

    def test_n_ventanas_evaluadas_es_menor_al_total_por_el_piso(self):
        rng = np.random.default_rng(1)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in MODELOS_ENSEMBLE}

        resultado = evaluar_ensemble_por_sku(reales, pronosticos)

        self.assertEqual(resultado["n_ventanas"], n - MIN_VENTANAS_AJUSTE_PESOS)

    def test_sin_ventanas_previas_suficientes_wape_queda_nan(self):
        n = MIN_VENTANAS_AJUSTE_PESOS  # exactamente el piso: ninguna ventana queda para evaluar
        rng = np.random.default_rng(2)
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in MODELOS_ENSEMBLE}

        resultado = evaluar_ensemble_por_sku(reales, pronosticos)

        self.assertEqual(resultado["n_ventanas"], 0)
        self.assertTrue(np.isnan(resultado["wape_medio"]))
        # los pesos de PRODUCCIÓN sí se calculan con lo disponible, aunque
        # no haya ventanas para evaluar honestamente el WAPE
        self.assertFalse(np.isnan(resultado["peso_ets"]))

    def test_wape_no_se_beneficia_de_ver_su_propia_ventana(self):
        # Un modelo "tramposo" que predice perfecto sólo en la ventana de
        # evaluación (i par) y muy mal en las de ajuste (i impar) no puede
        # ganar peso en el walk-forward anidado, porque optimizar_pesos
        # nunca ve la ventana de evaluación al ajustar. Si el código
        # ajustara los pesos con la misma ventana que evalúa (leakage),
        # el "tramposo" ganaría peso 1.0 y el WAPE combinado sería ~0.
        rng = np.random.default_rng(3)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        tramposo = []
        malo = []
        for i, real in enumerate(reales):
            if i % 2 == 0:
                tramposo.append(real.copy())  # perfecto sólo en ventanas "de evaluación"
                malo.append(rng.normal(500, 50, 2))
            else:
                tramposo.append(rng.normal(500, 50, 2))  # malo en ventanas "de ajuste"
                malo.append(real.copy())

        resultado = evaluar_ensemble_por_sku(reales, {"a": tramposo, "b": malo, "c": malo})

        self.assertGreater(resultado["wape_medio"], 0.5)


class TestEvaluarEnsemble(unittest.TestCase):
    def test_una_fila_por_sku_mismo_esquema_que_comparar_modelos(self):
        ventas = _ventas_multi_sku(3, 40, semilla=4)
        evaluacion = evaluar_ensemble(ventas, horizonte=2, ventana_minima=20)

        self.assertEqual(set(evaluacion["sku_id"]), {"SKU-0", "SKU-1", "SKU-2"})
        self.assertTrue((evaluacion["candidato"] == NOMBRE_ENSEMBLE).all())
        for columna in ("n_ventanas", "wape_indefinido", "wape_medio", "bias_medio", "mae_medio", "mase_medio", "tasa_fallback_backtest"):
            self.assertIn(columna, evaluacion.columns)
        for nombre in MODELOS_ENSEMBLE:
            self.assertIn(f"peso_{nombre}", evaluacion.columns)


class TestCompararModelosConEnsemble(unittest.TestCase):
    def test_agrega_el_candidato_ensemble_a_la_tabla_base(self):
        ventas = _ventas_multi_sku(2, 40, semilla=5)
        tabla = comparar_modelos_con_ensemble(ventas, horizonte=2, ventana_minima=20, candidatos=_CANDIDATOS_LIVIANOS)

        for sku_id in ("SKU-0", "SKU-1"):
            candidatos_sku = set(tabla.loc[tabla["sku_id"] == sku_id, "candidato"])
            self.assertIn(NOMBRE_ENSEMBLE, candidatos_sku)
            self.assertIn("lightgbm_global", candidatos_sku)
            self.assertIn("benchmark", candidatos_sku)

    def test_filas_de_otros_candidatos_no_tienen_pesos(self):
        ventas = _ventas_multi_sku(1, 40, semilla=6)
        tabla = comparar_modelos_con_ensemble(ventas, horizonte=2, ventana_minima=20, candidatos=_CANDIDATOS_LIVIANOS)

        fila_benchmark = tabla[tabla["candidato"] == "benchmark"].iloc[0]
        self.assertTrue(np.isnan(fila_benchmark["peso_ets"]))


if __name__ == "__main__":
    unittest.main()
