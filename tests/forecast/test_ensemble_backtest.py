"""Tests de ensemble_backtest.py: el candidato "ensemble" compite en
seleccionar_modelo.py con pesos ajustados por walk-forward anidado (ver
el docstring del módulo sobre por qué anidado y no un solo ajuste
global)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos_global import backtest_y_predicciones_lightgbm_global
from src.forecast.ensemble_backtest import (
    CANDIDATOS_ENSEMBLE,
    MIN_VENTANAS_AJUSTE_PESOS,
    NOMBRE_ENSEMBLE,
    _backtest_y_predicciones_por_candidato,
    comparar_modelos_con_ensemble,
    evaluar_ensemble,
    evaluar_ensemble_por_sku,
)

from ._helpers import CANDIDATOS_LIVIANOS


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

        pesos = [resultado[f"peso_{nombre}"] for nombre in CANDIDATOS_ENSEMBLE]
        self.assertAlmostEqual(sum(pesos), 1.0, places=6)
        self.assertTrue(all(p >= 0 for p in pesos))

    def test_n_ventanas_evaluadas_es_menor_al_total_por_el_piso(self):
        rng = np.random.default_rng(1)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in CANDIDATOS_ENSEMBLE}

        resultado = evaluar_ensemble_por_sku(reales, pronosticos)

        self.assertEqual(resultado["n_ventanas"], n - MIN_VENTANAS_AJUSTE_PESOS)

    def test_sin_ventanas_previas_suficientes_wape_queda_nan(self):
        n = MIN_VENTANAS_AJUSTE_PESOS  # exactamente el piso: ninguna ventana queda para evaluar
        rng = np.random.default_rng(2)
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in CANDIDATOS_ENSEMBLE}

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

    def test_mase_medio_nan_sin_historicos(self):
        rng = np.random.default_rng(4)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in CANDIDATOS_ENSEMBLE}

        resultado = evaluar_ensemble_por_sku(reales, pronosticos)

        self.assertTrue(np.isnan(resultado["mase_medio"]))

    def test_mase_medio_definido_con_historicos(self):
        rng = np.random.default_rng(5)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in CANDIDATOS_ENSEMBLE}
        historicos = [rng.normal(100, 5, 30) for _ in range(n)]

        resultado = evaluar_ensemble_por_sku(reales, pronosticos, historicos_entrenamiento=historicos)

        self.assertFalse(np.isnan(resultado["mase_medio"]))

    def test_tasa_fallback_es_el_maximo_de_los_componentes(self):
        rng = np.random.default_rng(6)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in CANDIDATOS_ENSEMBLE}
        tasas = {"ets": 0.1, "tsb": 0.7, "lightgbm_global": 0.0}

        resultado = evaluar_ensemble_por_sku(reales, pronosticos, tasas_fallback_componentes=tasas)

        self.assertEqual(resultado["tasa_fallback_backtest"], 0.7)

    def test_tasa_fallback_default_cero_sin_componentes(self):
        rng = np.random.default_rng(7)
        n = 20
        reales = [rng.normal(100, 5, 2) for _ in range(n)]
        pronosticos = {nombre: [r + rng.normal(0, 1, 2) for r in reales] for nombre in CANDIDATOS_ENSEMBLE}

        resultado = evaluar_ensemble_por_sku(reales, pronosticos)

        self.assertEqual(resultado["tasa_fallback_backtest"], 0.0)


class TestBacktestYPrediccionesPorCandidato(unittest.TestCase):
    def test_una_fila_por_sku_y_predicciones_alineadas(self):
        ventas = _ventas_multi_sku(2, 40, semilla=8)
        tabla, predicciones = _backtest_y_predicciones_por_candidato(ventas, "ets", horizonte=2, ventana_minima=20)

        self.assertEqual(set(tabla["sku_id"]), {"SKU-0", "SKU-1"})
        self.assertTrue((tabla["candidato"] == "ets").all())
        for sku_id in ("SKU-0", "SKU-1"):
            reales, pronosticos = predicciones[sku_id]
            self.assertEqual(len(reales), len(pronosticos))
            self.assertEqual(len(reales), int(tabla.loc[tabla["sku_id"] == sku_id, "n_ventanas"].iloc[0]))


class TestEvaluarEnsemble(unittest.TestCase):
    def _predicciones_y_tasas(self, ventas, horizonte, ventana_minima):
        tabla_ets, predicciones_ets = _backtest_y_predicciones_por_candidato(ventas, "ets", horizonte, ventana_minima)
        tabla_tsb, predicciones_tsb = _backtest_y_predicciones_por_candidato(ventas, "tsb", horizonte, ventana_minima)
        tabla_lgbm, predicciones_lgbm = backtest_y_predicciones_lightgbm_global(ventas, horizonte, ventana_minima)

        predicciones_por_candidato = {"ets": predicciones_ets, "tsb": predicciones_tsb, "lightgbm_global": predicciones_lgbm}
        tasas_por_candidato = {
            "ets": tabla_ets.set_index("sku_id")["tasa_fallback_backtest"].to_dict(),
            "tsb": tabla_tsb.set_index("sku_id")["tasa_fallback_backtest"].to_dict(),
            "lightgbm_global": tabla_lgbm.set_index("sku_id")["tasa_fallback_backtest"].to_dict(),
        }
        return predicciones_por_candidato, tasas_por_candidato

    def test_una_fila_por_sku_mismo_esquema_que_comparar_modelos(self):
        ventas = _ventas_multi_sku(3, 40, semilla=4)
        predicciones, tasas = self._predicciones_y_tasas(ventas, horizonte=2, ventana_minima=20)

        evaluacion = evaluar_ensemble(ventas, predicciones, tasas, horizonte=2, ventana_minima=20)

        self.assertEqual(set(evaluacion["sku_id"]), {"SKU-0", "SKU-1", "SKU-2"})
        self.assertTrue((evaluacion["candidato"] == NOMBRE_ENSEMBLE).all())
        for columna in ("n_ventanas", "wape_indefinido", "wape_medio", "bias_medio", "mae_medio", "mase_medio", "tasa_fallback_backtest"):
            self.assertIn(columna, evaluacion.columns)
        for nombre in CANDIDATOS_ENSEMBLE:
            self.assertIn(f"peso_{nombre}", evaluacion.columns)

    def test_wape_medio_no_es_nan_con_ventanas_alineadas(self):
        # En el dataset sintético todas las SKUs comparten calendario, así
        # que ETS/TSB (walk-forward posicional) y LightGBM global
        # (walk-forward por calendario) deben coincidir ventana a ventana
        # y el ensemble debe poder evaluarse — no quedar en NaN por una
        # falla espuria de la verificación de alineación.
        ventas = _ventas_multi_sku(2, 40, semilla=9)
        predicciones, tasas = self._predicciones_y_tasas(ventas, horizonte=2, ventana_minima=20)

        evaluacion = evaluar_ensemble(ventas, predicciones, tasas, horizonte=2, ventana_minima=20)

        self.assertFalse(evaluacion["wape_medio"].isna().any())

    def test_sin_datos_suficientes_si_las_ventanas_no_coinciden(self):
        ventas = _ventas_multi_sku(1, 40, semilla=10)
        predicciones, tasas = self._predicciones_y_tasas(ventas, horizonte=2, ventana_minima=20)
        # Desalinea a mano: LightGBM "ve" una ventana de más para el SKU.
        reales_lgbm, pronosticos_lgbm = predicciones["lightgbm_global"]["SKU-0"]
        predicciones["lightgbm_global"]["SKU-0"] = (reales_lgbm[:-1], pronosticos_lgbm[:-1])

        evaluacion = evaluar_ensemble(ventas, predicciones, tasas, horizonte=2, ventana_minima=20)

        fila = evaluacion[evaluacion["sku_id"] == "SKU-0"].iloc[0]
        self.assertTrue(np.isnan(fila["wape_medio"]))
        self.assertTrue(np.isnan(fila["peso_ets"]))


class TestCompararModelosConEnsemble(unittest.TestCase):
    def test_agrega_el_candidato_ensemble_a_la_tabla_base(self):
        ventas = _ventas_multi_sku(2, 40, semilla=5)
        tabla = comparar_modelos_con_ensemble(ventas, horizonte=2, ventana_minima=20, candidatos=CANDIDATOS_LIVIANOS)

        for sku_id in ("SKU-0", "SKU-1"):
            candidatos_sku = set(tabla.loc[tabla["sku_id"] == sku_id, "candidato"])
            self.assertIn(NOMBRE_ENSEMBLE, candidatos_sku)
            self.assertIn("lightgbm_global", candidatos_sku)
            self.assertIn("benchmark", candidatos_sku)
            self.assertIn("ets", candidatos_sku)
            self.assertIn("tsb", candidatos_sku)

    def test_filas_de_otros_candidatos_no_tienen_pesos(self):
        ventas = _ventas_multi_sku(1, 40, semilla=6)
        tabla = comparar_modelos_con_ensemble(ventas, horizonte=2, ventana_minima=20, candidatos=CANDIDATOS_LIVIANOS)

        fila_benchmark = tabla[tabla["candidato"] == "benchmark"].iloc[0]
        self.assertTrue(np.isnan(fila_benchmark["peso_ets"]))

    def test_ets_y_tsb_no_aparecen_duplicados(self):
        # Antes del fix, evaluar_ensemble corría su propio backtest de
        # ETS/TSB además del que ya corre comparar_modelos_con_ensemble
        # para la fila del candidato — dos backtest_walk_forward por
        # SKU y por candidato en vez de uno. Ahora hay una sola fila
        # "ets" y una sola "tsb" por SKU.
        ventas = _ventas_multi_sku(1, 40, semilla=11)
        tabla = comparar_modelos_con_ensemble(ventas, horizonte=2, ventana_minima=20, candidatos=CANDIDATOS_LIVIANOS)

        self.assertEqual((tabla["candidato"] == "ets").sum(), 1)
        self.assertEqual((tabla["candidato"] == "tsb").sum(), 1)

    def test_sin_ets_o_tsb_en_candidatos_no_hay_ensemble(self):
        # El seam de inyección de candidatos también decide si el
        # ensemble participa: sin ets/tsb no hay con qué combinarlo.
        ventas = _ventas_multi_sku(1, 40, semilla=12)
        candidatos_sin_ets = {k: v for k, v in CANDIDATOS_LIVIANOS.items() if k != "ets"}
        tabla = comparar_modelos_con_ensemble(ventas, horizonte=2, ventana_minima=20, candidatos=candidatos_sin_ets)

        self.assertNotIn(NOMBRE_ENSEMBLE, set(tabla["candidato"]))
        self.assertIn("lightgbm_global", set(tabla["candidato"]))


if __name__ == "__main__":
    unittest.main()
