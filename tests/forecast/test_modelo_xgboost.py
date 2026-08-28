"""Tests de XGBoost (src/forecast/modelo_xgboost.py): hiperparámetros
fijos, y fallback auditable a Seasonal Naive cuando no hay suficiente
historial para las features de árboles — mismo contrato que
modelo_ets.py. No depende de que xgboost esté instalado: mockea
`_crear_estimador`, igual que test_modelo_arboles.py hace con un doble
propio."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast.modelo_xgboost import _ajustar_xgboost, _crear_estimador, pronosticar_xgboost


class _EstimadorConstante:
    def fit(self, x, y):
        self._prediccion = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self._prediccion)


def _serie(n: int) -> pd.Series:
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")
    return pd.Series(np.arange(n, dtype=float), index=fechas)


class TestHiperparametrosFijos(unittest.TestCase):
    def test_sin_autotuning_valores_fijos(self):
        self.assertEqual(
            _crear_estimador.keywords,
            {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1, "random_state": 42},
        )


class TestAjusteExitoso(unittest.TestCase):
    @patch("src.forecast.modelo_xgboost._crear_estimador", new=_EstimadorConstante)
    def test_no_hace_fallback_con_historial_suficiente(self):
        forecast, fallback, motivo = _ajustar_xgboost(_serie(30), horizonte=3)

        self.assertFalse(fallback)
        self.assertIsNone(motivo)
        self.assertEqual(len(forecast), 3)


class TestFallbackAuditable(unittest.TestCase):
    def test_fallback_registra_motivo_con_historial_corto(self):
        forecast, fallback, motivo = _ajustar_xgboost(_serie(13), horizonte=3)

        self.assertTrue(fallback)
        self.assertIn("ValueError", motivo)
        self.assertEqual(len(forecast), 3)

    def test_pronosticar_xgboost_sigue_devolviendo_solo_array(self):
        resultado = pronosticar_xgboost(_serie(13), horizonte=3)

        self.assertIsInstance(resultado, np.ndarray)
        self.assertEqual(len(resultado), 3)


if __name__ == "__main__":
    unittest.main()
