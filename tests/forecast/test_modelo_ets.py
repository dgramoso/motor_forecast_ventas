"""Tests de ETS (src/forecast/modelo_ets.py): variantes de estacionalidad
según la longitud de la serie, y trazabilidad del fallback a Seasonal
Naive cuando el ajuste de Holt-Winters falla.
"""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast.modelo_ets import _ajustar_ets, pronosticar_ets


def _serie(n: int, nivel: float = 100.0, amplitud: float = 10.0, ruido: float = 1.0, semilla: int = 1) -> pd.Series:
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")
    mes = fechas.month.to_numpy() - 1
    rng = np.random.default_rng(semilla)
    valores = nivel + amplitud * np.sin(2 * np.pi * mes / 12) + rng.normal(0, ruido, n)
    return pd.Series(valores, index=fechas)


class TestVariantesEstacionalidad(unittest.TestCase):
    """TEST 6 y TEST 7: seasonal="add" solo a partir de 2*PERIODO_ESTACIONAL (24) obs."""

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_24_observaciones_activa_estacionalidad(self, mock_es):
        mock_es.return_value.fit.return_value.forecast.return_value = pd.Series([1.0, 2.0, 3.0])
        serie = _serie(n=24)

        pronosticar_ets(serie, horizonte=3)

        _args, kwargs = mock_es.call_args
        self.assertEqual(kwargs["seasonal"], "add")
        self.assertEqual(kwargs["seasonal_periods"], 12)

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_23_observaciones_no_activa_estacionalidad(self, mock_es):
        mock_es.return_value.fit.return_value.forecast.return_value = pd.Series([1.0, 2.0, 3.0])
        serie = _serie(n=23)

        pronosticar_ets(serie, horizonte=3)

        _args, kwargs = mock_es.call_args
        self.assertIsNone(kwargs["seasonal"])
        self.assertIsNone(kwargs["seasonal_periods"])


class TestFallbackAuditable(unittest.TestCase):
    """TEST 8: si ETS falla, el fallback a Seasonal Naive queda
    identificable (fallback=True) con su motivo registrado."""

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_fallback_registra_modelo_y_motivo(self, mock_es):
        mock_es.return_value.fit.side_effect = ValueError("no se pudo estimar el ajuste inicial")
        serie = _serie(n=30)

        forecast, fallback, motivo = _ajustar_ets(serie, horizonte=3)

        self.assertTrue(fallback)
        self.assertIn("ValueError", motivo)
        self.assertIn("no se pudo estimar", motivo)
        self.assertEqual(len(forecast), 3)

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_pronosticar_ets_sigue_devolviendo_solo_array(self, mock_es):
        """La API pública no cambia: sigue devolviendo np.ndarray, aunque
        por dentro haya fallback."""
        mock_es.return_value.fit.side_effect = ValueError("degenerado")
        serie = _serie(n=30)

        resultado = pronosticar_ets(serie, horizonte=3)

        self.assertIsInstance(resultado, np.ndarray)
        self.assertEqual(len(resultado), 3)

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_linalgerror_tambien_hace_fallback(self, mock_es):
        mock_es.return_value.fit.side_effect = np.linalg.LinAlgError("matriz singular")
        serie = _serie(n=30)

        _forecast, fallback, motivo = _ajustar_ets(serie, horizonte=3)

        self.assertTrue(fallback)
        self.assertIn("LinAlgError", motivo)


class TestNoOcultaErroresDeProgramacion(unittest.TestCase):
    """TEST 9: un error que no es de ajuste/convergencia (p.ej. un bug de
    programación) NO debe quedar atrapado por el fallback — debe
    propagarse."""

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_typeerror_se_propaga(self, mock_es):
        mock_es.return_value.fit.side_effect = TypeError("bug: argumento equivocado")
        serie = _serie(n=30)

        with self.assertRaises(TypeError):
            pronosticar_ets(serie, horizonte=3)

    @patch("src.forecast.modelo_ets.ExponentialSmoothing")
    def test_attributeerror_se_propaga(self, mock_es):
        mock_es.return_value.fit.side_effect = AttributeError("bug: atributo inexistente")
        serie = _serie(n=30)

        with self.assertRaises(AttributeError):
            _ajustar_ets(serie, horizonte=3)


if __name__ == "__main__":
    unittest.main()
