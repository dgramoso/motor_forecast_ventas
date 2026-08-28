"""Tests de SARIMA (src/forecast/modelo_sarima.py): usa período estacional
12, y ante un ajuste que falla (en la grilla y en el ajuste final) cae al
benchmark Seasonal Naive con motivo registrado — mismo contrato que
modelo_ets.py."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.forecast.modelo_sarima import _ajustar_sarima, pronosticar_sarima


def _serie(n: int = 30) -> pd.Series:
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    return pd.Series(100 + rng.normal(0, 5, n), index=fechas)


class TestAjusteExitoso(unittest.TestCase):
    @patch("src.forecast.modelo_sarima.SARIMAX")
    def test_no_hace_fallback_y_devuelve_horizonte_correcto(self, mock_sarimax):
        resultado = MagicMock()
        resultado.aic = 10.0
        resultado.forecast.return_value = pd.Series([1.0, 2.0, 3.0])
        mock_sarimax.return_value.fit.return_value = resultado

        forecast, fallback, motivo = _ajustar_sarima(_serie(), horizonte=3)

        self.assertFalse(fallback)
        self.assertIsNone(motivo)
        self.assertEqual(len(forecast), 3)

    @patch("src.forecast.modelo_sarima.SARIMAX")
    def test_usa_periodo_estacional_12(self, mock_sarimax):
        resultado = MagicMock()
        resultado.aic = 10.0
        resultado.forecast.return_value = pd.Series([1.0, 2.0, 3.0])
        mock_sarimax.return_value.fit.return_value = resultado

        pronosticar_sarima(_serie(), horizonte=3)

        _args, kwargs = mock_sarimax.call_args
        self.assertEqual(kwargs["seasonal_order"][3], 12)


class TestFallbackAuditable(unittest.TestCase):
    @patch("src.forecast.modelo_sarima.SARIMAX")
    def test_fallback_registra_motivo_si_todos_los_ajustes_fallan(self, mock_sarimax):
        mock_sarimax.return_value.fit.side_effect = ValueError("no convergió")

        forecast, fallback, motivo = _ajustar_sarima(_serie(), horizonte=3)

        self.assertTrue(fallback)
        self.assertIn("ValueError", motivo)
        self.assertEqual(len(forecast), 3)

    @patch("src.forecast.modelo_sarima.SARIMAX")
    def test_pronosticar_sarima_sigue_devolviendo_solo_array(self, mock_sarimax):
        mock_sarimax.return_value.fit.side_effect = ValueError("degenerado")

        resultado = pronosticar_sarima(_serie(), horizonte=3)

        self.assertIsInstance(resultado, np.ndarray)
        self.assertEqual(len(resultado), 3)


if __name__ == "__main__":
    unittest.main()
