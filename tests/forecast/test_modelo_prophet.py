"""Tests de Prophet (src/forecast/modelo_prophet.py): config para datos
mensuales (solo estacionalidad anual, aditiva), y fallback auditable a
Seasonal Naive ante un ajuste que falla — mismo contrato que
modelo_ets.py."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast.modelo_prophet import _ajustar_prophet, pronosticar_prophet


def _serie(n: int = 30) -> pd.Series:
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    return pd.Series(100 + rng.normal(0, 5, n), index=fechas)


class TestAjusteExitoso(unittest.TestCase):
    @patch("src.forecast.modelo_prophet.Prophet")
    def test_no_hace_fallback_y_devuelve_horizonte_correcto(self, mock_prophet):
        mock_prophet.return_value.predict.return_value = pd.DataFrame({"yhat": [1.0, 2.0, 3.0]})

        forecast, fallback, motivo = _ajustar_prophet(_serie(), horizonte=3)

        self.assertFalse(fallback)
        self.assertIsNone(motivo)
        self.assertEqual(len(forecast), 3)

    @patch("src.forecast.modelo_prophet.Prophet")
    def test_config_mensual_sin_estacionalidad_diaria_ni_semanal(self, mock_prophet):
        mock_prophet.return_value.predict.return_value = pd.DataFrame({"yhat": [1.0, 2.0, 3.0]})

        pronosticar_prophet(_serie(), horizonte=3)

        _args, kwargs = mock_prophet.call_args
        self.assertEqual(kwargs["growth"], "linear")
        self.assertTrue(kwargs["yearly_seasonality"])
        self.assertFalse(kwargs["weekly_seasonality"])
        self.assertFalse(kwargs["daily_seasonality"])
        self.assertEqual(kwargs["seasonality_mode"], "additive")


class TestFallbackAuditable(unittest.TestCase):
    @patch("src.forecast.modelo_prophet.Prophet")
    def test_fallback_registra_motivo_ante_valueerror(self, mock_prophet):
        mock_prophet.return_value.fit.side_effect = ValueError("historia insuficiente")

        forecast, fallback, motivo = _ajustar_prophet(_serie(), horizonte=3)

        self.assertTrue(fallback)
        self.assertIn("ValueError", motivo)
        self.assertEqual(len(forecast), 3)

    @patch("src.forecast.modelo_prophet.Prophet")
    def test_fallback_registra_motivo_ante_runtimeerror(self, mock_prophet):
        mock_prophet.return_value.fit.side_effect = RuntimeError("optimización de Stan falló")

        _forecast, fallback, motivo = _ajustar_prophet(_serie(), horizonte=3)

        self.assertTrue(fallback)
        self.assertIn("RuntimeError", motivo)

    @patch("src.forecast.modelo_prophet.Prophet")
    def test_pronosticar_prophet_sigue_devolviendo_solo_array(self, mock_prophet):
        mock_prophet.return_value.fit.side_effect = ValueError("degenerado")

        resultado = pronosticar_prophet(_serie(), horizonte=3)

        self.assertIsInstance(resultado, np.ndarray)
        self.assertEqual(len(resultado), 3)


if __name__ == "__main__":
    unittest.main()
