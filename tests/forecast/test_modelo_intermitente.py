"""Tests de TSB (src/forecast/modelo_intermitente.py): p_t*z_t con
suavizado exponencial, sin fallback (no ajusta ninguna librería externa
que pueda fallar) — mismo contrato de metadata (forecast, fallback,
motivo) que los demás candidatos de comparar_modelos.py."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.modelo_intermitente import _ajustar_tsb, pronosticar_tsb


def _serie(valores: list[float], inicio: str = "2020-01-01") -> pd.Series:
    fechas = pd.date_range(inicio, periods=len(valores), freq="MS")
    return pd.Series(valores, index=fechas, dtype=float)


class TestCasosExtremos(unittest.TestCase):
    def test_serie_completamente_cero_da_forecast_cero(self):
        forecast = pronosticar_tsb(_serie([0] * 10), horizonte=3)

        np.testing.assert_array_equal(forecast, [0.0, 0.0, 0.0])

    def test_un_solo_valor_positivo_no_rompe(self):
        forecast = pronosticar_tsb(_serie([0, 0, 0, 10, 0, 0, 0]), horizonte=2)

        self.assertEqual(len(forecast), 2)
        self.assertTrue(np.isfinite(forecast).all())
        self.assertGreater(forecast[0], 0.0)

    def test_serie_intermitente_da_forecast_positivo_menor_al_pico(self):
        # Ocurrencias esporádicas de magnitud 10 — el nivel p*z queda por
        # debajo del pico porque promedia los períodos sin demanda.
        valores = [10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0, 0]
        forecast = pronosticar_tsb(_serie(valores), horizonte=3)

        self.assertTrue((forecast > 0).all())
        self.assertTrue((forecast < 10).all())

    def test_demanda_frecuente_da_nivel_cercano_al_valor_tipico(self):
        rng = np.random.default_rng(7)
        valores = 100 + rng.normal(0, 5, 30)
        forecast = pronosticar_tsb(_serie(list(valores)), horizonte=1)

        self.assertAlmostEqual(forecast[0], 100, delta=15)

    def test_serie_constante_no_rompe(self):
        forecast = pronosticar_tsb(_serie([10] * 12), horizonte=3)

        np.testing.assert_allclose(forecast, [10.0, 10.0, 10.0])


class TestHorizonte(unittest.TestCase):
    def test_horizonte_mayor_a_uno_devuelve_vector_de_esa_longitud(self):
        forecast = pronosticar_tsb(_serie([5, 0, 5, 0, 5, 0]), horizonte=6)

        self.assertEqual(len(forecast), 6)

    def test_nivel_constante_en_todo_el_horizonte(self):
        # TSB no proyecta tendencia: el nivel p*z se mantiene plano para
        # todos los períodos futuros — es la solución estándar, no una
        # simplificación (ver docstring del módulo).
        forecast = pronosticar_tsb(_serie([5, 0, 5, 0, 5, 0]), horizonte=4)

        self.assertTrue(np.all(forecast == forecast[0]))


class TestSinNegativosNiNan(unittest.TestCase):
    def test_sin_negativos(self):
        forecast = pronosticar_tsb(_serie([0, 0, 3, 0, 0, 8, 0, 0]), horizonte=5)

        self.assertTrue((forecast >= 0).all())

    def test_sin_nan_ni_inf(self):
        forecast = pronosticar_tsb(_serie([0] * 5), horizonte=5)

        self.assertFalse(np.isnan(forecast).any())
        self.assertFalse(np.isinf(forecast).any())


class TestParametrosConfigurables(unittest.TestCase):
    def test_alpha_beta_distintos_dan_forecast_distinto(self):
        serie = _serie([10, 0, 0, 10, 0, 0, 10, 0, 0])

        forecast_default = pronosticar_tsb(serie, horizonte=1)
        forecast_alpha_alto = pronosticar_tsb(serie, horizonte=1, alpha=0.8, beta=0.8)

        self.assertNotAlmostEqual(forecast_default[0], forecast_alpha_alto[0])


class TestContratoConMetadata(unittest.TestCase):
    def test_ajustar_tsb_nunca_hace_fallback(self):
        forecast, fallback, motivo = _ajustar_tsb(_serie([0, 0, 5, 0, 0, 5]), horizonte=2)

        self.assertFalse(fallback)
        self.assertIsNone(motivo)
        self.assertEqual(len(forecast), 2)


if __name__ == "__main__":
    unittest.main()
