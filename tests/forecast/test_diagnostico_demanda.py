"""Tests de ADI, CV² y clasificación SBC (src/forecast/diagnostico_demanda.py)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.diagnostico_demanda import adi, clasificar_demanda, cv2, tasa_de_ceros


def _serie(valores: list[float]) -> pd.Series:
    fechas = pd.date_range("2020-01-01", periods=len(valores), freq="MS")
    return pd.Series(valores, index=fechas, dtype=float)


class TestTasaDeCeros(unittest.TestCase):
    def test_serie_completamente_cero(self):
        self.assertEqual(tasa_de_ceros(_serie([0, 0, 0, 0])), 1.0)

    def test_sin_ceros(self):
        self.assertEqual(tasa_de_ceros(_serie([1, 2, 3, 4])), 0.0)


class TestADI(unittest.TestCase):
    def test_serie_completamente_cero_da_infinito(self):
        self.assertTrue(np.isinf(adi(_serie([0, 0, 0, 0]))))

    def test_demanda_todos_los_periodos_da_uno(self):
        self.assertEqual(adi(_serie([1, 2, 3, 4])), 1.0)

    def test_un_solo_valor_positivo(self):
        # 7 períodos, 1 con demanda -> ADI = 7
        self.assertEqual(adi(_serie([0, 0, 0, 10, 0, 0, 0])), 7.0)


class TestCV2(unittest.TestCase):
    def test_serie_completamente_cero_da_cero_no_nan(self):
        self.assertEqual(cv2(_serie([0, 0, 0, 0])), 0.0)

    def test_un_solo_valor_positivo_da_cero_no_nan(self):
        self.assertEqual(cv2(_serie([0, 0, 0, 10, 0, 0, 0])), 0.0)

    def test_serie_constante_da_cero(self):
        self.assertEqual(cv2(_serie([10, 10, 10, 10])), 0.0)

    def test_demanda_positiva_variable_da_valor_mayor_a_cero(self):
        self.assertGreater(cv2(_serie([5, 0, 20, 0, 8, 0])), 0.0)


class TestClasificarDemanda(unittest.TestCase):
    def test_serie_completamente_cero_es_sin_demanda(self):
        self.assertEqual(clasificar_demanda(_serie([0] * 10)), "sin_demanda")

    def test_serie_constante_es_regular(self):
        self.assertEqual(clasificar_demanda(_serie([10] * 12)), "regular")

    def test_pocos_ceros_y_baja_variabilidad_es_regular(self):
        rng = np.random.default_rng(3)
        valores = 100 + rng.normal(0, 2, 24)
        self.assertEqual(clasificar_demanda(_serie(list(valores))), "regular")

    def test_muchos_ceros_y_magnitud_estable_es_intermitente(self):
        # ADI alto (pocas ocurrencias) pero siempre la misma magnitud -> CV2 bajo
        valores = [10 if i % 6 == 0 else 0 for i in range(24)]
        self.assertEqual(clasificar_demanda(_serie(valores)), "intermitente")

    def test_muchos_ceros_y_magnitud_variable_es_lumpy(self):
        valores = [0] * 24
        magnitudes = [5, 50, 8, 90, 3, 70]
        for i, m in zip(range(0, 24, 4), magnitudes):
            valores[i] = m
        self.assertEqual(clasificar_demanda(_serie(valores)), "lumpy")


if __name__ == "__main__":
    unittest.main()
