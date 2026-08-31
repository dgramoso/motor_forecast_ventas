"""Tests de MASE (src/forecast/metricas.py) — WAPE/bias/MAE ya se
ejercitan indirectamente vía test_backtest en test_benchmark.py."""

import unittest

import numpy as np

from src.forecast.metricas import mase


class TestMASE(unittest.TestCase):
    def test_historico_insuficiente_da_nan(self):
        self.assertTrue(np.isnan(mase([10.0], [10.0], historico_entrenamiento=[5.0], periodo_estacional=12)))

    def test_historico_constante_da_nan_no_division_por_cero(self):
        # Naive in-sample con error cero: no hay variación que escalar.
        historico = [10.0] * 20
        self.assertTrue(np.isnan(mase([10.0, 10.0], [10.0, 10.0], historico, periodo_estacional=12)))

    def test_pronostico_perfecto_da_cero(self):
        rng = np.random.default_rng(0)
        historico = 100 + rng.normal(0, 5, 24)
        real = [105.0, 110.0]
        self.assertEqual(mase(real, real, historico, periodo_estacional=12), 0.0)

    def test_pronostico_peor_que_naive_da_mayor_a_uno(self):
        rng = np.random.default_rng(0)
        historico = 100 + rng.normal(0, 2, 24)
        real = [100.0, 100.0]
        pronostico_malo = [200.0, 200.0]
        self.assertGreater(mase(real, pronostico_malo, historico, periodo_estacional=12), 1.0)

    def test_cae_a_naive_de_un_paso_si_no_alcanza_para_estacional(self):
        # 6 observaciones no alcanzan para periodo_estacional=12, pero sí
        # para el naive de un paso (m=1).
        historico = [10.0, 12.0, 9.0, 11.0, 10.0, 13.0]
        resultado = mase([10.0], [10.0], historico, periodo_estacional=12)
        self.assertFalse(np.isnan(resultado))


if __name__ == "__main__":
    unittest.main()
