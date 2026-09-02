"""Tests del benchmark Seasonal Naive + drift condicional (ver
src/forecast/benchmark.py). Corren con unittest (stdlib) — el proyecto
no trae pytest como dependencia.
"""

import unittest

import numpy as np
import pandas as pd

from src.forecast.benchmark import (
    PERIODO_ESTACIONAL,
    UMBRAL_P_VALOR_TENDENCIA,
    estimar_tendencia,
    pronosticar_seasonal_naive,
    tiene_tendencia,
)


def _serie(
    n: int,
    beta: float = 0.0,
    amplitud: float = 0.0,
    nivel: float = 100.0,
    ruido: float = 0.0,
    semilla: int = 1,
    inicio: str = "2020-01-01",
) -> pd.Series:
    fechas = pd.date_range(inicio, periods=n, freq="MS")
    t = np.arange(n)
    mes = fechas.month.to_numpy() - 1
    rng = np.random.default_rng(semilla)
    valores = nivel + beta * t + amplitud * np.sin(2 * np.pi * mes / 12) + rng.normal(0, ruido, n)
    return pd.Series(valores, index=fechas)


def _base_seasonal_naive(serie: pd.Series, horizonte: int, periodo: int = PERIODO_ESTACIONAL) -> np.ndarray:
    valores = serie.to_numpy(dtype=float)
    return np.array([valores[-periodo + (h % periodo)] for h in range(horizonte)])


class TestSeasonalPuraSinTendencia(unittest.TestCase):
    """TEST 1 y TEST 5: estacionalidad marcada, sin tendencia -> sin drift."""

    def test_no_activa_drift(self):
        serie = _serie(n=36, beta=0.0, amplitud=50.0, ruido=3.0, semilla=1)
        self.assertFalse(tiene_tendencia(serie))

        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        esperado = _base_seasonal_naive(serie, horizonte=3)
        np.testing.assert_allclose(forecast, esperado)

    def test_fuerte_estacionalidad_sin_tendencia(self):
        serie = _serie(n=48, beta=0.0, amplitud=120.0, ruido=4.0, semilla=5)
        resultado = estimar_tendencia(serie)

        self.assertGreaterEqual(resultado.p_valor, UMBRAL_P_VALOR_TENDENCIA)
        self.assertFalse(resultado.tiene_tendencia)
        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        np.testing.assert_allclose(forecast, _base_seasonal_naive(serie, horizonte=3))


class TestTendenciaPositiva(unittest.TestCase):
    """TEST 2: tendencia positiva + estacionalidad -> drift positivo."""

    def test_beta_positivo_significativo_y_drift_aplicado(self):
        serie = _serie(n=48, beta=2.0, amplitud=40.0, ruido=3.0, semilla=2)
        resultado = estimar_tendencia(serie)

        self.assertGreater(resultado.pendiente, 0)
        self.assertLess(resultado.p_valor, UMBRAL_P_VALOR_TENDENCIA)
        self.assertTrue(resultado.tiene_tendencia)

        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        base = _base_seasonal_naive(serie, horizonte=3)
        esperado = base + resultado.pendiente * np.arange(1, 4)
        np.testing.assert_allclose(forecast, esperado)
        self.assertTrue(np.all(forecast > base))


class TestTendenciaNegativa(unittest.TestCase):
    """TEST 3: tendencia negativa + estacionalidad -> drift negativo."""

    def test_beta_negativo_significativo_y_drift_aplicado(self):
        serie = _serie(n=48, beta=-2.0, amplitud=40.0, nivel=500.0, ruido=3.0, semilla=3)
        resultado = estimar_tendencia(serie)

        self.assertLess(resultado.pendiente, 0)
        self.assertLess(resultado.p_valor, UMBRAL_P_VALOR_TENDENCIA)

        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        base = _base_seasonal_naive(serie, horizonte=3)
        esperado = base + resultado.pendiente * np.arange(1, 4)
        np.testing.assert_allclose(forecast, esperado)
        self.assertTrue(np.all(forecast < base))


class TestDriftRobustoAOutlier(unittest.TestCase):
    """TEST 4: un outlier en el último período engaña al drift de dos
    puntos (último - primero) pero no a la pendiente OLS."""

    def test_usa_beta_ols_no_diferencia_de_extremos(self):
        serie = _serie(n=36, beta=-1.5, amplitud=20.0, nivel=400.0, ruido=2.0, semilla=4)
        serie.iloc[-1] += 100.0  # outlier puntual, no representa la tendencia real

        resultado = estimar_tendencia(serie)
        beta_ols, p_valor = resultado.pendiente, resultado.p_valor
        valores = serie.to_numpy(dtype=float)
        drift_dos_puntos = (valores[-1] - valores[0]) / (len(valores) - 1)

        # el outlier alcanza a invertir el signo del drift ingenuo de dos
        # puntos, pero la pendiente OLS (que usa toda la serie) sigue
        # reflejando la tendencia real: negativa y significativa
        self.assertGreater(drift_dos_puntos, 0)
        self.assertLess(beta_ols, 0)
        self.assertLess(p_valor, UMBRAL_P_VALOR_TENDENCIA)
        self.assertNotAlmostEqual(beta_ols, drift_dos_puntos, places=1)

        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        base = _base_seasonal_naive(serie, horizonte=3)
        esperado_con_beta_ols = base + beta_ols * np.arange(1, 4)
        esperado_con_drift_ingenuo = base + drift_dos_puntos * np.arange(1, 4)

        np.testing.assert_allclose(forecast, esperado_con_beta_ols)
        self.assertFalse(np.allclose(forecast, esperado_con_drift_ingenuo))


class TestNoLeakageEnWalkForward(unittest.TestCase):
    """TEST 6: la regresión sólo debe ver la ventana de entrenamiento
    recibida — un quiebre de tendencia posterior al origen no debe
    filtrarse al pronóstico."""

    def test_quiebre_futuro_no_afecta_pronostico_previo(self):
        fechas = pd.date_range("2020-01-01", periods=48, freq="MS")
        t = np.arange(48)
        mes = fechas.month.to_numpy() - 1
        rng = np.random.default_rng(6)

        valores = 200 + 30 * np.sin(2 * np.pi * mes / 12) + rng.normal(0, 3, 48)
        # quiebre de tendencia recién a partir del mes 24 (fuera del origen usado abajo)
        valores[24:] += 8.0 * (t[24:] - 24)
        serie_completa = pd.Series(valores, index=fechas)

        origen = 24
        entrenamiento = serie_completa.iloc[:origen]

        self.assertFalse(tiene_tendencia(entrenamiento))
        forecast_previo_al_quiebre = pronosticar_seasonal_naive(entrenamiento, horizonte=3)
        np.testing.assert_allclose(
            forecast_previo_al_quiebre, _base_seasonal_naive(entrenamiento, horizonte=3)
        )

        # con el quiebre ya incluido en el histórico, ahora sí se detecta tendencia
        self.assertTrue(tiene_tendencia(serie_completa))


class TestValidacionYRobustez(unittest.TestCase):
    """Casos límite de longitud de serie (item 4 del pedido)."""

    def test_menos_de_12_observaciones_lanza_valueerror(self):
        serie = _serie(n=6, amplitud=10.0, ruido=1.0, semilla=7)
        with self.assertRaises(ValueError):
            pronosticar_seasonal_naive(serie, horizonte=3)

    def test_menos_de_12_observaciones_tiene_tendencia_no_rompe(self):
        serie = _serie(n=6, beta=5.0, ruido=0.5, semilla=7)
        # con tan pocos puntos no hay dummies mensuales: cae a regresión simple
        resultado = tiene_tendencia(serie)
        self.assertIsInstance(resultado, (bool, np.bool_))

    def test_exactamente_12_observaciones_usa_fallback_simple(self):
        serie = _serie(n=12, beta=0.0, amplitud=15.0, ruido=1.0, semilla=8)
        # 12 < MIN_OBSERVACIONES_TENDENCIA_ESTACIONAL (24): regresión simple, sin romper
        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        self.assertEqual(len(forecast), 3)

    def test_24_observaciones_usa_regresion_con_dummies(self):
        serie = _serie(n=24, beta=3.0, amplitud=20.0, ruido=2.0, semilla=9)
        resultado = estimar_tendencia(serie)
        self.assertGreater(resultado.pendiente, 0)
        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        self.assertEqual(len(forecast), 3)

    def test_serie_constante_no_activa_drift(self):
        serie = pd.Series(
            np.full(30, 50.0), index=pd.date_range("2020-01-01", periods=30, freq="MS")
        )
        self.assertFalse(tiene_tendencia(serie))
        forecast = pronosticar_seasonal_naive(serie, horizonte=3)
        np.testing.assert_allclose(forecast, [50.0, 50.0, 50.0])

    def test_series_con_nan_no_rompe(self):
        serie = _serie(n=30, beta=1.0, amplitud=10.0, ruido=1.0, semilla=10)
        serie.iloc[[3, 10, 17]] = np.nan
        resultado = estimar_tendencia(serie)
        self.assertTrue(np.isfinite(resultado.pendiente))
        self.assertTrue(np.isfinite(resultado.p_valor))


class TestBacktestIntegracion(unittest.TestCase):
    """TEST 10: `backtest_walk_forward` (backtest.py) debe usar, en cada
    origen, solamente `serie.iloc[:origen]` como entrenamiento — nunca
    datos posteriores.

    En vez de inspeccionar p-values (frágil: con ventanas chicas y ruido,
    una tendencia espuria puede resultar "significativa" por azar sin que
    eso sea leakage), la forma robusta de probarlo es funcional: dos
    series que comparten el mismo prefijo pero difieren después de cierto
    punto deben producir resultados de backtest IDÉNTICOS en todo origen
    cuyo entrenamiento + test caiga enteramente dentro del prefijo común.
    Si hubiera leakage, ese origen "vería" la diferencia y los resultados
    no coincidirían.
    """

    def test_divergencia_futura_no_afecta_origenes_con_prefijo_comun(self):
        from src.forecast.backtest import backtest_walk_forward

        fechas = pd.date_range("2020-01-01", periods=40, freq="MS")
        mes = fechas.month.to_numpy() - 1
        rng = np.random.default_rng(11)

        base = 200 + 25 * np.sin(2 * np.pi * mes / 12) + rng.normal(0, 2, 40)

        prefijo_comun = 30
        valores_a = base.copy()
        valores_b = base.copy()
        valores_b[prefijo_comun:] += 500.0  # diverge fuerte después del prefijo común

        serie_a = pd.Series(valores_a, index=fechas)
        serie_b = pd.Series(valores_b, index=fechas)

        horizonte = 3
        ventana_minima = 24
        resultados_a = backtest_walk_forward(serie_a, pronosticar_seasonal_naive, horizonte, ventana_minima)
        resultados_b = backtest_walk_forward(serie_b, pronosticar_seasonal_naive, horizonte, ventana_minima)

        # origen (1-indexado) tal que entrenamiento [0:origen) y test
        # [origen:origen+horizonte) quedan dentro del prefijo común
        origenes_sin_divergencia = [
            origen
            for origen in range(ventana_minima, len(serie_a) - horizonte + 1)
            if origen + horizonte <= prefijo_comun
        ]
        self.assertGreater(len(origenes_sin_divergencia), 0)

        for origen in origenes_sin_divergencia:
            fila_a = resultados_a.iloc[origen - ventana_minima]
            fila_b = resultados_b.iloc[origen - ventana_minima]
            np.testing.assert_allclose(fila_a["wape"], fila_b["wape"])
            np.testing.assert_allclose(fila_a["bias"], fila_b["bias"])
            np.testing.assert_allclose(fila_a["mae"], fila_b["mae"])

        # el origen justo antes de la divergencia sí debe diferir (control
        # negativo: confirma que el test detectaría leakage si existiera)
        origen_afectado = prefijo_comun - horizonte + 1
        fila_a = resultados_a.iloc[origen_afectado - ventana_minima]
        fila_b = resultados_b.iloc[origen_afectado - ventana_minima]
        self.assertNotAlmostEqual(fila_a["wape"], fila_b["wape"])


if __name__ == "__main__":
    unittest.main()
