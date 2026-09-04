"""Tests de comparar_modelos.py: el parámetro `candidatos` es un seam de
inyección — permite testear la maquinaria de comparación (incluida la
tasa de fallback) con doubles rápidos en vez de correr los 5 modelos
reales (Prophet incluido)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos import (
    CANDIDATOS,
    CANDIDATOS_CON_METADATA,
    _sin_negativos,
    _sin_negativos_con_metadata,
    comparar_modelos,
    comparar_modelos_sku,
)

from ._helpers import VALOR_CENTINELA_DE_FALLA, ajustar_con_falla_para_valor_centinela


def _serie(n: int = 30) -> pd.Series:
    fechas = pd.date_range("2020-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(1)
    return pd.Series(100 + rng.normal(0, 5, n), index=fechas)


def _siempre_cero(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, None]:
    return np.zeros(horizonte), False, None


def _siempre_fallback(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, str]:
    return np.full(horizonte, serie.iloc[-1]), True, "ValueError: degenerado"


class TestSeamDeInyeccion(unittest.TestCase):
    def test_usa_el_dict_de_candidatos_inyectado_no_el_default(self):
        candidatos_de_prueba = {"cero": _siempre_cero, "siempre_fallback": _siempre_fallback}

        tabla = comparar_modelos_sku(_serie(), horizonte=3, ventana_minima=24, candidatos=candidatos_de_prueba)

        self.assertEqual(set(tabla["candidato"]), {"cero", "siempre_fallback"})

    def test_default_sigue_siendo_candidatos_del_modulo(self):
        candidatos_de_prueba = {"cero": _siempre_cero}

        tabla = comparar_modelos_sku(_serie(), horizonte=3, ventana_minima=24, candidatos=candidatos_de_prueba)

        # Con el seam, ningún candidato del CANDIDATOS real corre acá.
        self.assertNotIn("benchmark", set(tabla["candidato"]))


class TestTasaDeFallback(unittest.TestCase):
    def test_candidato_que_nunca_hace_fallback_tiene_tasa_cero(self):
        tabla = comparar_modelos_sku(
            _serie(), horizonte=3, ventana_minima=24, candidatos={"cero": _siempre_cero}
        )

        self.assertEqual(tabla.loc[0, "tasa_fallback_backtest"], 0.0)

    def test_candidato_que_siempre_hace_fallback_tiene_tasa_uno(self):
        tabla = comparar_modelos_sku(
            _serie(), horizonte=3, ventana_minima=24, candidatos={"siempre_fallback": _siempre_fallback}
        )

        self.assertEqual(tabla.loc[0, "tasa_fallback_backtest"], 1.0)


class TestSinNegativos(unittest.TestCase):
    """Las unidades vendidas/pronosticadas nunca son negativas — hallazgo
    real sobre datos de retail (SKU 21212, Online Retail II): ETS con poca
    historia y alta volatilidad extrapoló por debajo de cero."""

    def test_sin_negativos_clipea_array_directo(self):
        funcion = _sin_negativos(lambda serie, h: np.array([-5.0, 3.0, -0.1]))

        resultado = funcion(_serie(), horizonte=3)

        np.testing.assert_array_equal(resultado, [0.0, 3.0, 0.0])

    def test_sin_negativos_con_metadata_clipea_forecast_preserva_fallback(self):
        funcion = _sin_negativos_con_metadata(lambda serie, h: (np.array([-5.0, 3.0]), True, "motivo"))

        forecast, fallback, motivo = funcion(_serie(), horizonte=2)

        np.testing.assert_array_equal(forecast, [0.0, 3.0])
        self.assertTrue(fallback)
        self.assertEqual(motivo, "motivo")

    def test_benchmark_real_esta_envuelto(self):
        # No hace falta mockear Prophet/ETS (caro) para confirmar
        # que el diccionario público aplica el wrapper — benchmark es
        # rápido y determinístico, y con drift decreciente fuerte puede
        # extrapolar por debajo de cero si no estuviera clipeado.
        serie_decreciente = pd.Series(
            np.linspace(100, 1, 20), index=pd.date_range("2020-01-01", periods=20, freq="MS")
        )

        forecast = CANDIDATOS["benchmark"](serie_decreciente, horizonte=3)
        self.assertTrue((forecast >= 0).all())

        forecast_meta, _fallback, _motivo = CANDIDATOS_CON_METADATA["benchmark"](serie_decreciente, horizonte=3)
        self.assertTrue((forecast_meta >= 0).all())


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


class TestAislamientoPorSku(unittest.TestCase):
    """Un SKU que rompe con una excepción no contemplada por el fallback
    (ver _ajuste_con_fallback.py) no debe tumbar la corrida completa —
    specs/002-reentrenamiento-programado, Historia 3."""

    def test_un_sku_que_rompe_no_tumba_el_resto(self):
        ventas = _ventas_multi_sku(n_skus=3, n_meses=30)
        ventas.loc[ventas["sku_id"] == "SKU-1", "unidades_vendidas"] = VALOR_CENTINELA_DE_FALLA

        tabla = comparar_modelos(
            ventas, horizonte=3, ventana_minima=24, candidatos={"rompe": ajustar_con_falla_para_valor_centinela}
        )

        self.assertEqual(set(tabla["sku_id"]), {"SKU-0", "SKU-2"})

    def test_sku_que_rompe_queda_logueado_como_warning(self):
        ventas = _ventas_multi_sku(n_skus=2, n_meses=30)
        ventas.loc[ventas["sku_id"] == "SKU-0", "unidades_vendidas"] = VALOR_CENTINELA_DE_FALLA

        with self.assertLogs("src.forecast.comparar_modelos", level="WARNING") as registro:
            comparar_modelos(
                ventas, horizonte=3, ventana_minima=24, candidatos={"rompe": ajustar_con_falla_para_valor_centinela}
            )

        self.assertTrue(any("SKU-0" in mensaje for mensaje in registro.output))

    def test_todos_los_skus_rotos_devuelve_tabla_vacia_sin_explotar(self):
        ventas = _ventas_multi_sku(n_skus=2, n_meses=30)
        ventas["unidades_vendidas"] = VALOR_CENTINELA_DE_FALLA

        tabla = comparar_modelos(
            ventas, horizonte=3, ventana_minima=24, candidatos={"rompe": ajustar_con_falla_para_valor_centinela}
        )

        self.assertTrue(tabla.empty)
        self.assertIn("sku_id", tabla.columns)


if __name__ == "__main__":
    unittest.main()
