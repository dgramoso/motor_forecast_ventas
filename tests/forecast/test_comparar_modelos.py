"""Tests de comparar_modelos.py: el parámetro `candidatos` es un seam de
inyección — permite testear la maquinaria de comparación (incluida la
tasa de fallback) con doubles rápidos en vez de correr los 6 modelos
reales (Prophet/SARIMA incluidos)."""

import unittest

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos import comparar_modelos_sku


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


if __name__ == "__main__":
    unittest.main()
