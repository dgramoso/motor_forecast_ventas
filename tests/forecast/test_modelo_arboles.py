"""Check mínimo de `_modelo_arboles.py`: construcción de lags/medias
móviles y la precondición de filas mínimas antes de ajustar árboles."""

import unittest

import numpy as np
import pandas as pd

from src.forecast._modelo_arboles import (
    MIN_FILAS_ENTRENAMIENTO,
    _construir_features,
    pronosticar_directo,
)


class TestConstruirFeatures(unittest.TestCase):
    def test_lags_y_media_movil_correctos_en_la_ultima_fila(self):
        fechas = pd.date_range("2020-01-01", periods=24, freq="MS")
        serie = pd.Series(np.arange(1, 25, dtype=float), index=fechas)

        features = _construir_features(serie)
        ultima = features.iloc[-1]

        self.assertEqual(ultima["lag_1"], 24)
        self.assertEqual(ultima["lag_2"], 23)
        self.assertEqual(ultima["lag_3"], 22)
        self.assertEqual(ultima["lag_12"], 13)
        self.assertAlmostEqual(ultima["media_movil_3"], np.mean([22, 23, 24]))
        self.assertAlmostEqual(ultima["media_movil_12"], np.mean(range(13, 25)))


class TestPrecondicionFilasMinimas(unittest.TestCase):
    def test_historial_corto_lanza_valueerror(self):
        fechas = pd.date_range("2020-01-01", periods=13, freq="MS")
        serie = pd.Series(np.arange(13, dtype=float), index=fechas)

        with self.assertRaises(ValueError):
            pronosticar_directo(serie, horizonte=3, crear_estimador=lambda: _EstimadorConstante())

    def test_historial_suficiente_no_lanza(self):
        fechas = pd.date_range("2020-01-01", periods=13 + MIN_FILAS_ENTRENAMIENTO + 3, freq="MS")
        serie = pd.Series(np.arange(len(fechas), dtype=float), index=fechas)

        forecast = pronosticar_directo(serie, horizonte=3, crear_estimador=lambda: _EstimadorConstante())
        self.assertEqual(len(forecast), 3)


class _EstimadorConstante:
    """Doble de test: no depende de sklearn/xgboost instalados."""

    def fit(self, x, y):
        self._prediccion = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self._prediccion)


if __name__ == "__main__":
    unittest.main()
