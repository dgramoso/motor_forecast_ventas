"""Tests de pipeline.py: specs/002-reentrenamiento-programado, Historia 2
(resumen agregado logueado) e Historia 3 (abortar sin persistir ante
falla de una etapa compartida por todos los SKUs)."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.forecast import persistencia
from src.forecast.persistencia import obtener_pronostico_vigente
from src.forecast.pipeline import ejecutar_pipeline


def _ventas_pequenas(n_skus: int = 2, n_meses: int = 30, semilla: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semilla)
    fechas = pd.date_range("2020-01-01", periods=n_meses, freq="MS")
    filas = []
    for i in range(n_skus):
        nivel = 50 + i * 10
        valores = np.maximum(nivel + rng.normal(0, 5, n_meses), 0)
        for fecha, valor in zip(fechas, valores):
            filas.append({"sku_id": f"SKU-{i}", "fecha": fecha, "unidades_vendidas": valor})
    return pd.DataFrame(filas)


class _RutasDeCorridaAisladas:
    """Redirige la persistencia a un directorio temporal, para no escribir
    en data/runs/ real durante los tests."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        ruta_corridas = Path(self._tmp.name) / "corridas.parquet"
        ruta_pronosticos = Path(self._tmp.name) / "pronosticos.parquet"
        self._parches = [
            patch.object(persistencia, "RUTA_CORRIDAS", ruta_corridas),
            patch.object(persistencia, "RUTA_PRONOSTICOS", ruta_pronosticos),
        ]
        for parche in self._parches:
            parche.start()
        return ruta_corridas, ruta_pronosticos

    def __exit__(self, *exc):
        for parche in self._parches:
            parche.stop()
        self._tmp.cleanup()


class TestAbortaSinPersistirAnteFallaDeEtapaCompartida(unittest.TestCase):
    def test_no_persiste_nada_si_cargar_ventas_falla(self):
        with _RutasDeCorridaAisladas() as (ruta_corridas, ruta_pronosticos):
            with patch("src.forecast.pipeline.cargar_ventas", side_effect=RuntimeError("DWH caído")):
                with self.assertLogs("src.forecast.pipeline", level="ERROR"):
                    with self.assertRaises(RuntimeError):
                        ejecutar_pipeline()

            self.assertFalse(ruta_corridas.exists())
            self.assertFalse(ruta_pronosticos.exists())


class TestFlujoPrincipalExitoso(unittest.TestCase):
    """Happy path completo de spec.md sección 7: dispara → persiste →
    loguea resumen agregado → obtener_pronostico_vigente() lo refleja."""

    def test_corrida_exitosa_loguea_resumen_y_queda_vigente(self):
        with _RutasDeCorridaAisladas():
            with patch("src.forecast.pipeline.cargar_ventas", return_value=_ventas_pequenas()):
                with self.assertLogs("src.forecast.pipeline", level="INFO") as registro:
                    resultado = ejecutar_pipeline(horizonte=3, ventana_minima=24)

            mensajes = " ".join(registro.output)
            self.assertIn("SKUs procesados", mensajes)
            self.assertIn("tasa de fallback", mensajes)
            self.assertIn("candidatos ganadores", mensajes)

            vigente = obtener_pronostico_vigente()
            self.assertTrue((vigente["run_id"] == resultado["run_id"]).all())
            self.assertEqual(set(vigente["sku_id"]), {"SKU-0", "SKU-1"})


if __name__ == "__main__":
    unittest.main()
