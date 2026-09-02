"""Persistencia de corridas y pronósticos — esquema decidido en
.scratch/motor-forecast-pipeline/issues/02-esquema-persistencia.md:
parquet plano, append-only, dos archivos. `run_id` es un uuid4 por
corrida completa del pipeline (compartido entre todos los SKUs de esa
ejecución). No hay "vigente" persistido aparte: se deriva como la
corrida de mayor `timestamp_utc` por SKU.
"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RUTA_CORRIDAS = Path("data/runs/corridas.parquet")
RUTA_PRONOSTICOS = Path("data/runs/pronosticos.parquet")
RUTA_EVALUACION_ENSEMBLE = Path("data/runs/evaluacion_ensemble.parquet")
RUTA_PRONOSTICOS_ENSEMBLE = Path("data/runs/pronosticos_ensemble.parquet")


def nuevo_run_id() -> str:
    return uuid.uuid4().hex


def _agregar(ruta: Path, filas_nuevas: pd.DataFrame) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if ruta.exists():
        filas_nuevas = pd.concat([pd.read_parquet(ruta), filas_nuevas], ignore_index=True)
    filas_nuevas.to_parquet(ruta, index=False)


def guardar_corrida(run_id: str, selecciones: pd.DataFrame) -> None:
    """`selecciones` es la salida de `seleccionar_mejor_modelo` (sku_id,
    candidato, wape_medio, bias_medio, mae_medio, tasa_fallback_backtest,
    sin_datos_suficientes — ver CONTEXT.md)."""
    filas = selecciones.copy()
    filas.insert(0, "run_id", run_id)
    filas.insert(1, "timestamp_utc", datetime.now(timezone.utc))
    _agregar(RUTA_CORRIDAS, filas)


def guardar_pronosticos(run_id: str, pronostico: pd.DataFrame) -> None:
    """`pronostico` es la salida de `pronosticar_futuro` (sku_id, fecha,
    candidato, unidades_pronosticadas, fallback, más el diagnóstico de
    demanda de diagnostico_demanda.py — se persiste tal cual, sin
    whitelist de columnas, para no perder trazabilidad)."""
    filas = pronostico.copy()
    filas.insert(0, "run_id", run_id)
    _agregar(RUTA_PRONOSTICOS, filas)


def guardar_evaluacion_ensemble(run_id: str, evaluacion_ensemble: pd.DataFrame) -> None:
    """`evaluacion_ensemble` es la salida de `evaluar_ensemble_informativo`
    (ver ensemble_informativo.py) — vista comparativa, no compite en
    `seleccionar_modelo.py`; se persiste aparte de `corridas.parquet`
    para no confundirla con el candidato realmente seleccionado por SKU."""
    filas = evaluacion_ensemble.copy()
    filas.insert(0, "run_id", run_id)
    filas.insert(1, "timestamp_utc", datetime.now(timezone.utc))
    _agregar(RUTA_EVALUACION_ENSEMBLE, filas)


def guardar_pronosticos_ensemble(run_id: str, pronostico_ensemble: pd.DataFrame) -> None:
    """`pronostico_ensemble` es la salida de `pronosticar_futuro_ensemble`
    — aparte de `pronosticos.parquet` por el mismo motivo que
    `guardar_evaluacion_ensemble`."""
    filas = pronostico_ensemble.copy()
    filas.insert(0, "run_id", run_id)
    _agregar(RUTA_PRONOSTICOS_ENSEMBLE, filas)


def obtener_pronostico_vigente(sku_id: str | None = None) -> pd.DataFrame:
    """El pronóstico de la corrida más reciente, por SKU (o de un SKU puntual)."""
    corridas = pd.read_parquet(RUTA_CORRIDAS)
    pronosticos = pd.read_parquet(RUTA_PRONOSTICOS)

    vigentes = corridas.sort_values("timestamp_utc").groupby("sku_id").tail(1)
    if sku_id is not None:
        vigentes = vigentes[vigentes["sku_id"] == sku_id]

    return pronosticos.merge(vigentes[["run_id", "sku_id", "candidato"]], on=["run_id", "sku_id"])
