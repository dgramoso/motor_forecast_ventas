"""Pipeline end-to-end del motor de forecast: ingesta → ejecución y
comparación de modelos → selección del mejor modelo por SKU →
pronóstico futuro → persistencia de la corrida.

Ver .scratch/motor-forecast-pipeline/map.md para las decisiones detrás
de cada paso.
"""

import logging
from pathlib import Path

import pandas as pd

from src.datos.cargar_datos import cargar_ventas
from src.forecast.comparar_modelos import HORIZONTE, VENTANA_MINIMA
from src.forecast.ensemble_backtest import comparar_modelos_con_ensemble
from src.forecast.persistencia import RUTA_CORRIDAS, RUTA_PRONOSTICOS, guardar_corrida, guardar_pronosticos, nuevo_run_id
from src.forecast.pronosticar_futuro import pronosticar_futuro
from src.forecast.seleccionar_modelo import seleccionar_mejor_modelo

logger = logging.getLogger(__name__)


def _resumen_agregado(selecciones: pd.DataFrame) -> str:
    """SKUs procesados, tasa de fallback promedio y distribución de
    candidatos ganadores — el resumen que se loguea en cada corrida
    exitosa (specs/002-reentrenamiento-programado, Historia 2)."""
    return (
        f"SKUs procesados: {len(selecciones)} | "
        f"tasa de fallback promedio: {selecciones['tasa_fallback_backtest'].mean():.3f} | "
        f"candidatos ganadores: {selecciones['candidato'].value_counts().to_dict()}"
    )


def ejecutar_pipeline(horizonte: int = HORIZONTE, ventana_minima: int = VENTANA_MINIMA) -> dict:
    # Nada se persiste hasta guardar_corrida/guardar_pronosticos — si
    # una etapa compartida por todos los SKUs (carga de datos, LightGBM
    # global) rompe acá, la corrida se aborta sin persistir nada, sin
    # reintento (specs/002-reentrenamiento-programado, sección 6).
    try:
        ventas = cargar_ventas()
        tabla_comparativa = comparar_modelos_con_ensemble(ventas, horizonte, ventana_minima)
        selecciones = seleccionar_mejor_modelo(tabla_comparativa)
        pronostico = pronosticar_futuro(ventas, tabla_comparativa, horizonte)
    except Exception:
        logger.error("Corrida abortada antes de persistir nada: falló una etapa compartida", exc_info=True)
        raise

    run_id = nuevo_run_id()
    guardar_corrida(run_id, selecciones)
    guardar_pronosticos(run_id, pronostico)

    logger.info("Corrida %s completada. %s", run_id, _resumen_agregado(selecciones))

    return {
        "run_id": run_id,
        "tabla_comparativa": tabla_comparativa,
        "selecciones": selecciones,
        "pronostico": pronostico,
    }


RUTA_LOG = Path("logs/corridas_programadas.log")


def main() -> None:
    RUTA_LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(RUTA_LOG, encoding="utf-8")])

    resultado = ejecutar_pipeline()

    print(f"Corrida: {resultado['run_id']}\n")
    print("Modelo elegido por SKU:")
    print(resultado["selecciones"].round(3).to_string(index=False))
    pronostico = resultado["pronostico"].assign(
        unidades_pronosticadas=lambda df: df["unidades_pronosticadas"].round(1)
    )
    print("\nPronóstico futuro:")
    print(pronostico.to_string(index=False))
    print(f"\nPersistido en {RUTA_CORRIDAS} y {RUTA_PRONOSTICOS}")


if __name__ == "__main__":
    main()
