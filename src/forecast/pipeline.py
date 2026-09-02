"""Pipeline end-to-end del motor de forecast: ingesta → ejecución y
comparación de modelos → selección del mejor modelo por SKU →
pronóstico futuro → persistencia de la corrida.

Ver .scratch/motor-forecast-pipeline/map.md para las decisiones detrás
de cada paso.
"""

from src.datos.cargar_datos import cargar_ventas
from src.forecast.comparar_modelos import HORIZONTE, VENTANA_MINIMA
from src.forecast.comparar_modelos_global import comparar_modelos_con_lightgbm_global
from src.forecast.ensemble_informativo import evaluar_ensemble_informativo, pronosticar_futuro_ensemble
from src.forecast.persistencia import (
    RUTA_CORRIDAS,
    RUTA_PRONOSTICOS,
    guardar_corrida,
    guardar_evaluacion_ensemble,
    guardar_pronosticos,
    guardar_pronosticos_ensemble,
    nuevo_run_id,
)
from src.forecast.pronosticar_futuro import pronosticar_futuro
from src.forecast.seleccionar_modelo import seleccionar_mejor_modelo


def ejecutar_pipeline(horizonte: int = HORIZONTE, ventana_minima: int = VENTANA_MINIMA) -> dict:
    ventas = cargar_ventas()
    tabla_comparativa = comparar_modelos_con_lightgbm_global(ventas, horizonte, ventana_minima)
    selecciones = seleccionar_mejor_modelo(tabla_comparativa)
    pronostico = pronosticar_futuro(ventas, tabla_comparativa, horizonte)

    # Vista comparativa del ensemble ETS+TSB+LightGBM global — no compite
    # en `selecciones` (ver ensemble_informativo.py sobre por qué no).
    evaluacion_ensemble = evaluar_ensemble_informativo(ventas, horizonte, ventana_minima)
    pronostico_ensemble = pronosticar_futuro_ensemble(ventas, evaluacion_ensemble, horizonte)

    run_id = nuevo_run_id()
    guardar_corrida(run_id, selecciones)
    guardar_pronosticos(run_id, pronostico)
    guardar_evaluacion_ensemble(run_id, evaluacion_ensemble)
    if not pronostico_ensemble.empty:
        guardar_pronosticos_ensemble(run_id, pronostico_ensemble)

    return {
        "run_id": run_id,
        "tabla_comparativa": tabla_comparativa,
        "selecciones": selecciones,
        "pronostico": pronostico,
        "evaluacion_ensemble": evaluacion_ensemble,
        "pronostico_ensemble": pronostico_ensemble,
    }


def main() -> None:
    resultado = ejecutar_pipeline()

    print(f"Corrida: {resultado['run_id']}\n")
    print("Modelo elegido por SKU:")
    print(resultado["selecciones"].round(3).to_string(index=False))
    pronostico = resultado["pronostico"].assign(
        unidades_pronosticadas=lambda df: df["unidades_pronosticadas"].round(1)
    )
    print("\nPronóstico futuro:")
    print(pronostico.to_string(index=False))

    print("\nEnsemble ETS+TSB+LightGBM global (informativo, no compite en la selección):")
    print(resultado["evaluacion_ensemble"].round(3).to_string(index=False))
    print(f"\nPersistido en {RUTA_CORRIDAS} y {RUTA_PRONOSTICOS}")


if __name__ == "__main__":
    main()
