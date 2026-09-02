"""Ensemble de pronósticos por SKU — combinación lineal por pesos de
predicciones ya calculadas (no reentrena ni corre el backtest de nuevo,
ver `backtest.py`/`comparar_modelos_global.py` para eso). Etapa
preparatoria (sección 15 del pedido): primero tiene que andar bien la
selección de un único ganador por SKU (Fases 4-6) antes de combinar —
esto no reemplaza esa selección, queda disponible para cuando se
justifique con datos.

    ŷ = Σ w_i · ŷ_i        sujeto a w_i >= 0 y Σ w_i = 1
"""

import numpy as np
from scipy.optimize import minimize

from .metricas import wape

# Pesos de referencia (sección 15 del pedido) — puntos de partida
# EXPERIMENTALES para probar la mecánica, no una recomendación: no hay
# ningún respaldo de backtest detrás de estos tres. Claves alineadas con
# los nombres de candidato del proyecto (ver CONTEXT.md, "Candidato").
PESOS_A = {"ets": 0.33, "tsb": 0.33, "lightgbm_global": 0.34}
PESOS_B = {"ets": 0.20, "tsb": 0.20, "lightgbm_global": 0.60}
PESOS_C = {"ets": 0.40, "tsb": 0.40, "lightgbm_global": 0.20}


def combinar_pronosticos(pronosticos: dict[str, np.ndarray], pesos: dict[str, float]) -> np.ndarray:
    """ŷ = Σ w_i · ŷ_i. Exige que `pesos` cubra exactamente los mismos
    modelos que `pronosticos` (no combinar con pesos faltantes o de más),
    que todos sean >= 0 y sumen 1 — una configuración de pesos inválida
    debe fallar explícito, no combinarse en silencio. Nunca negativo,
    igual que el resto de los candidatos (ver `comparar_modelos._sin_negativos`)."""
    if set(pronosticos) != set(pesos):
        raise ValueError(
            f"Los modelos de `pronosticos` ({sorted(pronosticos)}) y de `pesos` "
            f"({sorted(pesos)}) deben coincidir exactamente"
        )
    if any(peso < 0 for peso in pesos.values()):
        raise ValueError("Los pesos deben ser >= 0")
    if not np.isclose(sum(pesos.values()), 1.0):
        raise ValueError(f"Los pesos deben sumar 1 (suman {sum(pesos.values())})")

    combinado = sum(pesos[nombre] * np.asarray(forecast, dtype=float) for nombre, forecast in pronosticos.items())
    return np.maximum(combinado, 0.0)


def optimizar_pesos(
    reales: list[np.ndarray], pronosticos_por_modelo: dict[str, list[np.ndarray]]
) -> dict[str, float]:
    """Pesos que minimizan el WAPE medio sobre las ventanas out-of-sample
    ya evaluadas por el backtest (sección 15: "los pesos deben
    determinarse únicamente utilizando predicciones out-of-sample del
    backtesting" — nunca sobre el ajuste in-sample ni sobre datos
    nuevos). `reales[i]` y `pronosticos_por_modelo[modelo][i]` son la
    misma ventana i para todos los modelos — alinearlos es
    responsabilidad de quien llama (ver `recolectar_predicciones_*`).

    Restricciones w_i >= 0 y Σw_i = 1 vía `scipy.optimize.minimize`
    (SLSQP), arrancando desde pesos iguales entre los modelos para no
    sesgar el punto de partida hacia ninguno."""
    nombres = sorted(pronosticos_por_modelo)
    n_ventanas = len(reales)
    if n_ventanas == 0 or any(len(pronosticos_por_modelo[nombre]) != n_ventanas for nombre in nombres):
        raise ValueError("`reales` y cada lista de `pronosticos_por_modelo` deben tener la misma longitud (> 0)")

    def wape_promedio(pesos_array: np.ndarray) -> float:
        errores = []
        for i in range(n_ventanas):
            combinado = sum(
                pesos_array[j] * np.asarray(pronosticos_por_modelo[nombre][i], dtype=float)
                for j, nombre in enumerate(nombres)
            )
            errores.append(wape(reales[i], combinado))
        errores_validos = [e for e in errores if not np.isnan(e)]
        return float(np.mean(errores_validos)) if errores_validos else np.inf

    n = len(nombres)
    resultado = minimize(
        wape_promedio,
        x0=np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n,
        constraints={"type": "eq", "fun": lambda pesos: np.sum(pesos) - 1.0},
    )

    pesos = np.maximum(resultado.x, 0.0)
    total = pesos.sum()
    pesos = pesos / total if total > 0 else np.full(n, 1.0 / n)
    return dict(zip(nombres, pesos))
