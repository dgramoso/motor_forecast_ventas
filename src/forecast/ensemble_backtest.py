"""Evaluación walk-forward del candidato "ensemble" (ETS + TSB + LightGBM
global) — compite en `seleccionar_modelo.py` como cualquier otro
candidato (ver `comparar_modelos_con_ensemble`).

Los pesos de combinación se ajustan por walk-forward anidado: para
evaluar la ventana i, `optimizar_pesos` (ver ensemble.py) sólo ve las
ventanas ANTERIORES a i — nunca la que se está evaluando. Así el WAPE
del ensemble es genuinamente out-of-sample, comparable al de los demás
candidatos (que tampoco ven su ventana de test al ajustarse). Sin este
cuidado, los pesos quedarían afinados directamente sobre la ventana de
evaluación, dándole al ensemble una ventaja que ningún otro candidato
tiene.

Los primeros `MIN_VENTANAS_AJUSTE_PESOS` orígenes del backtest no se
evalúan — no hay suficiente historia previa para ajustar 3 pesos de
forma confiable. Esto deja al ensemble con menos ventanas evaluadas que
el resto de los candidatos: una limitación conocida, no una fuente de
leakage.

Los pesos de PRODUCCIÓN (los que sirve `pronosticar_futuro.py` cuando el
ensemble gana) se ajustan aparte, sobre TODAS las ventanas out-of-sample
disponibles — igual que el resto de los candidatos, que también se
reajustan sobre todo el histórico al servir el pronóstico futuro (ver
CONTEXT.md, "Candidato"). No hay tensión ahí: la evaluación que decide
el ranking es honesta; producción usa toda la información disponible.
"""

import numpy as np
import pandas as pd

from src.datos.cargar_datos import serie_por_sku

from .backtest import backtest_walk_forward
from .comparar_modelos import CANDIDATOS, CANDIDATOS_CON_METADATA, HORIZONTE, VENTANA_MINIMA, comparar_modelos
from .comparar_modelos_global import backtest_y_predicciones_lightgbm_global
from .ensemble import combinar_pronosticos, optimizar_pesos
from .metricas import bias, mae, wape

NOMBRE_ENSEMBLE = "ensemble"
MODELOS_ENSEMBLE = ("ets", "tsb", "lightgbm_global")

# Ajustar 3 pesos (2 grados de libertad, tras la restricción de que
# sumen 1) necesita algo de historia previa para no sobreajustar a un
# puñado de ventanas — piso conservador, sin derivación formal.
MIN_VENTANAS_AJUSTE_PESOS = 5


def evaluar_ensemble_por_sku(
    reales: list[np.ndarray],
    pronosticos_por_modelo: dict[str, list[np.ndarray]],
    min_ventanas_ajuste_pesos: int = MIN_VENTANAS_AJUSTE_PESOS,
) -> dict:
    """Métricas del ensemble evaluadas por walk-forward anidado (ver
    docstring del módulo), más los pesos de producción. `reales` y
    `pronosticos_por_modelo` son la misma alineación por ventana que usa
    `ensemble.optimizar_pesos` (ver `recolectar_predicciones_*` /
    `backtest_walk_forward`): la ventana i de `reales` corresponde a la
    ventana i de cada lista de `pronosticos_por_modelo`."""
    nombres = sorted(pronosticos_por_modelo)
    n_total = len(reales)

    filas_error = []
    for i in range(min_ventanas_ajuste_pesos, n_total):
        pesos_previos = optimizar_pesos(
            reales[:i], {nombre: pronosticos_por_modelo[nombre][:i] for nombre in nombres}
        )
        combinado = combinar_pronosticos(
            {nombre: pronosticos_por_modelo[nombre][i] for nombre in nombres}, pesos_previos
        )
        filas_error.append(
            {"wape": wape(reales[i], combinado), "bias": bias(reales[i], combinado), "mae": mae(reales[i], combinado)}
        )

    tabla_errores = pd.DataFrame(filas_error, columns=["wape", "bias", "mae"])
    n_ventanas = len(tabla_errores)

    resultado = {
        "n_ventanas": n_ventanas,
        "wape_indefinido": int(tabla_errores["wape"].isna().sum()) if n_ventanas else 0,
        "wape_medio": tabla_errores["wape"].mean() if n_ventanas else np.nan,
        "bias_medio": tabla_errores["bias"].mean() if n_ventanas else np.nan,
        "mae_medio": tabla_errores["mae"].mean() if n_ventanas else np.nan,
        # MASE necesita el histórico de entrenamiento de cada origen (ver
        # metricas.mase); no se calcula acá para no arrastrar la serie
        # completa a través del ajuste de pesos. seleccionar_modelo.py no
        # depende de esta columna (usa wape_medio y bias_medio).
        "mase_medio": np.nan,
        # El ensemble no "cae en fallback" como candidato único: es una
        # combinación de candidatos que ya son defensivos por su cuenta
        # (ver CONTEXT.md, "Fallback") — no hay una tasa propia que
        # reportar acá.
        "tasa_fallback_backtest": 0.0,
    }

    pesos_produccion = optimizar_pesos(reales, pronosticos_por_modelo) if n_total > 0 else {n: np.nan for n in nombres}
    resultado.update({f"peso_{nombre}": pesos_produccion[nombre] for nombre in nombres})
    return resultado


def evaluar_ensemble(
    ventas: pd.DataFrame,
    predicciones_lgbm: dict[str, tuple[list, list]],
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
) -> pd.DataFrame:
    """Una fila por SKU con el candidato "ensemble", mismo esquema que
    `comparar_modelos_sku` (para poder concatenar, ver
    `comparar_modelos_con_ensemble`) más las columnas `peso_*` de
    producción. Queda en `NaN` (wape/bias/mae y pesos) si el SKU no tuvo
    ventanas suficientes para los tres modelos — mismo criterio de "sin
    datos suficientes" que el resto de los candidatos.

    `predicciones_lgbm` (salida de `backtest_y_predicciones_lightgbm_global`
    o de `recolectar_predicciones_lightgbm_global`) se recibe ya
    calculado a propósito: quien llama normalmente también necesita la
    tabla del candidato "lightgbm_global", y reentrenarlo acá pagaría
    dos veces el paso más caro del backtest."""
    filas = []
    for sku_id in ventas["sku_id"].unique():
        reales_lgbm, pronosticos_lgbm = predicciones_lgbm[sku_id]
        serie = serie_por_sku(ventas, sku_id)
        backtest_ets = backtest_walk_forward(serie, CANDIDATOS["ets"], horizonte, ventana_minima)
        backtest_tsb = backtest_walk_forward(serie, CANDIDATOS["tsb"], horizonte, ventana_minima)

        n = len(reales_lgbm)
        if n == 0 or len(backtest_ets) != n or len(backtest_tsb) != n:
            fila = evaluar_ensemble_por_sku([], {"ets": [], "tsb": [], "lightgbm_global": []})
        else:
            pronosticos_por_modelo = {
                "ets": list(backtest_ets["pronostico"]),
                "tsb": list(backtest_tsb["pronostico"]),
                "lightgbm_global": pronosticos_lgbm,
            }
            fila = evaluar_ensemble_por_sku(reales_lgbm, pronosticos_por_modelo)

        filas.append({"sku_id": sku_id, "candidato": NOMBRE_ENSEMBLE} | fila)

    return pd.DataFrame(filas)


def comparar_modelos_con_ensemble(
    ventas: pd.DataFrame,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    candidatos: dict = CANDIDATOS_CON_METADATA,
) -> pd.DataFrame:
    """La tabla comparativa completa: los candidatos por SKU
    (`comparar_modelos`), el candidato LightGBM global y el candidato
    "ensemble" (ver `evaluar_ensemble`), lista para
    `seleccionar_modelo.py` sin cambiarlo — las columnas `peso_*` sólo
    tienen valor en las filas de "ensemble"; `NaN` en el resto tras el
    `concat`. `candidatos` es el mismo seam de inyección que
    `comparar_modelos_sku` (default `CANDIDATOS_CON_METADATA`).

    El backtest de LightGBM global corre UNA sola vez y sus predicciones
    alimentan tanto la fila `"lightgbm_global"` como `evaluar_ensemble`:
    calcularlo dos veces sería pagar dos veces el paso más caro del
    backtest."""
    tabla_por_sku = comparar_modelos(ventas, horizonte, ventana_minima, candidatos)
    tabla_lightgbm, predicciones_lgbm = backtest_y_predicciones_lightgbm_global(ventas, horizonte, ventana_minima)
    tabla_ensemble = evaluar_ensemble(ventas, predicciones_lgbm, horizonte, ventana_minima)
    return pd.concat([tabla_por_sku, tabla_lightgbm, tabla_ensemble], ignore_index=True)
