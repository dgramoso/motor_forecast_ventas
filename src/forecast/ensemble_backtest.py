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

ETS y TSB entran acá con el mismo backtest walk-forward que ya corre
`comparar_modelos_con_ensemble` para su propia fila de candidato — no se
reentrenan de nuevo (ver `_backtest_y_predicciones_por_candidato`), el
mismo criterio que ya se aplicó a LightGBM global. Reentrenar dos veces
es el paso más caro del backtest, no algo a repetir por candidato.

LightGBM global recorre sus ventanas por calendario compartido entre
SKUs; ETS/TSB las recorren por posición dentro de la propia serie del
SKU (ver comparar_modelos_global.py sobre el supuesto de calendario
compartido). Coincidir en cantidad de ventanas no alcanza para
combinarlas: se verifica también que los valores reales de cada ventana
coincidan entre los tres candidatos antes de combinar — si no coinciden,
se trata como "sin datos suficientes" para ese SKU en vez de mezclar
ventanas de orígenes distintos.
"""

import numpy as np
import pandas as pd

from src.datos.cargar_datos import serie_por_sku

from .backtest import backtest_walk_forward
from .benchmark import PERIODO_ESTACIONAL
from .comparar_modelos import CANDIDATOS_CON_METADATA, HORIZONTE, VENTANA_MINIMA, comparar_modelos
from .comparar_modelos_global import backtest_y_predicciones_lightgbm_global
from .ensemble import combinar_pronosticos, optimizar_pesos
from .metricas import bias, mae, mase, wape

NOMBRE_ENSEMBLE = "ensemble"
CANDIDATOS_ENSEMBLE = ("ets", "tsb", "lightgbm_global")

# Ajustar 3 pesos (2 grados de libertad, tras la restricción de que
# sumen 1) necesita algo de historia previa para no sobreajustar a un
# puñado de ventanas — piso conservador, sin derivación formal.
MIN_VENTANAS_AJUSTE_PESOS = 5


def evaluar_ensemble_por_sku(
    reales: list[np.ndarray],
    pronosticos_por_candidato: dict[str, list[np.ndarray]],
    min_ventanas_ajuste_pesos: int = MIN_VENTANAS_AJUSTE_PESOS,
    historicos_entrenamiento: list[np.ndarray] | None = None,
    tasas_fallback_componentes: dict[str, float] | None = None,
) -> dict:
    """Métricas del ensemble evaluadas por walk-forward anidado (ver
    docstring del módulo), más los pesos de producción. `reales` y
    `pronosticos_por_candidato` son la misma alineación por ventana que
    usa `ensemble.optimizar_pesos`: la ventana i de `reales` corresponde
    a la ventana i de cada lista de `pronosticos_por_candidato`.

    `historicos_entrenamiento[i]` es el histórico de entrenamiento del
    origen de la ventana i (para MASE, ver metricas.mase) — si se omite,
    `mase_medio` queda en `NaN`. `tasas_fallback_componentes` es
    `{candidato: tasa_fallback_backtest}` de ETS/TSB/LightGBM global — el
    ensemble reporta el máximo de los tres: alcanza con que UNO de los
    candidatos esté cayendo en fallback siempre para que esa parte de la
    combinación no sea un ajuste real (ver CONTEXT.md, "Tasa de
    fallback")."""
    nombres = sorted(pronosticos_por_candidato)
    n_total = len(reales)

    filas_error = []
    for i in range(min_ventanas_ajuste_pesos, n_total):
        pesos_previos = optimizar_pesos(
            reales[:i], {nombre: pronosticos_por_candidato[nombre][:i] for nombre in nombres}
        )
        combinado = combinar_pronosticos(
            {nombre: pronosticos_por_candidato[nombre][i] for nombre in nombres}, pesos_previos
        )
        filas_error.append(
            {
                "wape": wape(reales[i], combinado),
                "bias": bias(reales[i], combinado),
                "mae": mae(reales[i], combinado),
                "mase": (
                    mase(reales[i], combinado, historicos_entrenamiento[i], PERIODO_ESTACIONAL)
                    if historicos_entrenamiento is not None
                    else np.nan
                ),
            }
        )

    tabla_errores = pd.DataFrame(filas_error, columns=["wape", "bias", "mae", "mase"])
    n_ventanas = len(tabla_errores)

    resultado = {
        "n_ventanas": n_ventanas,
        "wape_indefinido": int(tabla_errores["wape"].isna().sum()) if n_ventanas else 0,
        "wape_medio": tabla_errores["wape"].mean() if n_ventanas else np.nan,
        "bias_medio": tabla_errores["bias"].mean() if n_ventanas else np.nan,
        "mae_medio": tabla_errores["mae"].mean() if n_ventanas else np.nan,
        "mase_medio": tabla_errores["mase"].mean() if n_ventanas else np.nan,
        "tasa_fallback_backtest": max(tasas_fallback_componentes.values()) if tasas_fallback_componentes else 0.0,
    }

    pesos_produccion = optimizar_pesos(reales, pronosticos_por_candidato) if n_total > 0 else {n: np.nan for n in nombres}
    resultado.update({f"peso_{nombre}": pesos_produccion[nombre] for nombre in nombres})
    return resultado


def _backtest_y_predicciones_por_candidato(
    ventas: pd.DataFrame, nombre: str, horizonte: int, ventana_minima: int
) -> tuple[pd.DataFrame, dict[str, tuple[list[np.ndarray], list[np.ndarray]]]]:
    """Corre `backtest_walk_forward` de un candidato con metadata de
    fallback (`ets` o `tsb`) UNA sola vez por SKU y devuelve tanto su
    fila agregada (mismo esquema que `comparar_modelos_sku`) como sus
    predicciones crudas por SKU — mismo patrón que
    `backtest_y_predicciones_lightgbm_global`, para no reentrenar el
    mismo candidato dos veces cuando además hace falta para el ensemble
    (ver `comparar_modelos_con_ensemble`)."""
    ajustar_con_metadata = CANDIDATOS_CON_METADATA[nombre]
    filas = []
    predicciones = {}
    for sku_id in ventas["sku_id"].unique():
        serie = serie_por_sku(ventas, sku_id)
        fallbacks: list[bool] = []

        def funcion_pronostico(entrenamiento, h, _ajustar=ajustar_con_metadata, _fallbacks=fallbacks):
            forecast, fallback, _motivo = _ajustar(entrenamiento, h)
            _fallbacks.append(fallback)
            return forecast

        resultados = backtest_walk_forward(serie, funcion_pronostico, horizonte, ventana_minima)
        filas.append(
            {
                "sku_id": sku_id,
                "candidato": nombre,
                "n_ventanas": len(resultados),
                "wape_indefinido": resultados["wape"].isna().sum(),
                "wape_medio": resultados["wape"].mean(),
                "bias_medio": resultados["bias"].mean(),
                "mae_medio": resultados["mae"].mean(),
                "mase_medio": resultados["mase"].mean(),
                "tasa_fallback_backtest": float(np.mean(fallbacks)) if fallbacks else 0.0,
            }
        )
        predicciones[sku_id] = (list(resultados["real"]), list(resultados["pronostico"]))

    return pd.DataFrame(filas), predicciones


def evaluar_ensemble(
    ventas: pd.DataFrame,
    predicciones_por_candidato: dict[str, dict[str, tuple[list[np.ndarray], list[np.ndarray]]]],
    tasas_fallback_por_candidato: dict[str, dict[str, float]],
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
) -> pd.DataFrame:
    """Una fila por SKU con el candidato "ensemble", mismo esquema que
    `comparar_modelos_sku` (para poder concatenar, ver
    `comparar_modelos_con_ensemble`) más las columnas `peso_*` de
    producción. Queda en `NaN` (wape/bias/mae/mase y pesos) si el SKU no
    tuvo ventanas suficientes para los tres candidatos, o si sus
    ventanas no coinciden ventana a ventana (ver docstring del módulo)
    — mismo criterio de "sin datos suficientes" que el resto de los
    candidatos.

    `predicciones_por_candidato` y `tasas_fallback_por_candidato` son
    `{candidato: {sku_id: ...}}` para cada uno de `CANDIDATOS_ENSEMBLE` —
    salida de `_backtest_y_predicciones_por_candidato` (ets/tsb) y de
    `backtest_y_predicciones_lightgbm_global` (lightgbm_global), ya
    calculadas por quien llama: reentrenarlas acá pagaría de nuevo el
    backtest de cada candidato."""
    filas = []
    for sku_id in ventas["sku_id"].unique():
        entradas = {
            nombre: predicciones_por_candidato[nombre].get(sku_id, ([], [])) for nombre in CANDIDATOS_ENSEMBLE
        }
        longitudes = {len(reales) for reales, _pronosticos in entradas.values()}
        alineado = len(longitudes) == 1 and longitudes != {0}

        if alineado:
            n = longitudes.pop()
            reales_por_candidato = {nombre: entradas[nombre][0] for nombre in CANDIDATOS_ENSEMBLE}
            base = reales_por_candidato[CANDIDATOS_ENSEMBLE[0]]
            alineado = all(
                np.allclose(base[i], reales_por_candidato[nombre][i])
                for nombre in CANDIDATOS_ENSEMBLE[1:]
                for i in range(n)
            )

        if not alineado:
            fila = evaluar_ensemble_por_sku([], {nombre: [] for nombre in CANDIDATOS_ENSEMBLE})
        else:
            reales = reales_por_candidato[CANDIDATOS_ENSEMBLE[0]]
            pronosticos_por_candidato = {nombre: entradas[nombre][1] for nombre in CANDIDATOS_ENSEMBLE}
            serie = serie_por_sku(ventas, sku_id)
            historicos = [serie.iloc[: ventana_minima + i].to_numpy() for i in range(n)]
            tasas = {
                nombre: tasas_fallback_por_candidato[nombre].get(sku_id, 0.0) for nombre in CANDIDATOS_ENSEMBLE
            }
            fila = evaluar_ensemble_por_sku(
                reales,
                pronosticos_por_candidato,
                historicos_entrenamiento=historicos,
                tasas_fallback_componentes=tasas,
            )

        filas.append({"sku_id": sku_id, "candidato": NOMBRE_ENSEMBLE} | fila)

    return pd.DataFrame(filas)


def comparar_modelos_con_ensemble(
    ventas: pd.DataFrame,
    horizonte: int = HORIZONTE,
    ventana_minima: int = VENTANA_MINIMA,
    candidatos: dict = CANDIDATOS_CON_METADATA,
) -> pd.DataFrame:
    """La tabla comparativa completa: los candidatos por SKU
    (`comparar_modelos`), el candidato LightGBM global y, si `ets` y
    `tsb` están en `candidatos`, el candidato "ensemble" (ver
    `evaluar_ensemble`) — lista para `seleccionar_modelo.py` sin
    cambiarlo. Las columnas `peso_*` sólo tienen valor en las filas de
    "ensemble"; `NaN` en el resto tras el `concat`. `candidatos` es el
    mismo seam de inyección que `comparar_modelos_sku` (default
    `CANDIDATOS_CON_METADATA`) — y también decide si el ensemble
    participa: sin `ets` y `tsb` no hay con qué combinarlo.

    ETS y TSB corren UNA sola vez cada uno (ver
    `_backtest_y_predicciones_por_candidato`) y sus predicciones
    alimentan tanto su propia fila de candidato como `evaluar_ensemble`
    — igual que LightGBM global. Ninguno de los tres se reentrena dos
    veces."""
    tiene_ets_tsb = "ets" in candidatos and "tsb" in candidatos
    candidatos_resto = {k: v for k, v in candidatos.items() if k not in ("ets", "tsb")} if tiene_ets_tsb else candidatos

    tabla_resto = comparar_modelos(ventas, horizonte, ventana_minima, candidatos_resto)
    tabla_lightgbm, predicciones_lgbm = backtest_y_predicciones_lightgbm_global(ventas, horizonte, ventana_minima)

    if not tiene_ets_tsb:
        return pd.concat([tabla_resto, tabla_lightgbm], ignore_index=True)

    tabla_ets, predicciones_ets = _backtest_y_predicciones_por_candidato(ventas, "ets", horizonte, ventana_minima)
    tabla_tsb, predicciones_tsb = _backtest_y_predicciones_por_candidato(ventas, "tsb", horizonte, ventana_minima)

    predicciones_por_candidato = {"ets": predicciones_ets, "tsb": predicciones_tsb, "lightgbm_global": predicciones_lgbm}
    tasas_por_candidato = {
        "ets": tabla_ets.set_index("sku_id")["tasa_fallback_backtest"].to_dict(),
        "tsb": tabla_tsb.set_index("sku_id")["tasa_fallback_backtest"].to_dict(),
        "lightgbm_global": tabla_lightgbm.set_index("sku_id")["tasa_fallback_backtest"].to_dict(),
    }
    tabla_ensemble = evaluar_ensemble(ventas, predicciones_por_candidato, tasas_por_candidato, horizonte, ventana_minima)

    return pd.concat([tabla_resto, tabla_ets, tabla_tsb, tabla_lightgbm, tabla_ensemble], ignore_index=True)
