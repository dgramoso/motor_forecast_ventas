"""Ensemble ETS + TSB + LightGBM global — evaluación y pronóstico futuro
como vista comparativa, sin competir en `seleccionar_modelo.py`.

No compite en el ranking real porque `optimizar_pesos` (ver ensemble.py)
ajusta los pesos sobre las MISMAS ventanas out-of-sample con las que
después se mediría su WAPE: los demás candidatos no tuvieron ese ajuste
sobre esas ventanas, así que no sería una comparación justa. Mientras esa
decisión no se resuelva (walk-forward anidado de pesos, o alguna otra
forma de evitarlo), el ensemble queda fuera de `seleccionar_mejor_modelo`
y se reporta aparte, para trazabilidad y para decidir con datos si vale
la pena resolverlo.
"""

import numpy as np
import pandas as pd

from src.datos.cargar_datos import serie_por_sku

from .backtest import backtest_walk_forward
from .comparar_modelos import CANDIDATOS, CANDIDATOS_CON_METADATA, HORIZONTE, VENTANA_MINIMA
from .comparar_modelos_global import recolectar_predicciones_lightgbm_global
from .ensemble import combinar_pronosticos, optimizar_pesos
from .metricas import wape
from .pronosticar_futuro import pronosticar_futuro_lightgbm_global

NOMBRE_ENSEMBLE = "ensemble"
MODELOS_ENSEMBLE = ("ets", "tsb", "lightgbm_global")


def _wape_medio_combinado(
    reales: list[np.ndarray], pronosticos_por_modelo: dict[str, list[np.ndarray]], pesos: dict[str, float]
) -> float:
    errores = []
    for i in range(len(reales)):
        combinado = combinar_pronosticos({nombre: arr[i] for nombre, arr in pronosticos_por_modelo.items()}, pesos)
        errores.append(wape(reales[i], combinado))
    errores_validos = [e for e in errores if not np.isnan(e)]
    return float(np.mean(errores_validos)) if errores_validos else np.nan


def evaluar_ensemble_informativo(
    ventas: pd.DataFrame, horizonte: int = HORIZONTE, ventana_minima: int = VENTANA_MINIMA
) -> pd.DataFrame:
    """Una fila por SKU: WAPE medio y pesos óptimos del ensemble ETS + TSB
    + LightGBM global, ajustados sobre las mismas ventanas del backtest
    walk-forward de cada modelo. `wape_medio`/pesos quedan en `NaN` si el
    SKU no tuvo ninguna ventana con historia suficiente para los tres
    modelos (mismo criterio que `seleccionar_mejor_modelo_sku` para "sin
    datos suficientes")."""
    predicciones_lgbm = recolectar_predicciones_lightgbm_global(ventas, horizonte, ventana_minima)

    filas = []
    for sku_id in ventas["sku_id"].unique():
        reales_lgbm, pronosticos_lgbm = predicciones_lgbm[sku_id]
        serie = serie_por_sku(ventas, sku_id)
        backtest_ets = backtest_walk_forward(serie, CANDIDATOS["ets"], horizonte, ventana_minima)
        backtest_tsb = backtest_walk_forward(serie, CANDIDATOS["tsb"], horizonte, ventana_minima)

        n_ventanas = len(reales_lgbm)
        if n_ventanas == 0 or len(backtest_ets) != n_ventanas or len(backtest_tsb) != n_ventanas:
            filas.append(
                {"sku_id": sku_id, "candidato": NOMBRE_ENSEMBLE, "n_ventanas": 0, "wape_medio": np.nan}
                | {f"peso_{nombre}": np.nan for nombre in MODELOS_ENSEMBLE}
            )
            continue

        pronosticos_por_modelo = {
            "ets": list(backtest_ets["pronostico"]),
            "tsb": list(backtest_tsb["pronostico"]),
            "lightgbm_global": pronosticos_lgbm,
        }
        pesos = optimizar_pesos(reales_lgbm, pronosticos_por_modelo)
        filas.append(
            {
                "sku_id": sku_id,
                "candidato": NOMBRE_ENSEMBLE,
                "n_ventanas": n_ventanas,
                "wape_medio": _wape_medio_combinado(reales_lgbm, pronosticos_por_modelo, pesos),
            }
            | {f"peso_{nombre}": pesos[nombre] for nombre in MODELOS_ENSEMBLE}
        )

    return pd.DataFrame(filas)


def pronosticar_futuro_ensemble(
    ventas: pd.DataFrame, evaluacion_ensemble: pd.DataFrame, horizonte: int = HORIZONTE
) -> pd.DataFrame:
    """Pronóstico futuro del ensemble para las SKUs con pesos válidos en
    `evaluacion_ensemble` (ver `evaluar_ensemble_informativo`) — SKUs sin
    datos suficientes quedan afuera, no hay pesos con qué combinar. Ajusta
    ETS y TSB sobre todo el histórico (igual que `pronosticar_futuro_sku`)
    y reusa el pronóstico futuro de LightGBM global (un solo entrenamiento
    para todas las SKUs, ver `pronosticar_futuro_lightgbm_global`)."""
    validas = evaluacion_ensemble.dropna(subset=["wape_medio"])
    if validas.empty:
        return pd.DataFrame()

    skus = validas["sku_id"].tolist()
    pron_lgbm_futuro = pronosticar_futuro_lightgbm_global(ventas, skus, horizonte).set_index("sku_id")

    filas = []
    for _, fila in validas.iterrows():
        sku_id = fila["sku_id"]
        pesos = {nombre: fila[f"peso_{nombre}"] for nombre in MODELOS_ENSEMBLE}
        serie = serie_por_sku(ventas, sku_id)

        pron_ets, _fallback_ets, _motivo = CANDIDATOS_CON_METADATA["ets"](serie, horizonte)
        pron_tsb, _fallback_tsb, _motivo = CANDIDATOS_CON_METADATA["tsb"](serie, horizonte)
        pron_lgbm = pron_lgbm_futuro.loc[[sku_id]].sort_values("fecha")["unidades_pronosticadas"].to_numpy()

        combinado = combinar_pronosticos({"ets": pron_ets, "tsb": pron_tsb, "lightgbm_global": pron_lgbm}, pesos)
        fechas_futuras = pd.date_range(start=serie.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq="MS")

        filas.append(
            pd.DataFrame(
                {
                    "sku_id": sku_id,
                    "fecha": fechas_futuras,
                    "candidato": NOMBRE_ENSEMBLE,
                    "unidades_pronosticadas": combinado,
                    "peso_ets": pesos["ets"],
                    "peso_tsb": pesos["tsb"],
                    "peso_lightgbm_global": pesos["lightgbm_global"],
                }
            )
        )

    return pd.concat(filas, ignore_index=True)
