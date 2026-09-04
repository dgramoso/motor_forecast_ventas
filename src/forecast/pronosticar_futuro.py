"""Pronóstico futuro real: a diferencia del backtest, acá no hay "real"
contra qué comparar — es el pronóstico que se serviría hoy. Usa el
candidato seleccionado (ver seleccionar_modelo.py) ajustado sobre todo
el histórico disponible, no una ventana de entrenamiento parcial.
"""

from typing import Iterable

import pandas as pd

from src.datos.cargar_datos import cargar_ventas, serie_por_sku
from src.forecast.benchmark import pronosticar_seasonal_naive
from src.forecast.comparar_modelos import CANDIDATOS_CON_METADATA, HORIZONTE
from src.forecast.comparar_modelos_global import NOMBRE_CANDIDATO as NOMBRE_CANDIDATO_LIGHTGBM_GLOBAL
from src.forecast.diagnostico_demanda import adi, clasificar_demanda, cv2, tasa_de_ceros
from src.forecast.ensemble import combinar_pronosticos
from src.forecast.ensemble_backtest import CANDIDATOS_ENSEMBLE, NOMBRE_ENSEMBLE
from src.forecast.features_lightgbm import LAGS, VENTANAS_ROLLING, construir_dataset_supervisado
from src.forecast.modelo_lightgbm_global import entrenar_lightgbm_global, pronosticar_lightgbm_global
from src.forecast.seleccionar_modelo import seleccionar_mejor_modelo_sku


def pronosticar_futuro_sku(serie: pd.Series, candidato: str, horizonte: int = HORIZONTE) -> pd.DataFrame:
    """El pronóstico servido, más si el candidato elegido cayó en
    fallback al ajustar sobre todo el histórico (ver CONTEXT.md,
    "Fallback") — para saber si lo que se sirve es realmente `candidato`
    o el benchmark disfrazado de tal. Incluye el diagnóstico de demanda
    (ver diagnostico_demanda.py) para trazabilidad: por qué un SKU terminó
    en TSB en vez de ETS se explica por su WAPE en el backtest, no por
    este diagnóstico — pero queda auditable junto al pronóstico."""
    ajustar_con_metadata = CANDIDATOS_CON_METADATA[candidato]
    valores, fallback, _motivo = ajustar_con_metadata(serie, horizonte)

    fechas_futuras = pd.date_range(
        start=serie.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq="MS"
    )
    return pd.DataFrame(
        {
            "fecha": fechas_futuras,
            "unidades_pronosticadas": valores,
            "fallback": fallback,
            "tasa_de_ceros": tasa_de_ceros(serie),
            "adi": adi(serie),
            "cv2": cv2(serie),
            "clase_demanda": clasificar_demanda(serie),
            "observaciones_entrenamiento": len(serie),
            "observaciones_demanda_positiva": int((serie > 0).sum()),
        }
    )


def pronosticar_futuro_lightgbm_global(
    ventas: pd.DataFrame,
    skus: Iterable[str],
    horizonte: int = HORIZONTE,
    lags: tuple[int, ...] = LAGS,
    ventanas_rolling: tuple[int, ...] = VENTANAS_ROLLING,
) -> pd.DataFrame:
    """Sirve el pronóstico de LightGBM global para las SKUs de `skus` que
    lo ganaron en la selección (ver seleccionar_modelo.py) — entrena UNA
    sola vez con el histórico completo de TODAS las SKUs de `ventas` (no
    solo las de `skus`, para que el modelo siga siendo global) y filtra
    el resultado, en vez de reentrenar por SKU.

    Si el ajuste global falla (o alguna SKU de `skus` queda afuera del
    último origen, p.ej. por falta de historia), cae a Seasonal Naive
    para esa SKU — mismo criterio de fallback que el resto de los
    candidatos (ver CONTEXT.md, "Fallback"): `candidato` sigue diciendo
    "lightgbm_global" pero `fallback=True` avisa que en realidad se sirvió
    el benchmark."""
    skus = set(skus)
    try:
        dataset = construir_dataset_supervisado(ventas, horizonte, lags, ventanas_rolling)
        modelos = entrenar_lightgbm_global(dataset, horizonte)
        pronostico_lgbm = {
            sku_id: grupo.sort_values("paso_horizonte")["unidades_pronosticadas"].to_numpy()
            for sku_id, grupo in pronosticar_lightgbm_global(modelos, dataset).groupby("sku_id")
        }
    except (ValueError, LookupError):
        pronostico_lgbm = {}

    filas = []
    for sku_id in skus:
        serie = serie_por_sku(ventas, sku_id)
        if sku_id in pronostico_lgbm:
            valores, fallback = pronostico_lgbm[sku_id], False
        else:
            valores, fallback = pronosticar_seasonal_naive(serie, horizonte), True

        fechas_futuras = pd.date_range(
            start=serie.index[-1] + pd.DateOffset(months=1), periods=horizonte, freq="MS"
        )
        filas.append(
            pd.DataFrame(
                {
                    "sku_id": sku_id,
                    "fecha": fechas_futuras,
                    "candidato": NOMBRE_CANDIDATO_LIGHTGBM_GLOBAL,
                    "unidades_pronosticadas": valores,
                    "fallback": fallback,
                    "tasa_de_ceros": tasa_de_ceros(serie),
                    "adi": adi(serie),
                    "cv2": cv2(serie),
                    "clase_demanda": clasificar_demanda(serie),
                    "observaciones_entrenamiento": len(serie),
                    "observaciones_demanda_positiva": int((serie > 0).sum()),
                }
            )
        )

    return pd.concat(filas, ignore_index=True)


def pronosticar_futuro_ensemble(
    ventas: pd.DataFrame, skus: Iterable[str], tabla_comparativa: pd.DataFrame, horizonte: int = HORIZONTE
) -> pd.DataFrame:
    """Sirve el pronóstico del candidato "ensemble" para las SKUs que lo
    ganaron: ajusta ETS y TSB sobre todo el histórico (igual que
    `pronosticar_futuro_sku`) y los combina con el pronóstico futuro
    global de LightGBM (`pronosticar_futuro_lightgbm_global`, un solo
    entrenamiento para todas las SKUs de `skus`) usando los pesos de
    PRODUCCIÓN que ya calculó `ensemble_backtest.evaluar_ensemble`
    (columnas `peso_*` de la fila `candidato == "ensemble"` de cada SKU
    en `tabla_comparativa`) — no reajusta pesos acá."""
    skus = list(skus)
    pesos_por_sku = tabla_comparativa[tabla_comparativa["candidato"] == NOMBRE_ENSEMBLE].set_index("sku_id")
    pron_lgbm_futuro = pronosticar_futuro_lightgbm_global(ventas, skus, horizonte).set_index("sku_id")

    filas = []
    for sku_id in skus:
        pesos = {nombre: pesos_por_sku.loc[sku_id, f"peso_{nombre}"] for nombre in CANDIDATOS_ENSEMBLE}
        serie = serie_por_sku(ventas, sku_id)

        pron_ets, _fallback, _motivo = CANDIDATOS_CON_METADATA["ets"](serie, horizonte)
        pron_tsb, _fallback, _motivo = CANDIDATOS_CON_METADATA["tsb"](serie, horizonte)
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
                    # El ensemble en sí no cae en fallback (ver
                    # ensemble_backtest.py) — combina candidatos que ya
                    # son defensivos por su cuenta.
                    "fallback": False,
                    "tasa_de_ceros": tasa_de_ceros(serie),
                    "adi": adi(serie),
                    "cv2": cv2(serie),
                    "clase_demanda": clasificar_demanda(serie),
                    "observaciones_entrenamiento": len(serie),
                    "observaciones_demanda_positiva": int((serie > 0).sum()),
                    # Parámetros propios del candidato (ver
                    # seleccionar_modelo.py) — quedan también junto al
                    # pronóstico servido, no sólo en la corrida.
                    **{f"peso_{nombre}": pesos[nombre] for nombre in CANDIDATOS_ENSEMBLE},
                }
            )
        )

    return pd.concat(filas, ignore_index=True)


def pronosticar_futuro(
    ventas: pd.DataFrame, tabla_comparativa: pd.DataFrame, horizonte: int = HORIZONTE
) -> pd.DataFrame:
    """Selecciona el mejor candidato por SKU (ver seleccionar_modelo.py) y
    genera su pronóstico futuro. `tabla_comparativa` es la salida de
    `comparar_modelos` (o `comparar_modelos_con_ensemble`) sobre el
    mismo `ventas`. Los candidatos "globales" (`servidores_globales`) se
    sirven aparte de `pronosticar_futuro_sku`: un solo entrenamiento para
    todas las SKUs que los ganaron, no uno por SKU. Agregar un nuevo
    candidato global es un solo punto de cambio: sumar su entrada acá."""
    servidores_globales = {
        NOMBRE_CANDIDATO_LIGHTGBM_GLOBAL: lambda skus: pronosticar_futuro_lightgbm_global(ventas, skus, horizonte),
        NOMBRE_ENSEMBLE: lambda skus: pronosticar_futuro_ensemble(ventas, skus, tabla_comparativa, horizonte),
    }
    skus_por_candidato_global: dict[str, list] = {nombre: [] for nombre in servidores_globales}

    tablas = []
    for sku_id, tabla_sku in tabla_comparativa.groupby("sku_id"):
        seleccion = seleccionar_mejor_modelo_sku(tabla_sku)
        candidato = seleccion["candidato"]
        if candidato in servidores_globales:
            skus_por_candidato_global[candidato].append(sku_id)
            continue

        serie = serie_por_sku(ventas, sku_id)
        pronostico = pronosticar_futuro_sku(serie, candidato, horizonte)
        pronostico.insert(0, "sku_id", sku_id)
        pronostico.insert(2, "candidato", candidato)
        tablas.append(pronostico)

    for candidato, skus in skus_por_candidato_global.items():
        if skus:
            tablas.append(servidores_globales[candidato](skus))

    return pd.concat(tablas, ignore_index=True)
