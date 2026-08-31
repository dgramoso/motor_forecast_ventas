"""Features para el candidato LightGBM global (ver modelo_lightgbm_global.py)
— conjunto chico a propósito (sección 7 del pedido): lags, un par de
rolling stats y el mes calendario del período a pronosticar. Nada de
variables futuras (precio, promoción, stock) todavía: eso requeriría
garantizar que están disponibles en el momento real del forecast, y por
ahora ni siquiera existen en el dataset (ver sección 9).

Vectorizado por `groupby` de pandas (`shift`/`rolling` a nivel de grupo),
sin loop de Python por SKU — con 50.000 series, un loop de Python por
serie sería el cuello de botella (sección 18 del pedido). La misma
convención de índice que ya usa `_modelo_arboles._construir_features`
para XGBoost/Random Forest: `lag_1` es el último valor conocido (el
propio origen), `lag_2` el anterior, etc. — no el lag "puro" respecto al
target, que depende del paso de horizonte (ver `construir_dataset_supervisado`).

Asume que `ventas` ya tiene una fila por SKU y por período, sin huecos
(mismo supuesto que `serie_por_sku.asfreq("MS")` en el resto del
proyecto) — el chequeo de esa calidad de dato es responsabilidad de un
paso previo ("Data Quality Checks" en el diagrama del pedido), no de este
módulo.
"""

import pandas as pd

LAGS = (1, 2, 3, 12)
VENTANAS_ROLLING = (3,)


def construir_features_lightgbm(
    ventas: pd.DataFrame,
    lags: tuple[int, ...] = LAGS,
    ventanas_rolling: tuple[int, ...] = VENTANAS_ROLLING,
) -> pd.DataFrame:
    """Una fila por (sku_id, fecha) con los lags/rolling stats calculados
    únicamente con datos hasta esa fecha inclusive — nunca con el futuro.
    Filas sin suficiente historia para todos los lags/rolling quedan
    afuera (NaN en algún lag o rolling)."""
    datos = ventas.sort_values(["sku_id", "fecha"]).reset_index(drop=True)
    agrupado = datos.groupby("sku_id")["unidades_vendidas"]

    features = datos[["sku_id", "fecha"]].copy()
    for lag in lags:
        features[f"lag_{lag}"] = agrupado.shift(lag - 1)
    for ventana in ventanas_rolling:
        features[f"rolling_mean_{ventana}"] = agrupado.rolling(ventana).mean().reset_index(level=0, drop=True)
        features[f"rolling_std_{ventana}"] = agrupado.rolling(ventana).std().reset_index(level=0, drop=True)

    columnas_requeridas = [c for c in features.columns if c.startswith(("lag_", "rolling_"))]
    return features.dropna(subset=columnas_requeridas)


def construir_dataset_supervisado(
    ventas: pd.DataFrame,
    horizonte: int,
    lags: tuple[int, ...] = LAGS,
    ventanas_rolling: tuple[int, ...] = VENTANAS_ROLLING,
) -> pd.DataFrame:
    """Dataset supervisado para los `horizonte` pasos, estrategia directa
    (ver `_modelo_arboles.py`): una fila por (sku_id, fecha_origen,
    paso_horizonte), con las features de `construir_features_lightgbm`
    más `mes_objetivo` (el mes calendario de la fecha a pronosticar,
    `fecha + paso_horizonte` meses — no el mes del origen, para que el
    modelo aprenda la estacionalidad del período que realmente pronostica)
    y `target` (el valor real de esa fecha futura).

    `target` queda `NaN` cuando esa fecha futura todavía no ocurrió en
    `ventas` — son las filas del último origen disponible de cada SKU,
    exactamente las que `pronosticar_lightgbm_global` necesita para
    predecir. `entrenar_lightgbm_global` descarta esas filas al ajustar,
    no hace falta filtrarlas acá."""
    datos = ventas.sort_values(["sku_id", "fecha"]).reset_index(drop=True)
    features = construir_features_lightgbm(datos, lags, ventanas_rolling)
    objetivo_por_paso = {
        paso: datos.groupby("sku_id")["unidades_vendidas"].shift(-paso) for paso in range(1, horizonte + 1)
    }

    bloques = []
    for paso in range(1, horizonte + 1):
        bloque = features.copy()
        bloque["paso_horizonte"] = paso
        bloque["mes_objetivo"] = (bloque["fecha"] + pd.DateOffset(months=paso)).dt.month
        bloque["target"] = objetivo_por_paso[paso].reindex(bloque.index)
        bloques.append(bloque)

    return pd.concat(bloques, ignore_index=True)
