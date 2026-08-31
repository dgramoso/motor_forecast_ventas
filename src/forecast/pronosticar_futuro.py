"""Pronóstico futuro real: a diferencia del backtest, acá no hay "real"
contra qué comparar — es el pronóstico que se serviría hoy. Usa el
candidato seleccionado (ver seleccionar_modelo.py) ajustado sobre todo
el histórico disponible, no una ventana de entrenamiento parcial.
"""

import pandas as pd

from src.datos.cargar_datos import cargar_ventas, serie_por_sku
from src.forecast.comparar_modelos import CANDIDATOS_CON_METADATA, HORIZONTE
from src.forecast.diagnostico_demanda import adi, clasificar_demanda, cv2, tasa_de_ceros
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


def pronosticar_futuro(
    ventas: pd.DataFrame, tabla_comparativa: pd.DataFrame, horizonte: int = HORIZONTE
) -> pd.DataFrame:
    """Selecciona el mejor candidato por SKU (ver seleccionar_modelo.py) y
    genera su pronóstico futuro. `tabla_comparativa` es la salida de
    `comparar_modelos` sobre el mismo `ventas`.
    """
    tablas = []
    for sku_id, tabla_sku in tabla_comparativa.groupby("sku_id"):
        seleccion = seleccionar_mejor_modelo_sku(tabla_sku)
        serie = serie_por_sku(ventas, sku_id)

        pronostico = pronosticar_futuro_sku(serie, seleccion["candidato"], horizonte)
        pronostico.insert(0, "sku_id", sku_id)
        pronostico.insert(2, "candidato", seleccion["candidato"])
        tablas.append(pronostico)

    return pd.concat(tablas, ignore_index=True)
