"""Carga el histórico de ventas por SKU.

Hoy lee el CSV sintético (data/synthetic/ventas_historicas.csv). Cuando
se defina el acceso al DWH real (spec.md:54), esta es la función a
reemplazar — el resto del pipeline consume una Series de pandas indexada
por fecha, sin importar el origen.
"""

import pandas as pd

RUTA_VENTAS = "data/synthetic/ventas_historicas.csv"


def cargar_ventas(ruta: str = RUTA_VENTAS) -> pd.DataFrame:
    df = pd.read_csv(ruta, parse_dates=["fecha"])
    return df.sort_values(["sku_id", "fecha"]).reset_index(drop=True)


def serie_por_sku(df: pd.DataFrame, sku_id: str) -> pd.Series:
    serie = df.loc[df["sku_id"] == sku_id].set_index("fecha")["unidades_vendidas"]
    return serie.asfreq("MS")
