"""Genera un dataset sintético de ventas mensuales por SKU.

Simula la extracción que en producción vendría del DWH: una tabla larga
sku_id / fecha / unidades_vendidas. Se usa como sandbox de desarrollo
mientras se define el acceso real a la base de datos (ver spec.md:54).

5 SKUs, 60 meses cada uno (2021-01 a 2025-12), cada uno con un patrón
distinto para estresar el motor de forecast:

- SKU-001: tendencia creciente + estacionalidad marcada
- SKU-002: estacionalidad marcada, sin tendencia
- SKU-003: bajo volumen / demanda intermitente
- SKU-004: nivel + estacionalidad leve, con outliers de promoción y alguna devolución (valor negativo)
- SKU-005: nivel estable con quiebre estructural (discontinuación a mitad de serie)
"""

import numpy as np
import pandas as pd

SEMILLA = 42
FECHA_INICIO = "2021-01-01"
N_MESES = 60

SKUS = [
    {
        "sku_id": "SKU-001",
        "nombre": "Producto con tendencia y estacionalidad",
        "patron": "tendencia_estacional",
    },
    {
        "sku_id": "SKU-002",
        "nombre": "Producto estacional sin tendencia",
        "patron": "estacional_puro",
    },
    {
        "sku_id": "SKU-003",
        "nombre": "Producto de bajo volumen / demanda intermitente",
        "patron": "intermitente",
    },
    {
        "sku_id": "SKU-004",
        "nombre": "Producto con promociones y devoluciones",
        "patron": "outliers_devoluciones",
    },
    {
        "sku_id": "SKU-005",
        "nombre": "Producto discontinuado a mitad de serie",
        "patron": "quiebre_estructural",
    },
]


def _estacionalidad(mes: np.ndarray, amplitud: float, fase: float) -> np.ndarray:
    return amplitud * np.sin(2 * np.pi * (mes % 12) / 12 + fase)


def _serie_tendencia_estacional(t: np.ndarray, mes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nivel_base = 200
    tendencia = 3.5 * t
    estacional = _estacionalidad(mes, amplitud=80, fase=0.0)
    ruido = rng.normal(0, 15, size=t.size)
    return nivel_base + tendencia + estacional + ruido


def _serie_estacional_pura(t: np.ndarray, mes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nivel_base = 150
    estacional = _estacionalidad(mes, amplitud=100, fase=np.pi / 3)
    ruido = rng.normal(0, 12, size=t.size)
    return nivel_base + estacional + ruido


def _serie_intermitente(t: np.ndarray, mes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nivel_base = 8
    ocurre_venta = rng.random(size=t.size) < 0.35
    magnitud = rng.poisson(lam=nivel_base, size=t.size)
    return np.where(ocurre_venta, magnitud, 0)


def _serie_outliers_devoluciones(t: np.ndarray, mes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nivel_base = 120
    estacional = _estacionalidad(mes, amplitud=20, fase=np.pi)
    ruido = rng.normal(0, 10, size=t.size)
    serie = nivel_base + estacional + ruido

    meses_promo = rng.choice(t.size, size=5, replace=False)
    serie[meses_promo] += rng.uniform(150, 300, size=5)

    meses_devolucion = rng.choice(t.size, size=2, replace=False)
    serie[meses_devolucion] -= rng.uniform(120, 220, size=2)

    return serie


def _serie_quiebre_estructural(t: np.ndarray, mes: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    nivel_base = 180
    punto_quiebre = 42
    ruido = rng.normal(0, 10, size=t.size)
    serie = np.where(t < punto_quiebre, nivel_base, 5) + ruido
    return serie


GENERADORES = {
    "tendencia_estacional": _serie_tendencia_estacional,
    "estacional_puro": _serie_estacional_pura,
    "intermitente": _serie_intermitente,
    "outliers_devoluciones": _serie_outliers_devoluciones,
    "quiebre_estructural": _serie_quiebre_estructural,
}


def generar_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEMILLA)
    fechas = pd.date_range(FECHA_INICIO, periods=N_MESES, freq="MS")
    t = np.arange(N_MESES)
    mes = fechas.month.to_numpy() - 1

    filas = []
    for sku in SKUS:
        serie = GENERADORES[sku["patron"]](t, mes, rng)
        if sku["patron"] != "outliers_devoluciones":
            serie = np.clip(serie, 0, None)
        unidades = np.round(serie).astype(int)
        for fecha, cantidad in zip(fechas, unidades):
            filas.append(
                {
                    "sku_id": sku["sku_id"],
                    "fecha": fecha,
                    "unidades_vendidas": cantidad,
                }
            )

    return pd.DataFrame(filas)


def main() -> None:
    dataset = generar_dataset()
    dataset.to_csv("data/synthetic/ventas_historicas.csv", index=False)

    catalogo = pd.DataFrame(SKUS)
    catalogo.to_csv("data/synthetic/skus.csv", index=False)

    print(f"Generadas {len(dataset)} filas para {len(SKUS)} SKUs en data/synthetic/")


if __name__ == "__main__":
    main()
