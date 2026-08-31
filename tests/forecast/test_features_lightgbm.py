"""Tests de features_lightgbm.py: construcción de lags/rolling sin fuga
de información futura, y del dataset supervisado (target + mes del
período objetivo, no del origen)."""

import unittest

import pandas as pd

from src.forecast.features_lightgbm import construir_dataset_supervisado, construir_features_lightgbm


def _ventas(sku_valores: dict[str, list[float]], inicio: str = "2020-01-01") -> pd.DataFrame:
    filas = []
    for sku, valores in sku_valores.items():
        fechas = pd.date_range(inicio, periods=len(valores), freq="MS")
        for fecha, valor in zip(fechas, valores):
            filas.append({"sku_id": sku, "fecha": fecha, "unidades_vendidas": valor})
    return pd.DataFrame(filas)


class TestConstruirFeatures(unittest.TestCase):
    def test_lag_1_es_el_valor_de_origen(self):
        valores = list(range(1, 21))
        ventas = _ventas({"SKU-A": valores})
        features = construir_features_lightgbm(ventas, lags=(1, 2), ventanas_rolling=(3,))

        fecha_origen = pd.Timestamp("2020-01-01") + pd.DateOffset(months=15)
        fila = features[features["fecha"] == fecha_origen].iloc[0]

        self.assertEqual(fila["lag_1"], 16)
        self.assertEqual(fila["lag_2"], 15)

    def test_descarta_filas_sin_historia_suficiente_para_el_lag_mayor(self):
        ventas = _ventas({"SKU-A": list(range(1, 6))})
        features = construir_features_lightgbm(ventas, lags=(1, 12), ventanas_rolling=(3,))

        self.assertTrue(features.empty)

    def test_lags_no_se_mezclan_entre_skus(self):
        ventas = _ventas({"SKU-A": [999] * 5, "SKU-B": [1, 2, 3, 4, 5]})
        features = construir_features_lightgbm(ventas, lags=(1, 2), ventanas_rolling=())

        fila_b = features[features["sku_id"] == "SKU-B"].iloc[-1]
        self.assertEqual(fila_b["lag_1"], 5)
        self.assertEqual(fila_b["lag_2"], 4)


class TestDatasetSupervisado(unittest.TestCase):
    def test_mes_objetivo_es_el_del_periodo_futuro_no_el_del_origen(self):
        ventas = _ventas({"SKU-A": list(range(1, 15))})
        dataset = construir_dataset_supervisado(ventas, horizonte=2, lags=(1,), ventanas_rolling=())

        fila = dataset[
            (dataset["fecha"] == pd.Timestamp("2020-01-01")) & (dataset["paso_horizonte"] == 1)
        ].iloc[0]
        self.assertEqual(fila["mes_objetivo"], 2)  # enero + 1 paso = febrero

    def test_target_es_el_valor_real_del_paso_de_horizonte(self):
        valores = list(range(1, 15))
        ventas = _ventas({"SKU-A": valores})
        dataset = construir_dataset_supervisado(ventas, horizonte=1, lags=(1,), ventanas_rolling=())

        fila = dataset[dataset["fecha"] == pd.Timestamp("2020-02-01")].iloc[0]
        self.assertEqual(fila["target"], 3)

    def test_ultimo_origen_tiene_target_nan(self):
        ventas = _ventas({"SKU-A": list(range(1, 15))})
        dataset = construir_dataset_supervisado(ventas, horizonte=1, lags=(1,), ventanas_rolling=())

        ultima_fila = dataset.loc[dataset["fecha"].idxmax()]
        self.assertTrue(pd.isna(ultima_fila["target"]))

    def test_dataset_apila_multiples_skus(self):
        ventas = _ventas({"SKU-A": list(range(1, 15)), "SKU-B": list(range(100, 114))})
        dataset = construir_dataset_supervisado(ventas, horizonte=1, lags=(1,), ventanas_rolling=())

        self.assertEqual(set(dataset["sku_id"]), {"SKU-A", "SKU-B"})


if __name__ == "__main__":
    unittest.main()
