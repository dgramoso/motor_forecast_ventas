"""Fixtures compartidas entre tests de forecast — no es un módulo de
producción, sólo evita duplicar los mismos doubles en varios archivos."""

import numpy as np
import pandas as pd

from src.forecast.comparar_modelos import _ajustar_benchmark
from src.forecast.modelo_ets import _ajustar_ets
from src.forecast.modelo_intermitente import _ajustar_tsb

# Candidatos livianos (sin Prophet/XGBoost/Random Forest) para que los
# tests de comparar_modelos_con_ensemble sean rápidos — ver el mismo
# criterio en test_comparar_modelos_global.py::TestComparacionCombinada.
CANDIDATOS_LIVIANOS = {"benchmark": _ajustar_benchmark, "ets": _ajustar_ets, "tsb": _ajustar_tsb}

# Candidato de prueba para simular una excepción no contemplada por el
# fallback existente (ver _ajuste_con_fallback.py: sólo atrapa las
# excepciones que cada modelo declara conocidas) — usado por los tests
# de aislamiento por SKU (specs/002-reentrenamiento-programado). Rompe
# solo si la serie contiene el valor centinela, para poder simular que
# UN SKU puntual rompe sin que el candidato necesite conocer su sku_id.
VALOR_CENTINELA_DE_FALLA = 12345.0


def ajustar_con_falla_para_valor_centinela(serie: pd.Series, horizonte: int) -> tuple[np.ndarray, bool, None]:
    if (serie == VALOR_CENTINELA_DE_FALLA).any():
        raise RuntimeError("fallo simulado no contemplado por el fallback")
    return np.full(horizonte, serie.iloc[-1] if len(serie) else 0.0), False, None
