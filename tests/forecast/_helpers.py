"""Fixtures compartidas entre tests de forecast — no es un módulo de
producción, sólo evita duplicar los mismos doubles en varios archivos."""

from src.forecast.comparar_modelos import _ajustar_benchmark
from src.forecast.modelo_ets import _ajustar_ets
from src.forecast.modelo_intermitente import _ajustar_tsb

# Candidatos livianos (sin Prophet/XGBoost/Random Forest) para que los
# tests de comparar_modelos_con_ensemble sean rápidos — ver el mismo
# criterio en test_comparar_modelos_global.py::TestComparacionCombinada.
CANDIDATOS_LIVIANOS = {"benchmark": _ajustar_benchmark, "ets": _ajustar_ets, "tsb": _ajustar_tsb}
