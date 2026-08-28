Type: task
Status: resolved

## Question

Refactorizar `src/forecast/ejecutar_backtest_benchmark.py` (hoy un script de solo-print) en una función reusable que:

- Corra el benchmark Seasonal Naive y los modelos candidatos (ETS vía `src/forecast/modelo_ets.py`, TSB vía `src/forecast/modelo_intermitente.py`, o el router `src/forecast/modelo.py`) sobre un SKU dado, usando `src/forecast/backtest.py`.
- Devuelva la tabla comparativa (WAPE/Bias/MAE por candidato) como un objeto de datos (DataFrame o similar), no solo la imprima.
- Sirva tanto para un SKU individual como para el conjunto completo (los 5 SKUs sintéticos), reusando `src/datos/cargar_datos.py`.

No depende de ningún otro ticket — puede arrancar ya.

## Answer

Creado `src/forecast/comparar_modelos.py`:
- `comparar_modelos_sku(serie, horizonte, ventana_minima)` — corre el backtest walk-forward para `benchmark` y `modelo` (el router de `src/forecast/modelo.py`) sobre una serie, devuelve un DataFrame con una fila por candidato (wape_medio, bias_medio, mae_medio, wape_indefinido, n_ventanas).
- `comparar_modelos(ventas, horizonte, ventana_minima)` — igual, para todos los SKUs de un DataFrame de ventas (formato largo, con `sku_id`).

`src/forecast/ejecutar_backtest_benchmark.py` quedó como wrapper de CLI de una función que solo imprime la tabla. Verificado: mismos números que antes del refactor.

