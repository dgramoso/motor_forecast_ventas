Type: task
Status: resolved
Blocked by: 01

## Question

Implementar la función de selección del mejor modelo por SKU, aplicando el criterio decidido en el ticket 01 sobre la tabla comparativa que produce el ticket 03.

Debe devolver, por SKU, cuál modelo quedó seleccionado (benchmark/ETS/TSB) y por qué (las métricas que sustentan la elección), no solo el nombre del ganador.

## Answer

Creado `src/forecast/seleccionar_modelo.py`:
- `seleccionar_mejor_modelo_sku(tabla_comparativa_sku)` — aplica el criterio del ticket 01 (menor WAPE medio, redondeado a 3 decimales; empate por menor |Bias medio|) sobre la salida de `comparar_modelos_sku`. Devuelve candidato + sus tres métricas.
- `seleccionar_mejor_modelo(tabla_comparativa)` — igual, para todos los SKUs.

Verificado sobre el dataset sintético: el candidato `modelo` (router ETS/TSB) le gana al `benchmark` en los 5 SKUs, consistente con la comparación del ticket 03.

