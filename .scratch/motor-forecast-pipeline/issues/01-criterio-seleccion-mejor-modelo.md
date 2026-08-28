Type: grilling
Status: resolved

## Question

¿Con qué criterio se elige el "mejor modelo" por SKU a partir de los resultados del backtest walk-forward (benchmark vs. candidatos)?

Alternativas a discutir:
- WAPE medio del backtest (más simple, pero sensible a ventanas atípicas — ver el caso de SKU-005 con quiebre estructural).
- % de ventanas donde el candidato le gana al benchmark en WAPE (más robusto a outliers de ventana, ya calculado en `src/forecast/analizar_tasa_de_exito.py`).
- Combinación con Bias (no solo minimizar WAPE, sino exigir que el Bias no empeore vs. benchmark).
- Qué pasa si ningún candidato le gana al benchmark de forma consistente — ¿el "mejor modelo" puede ser el propio benchmark?
- Desempate cuando dos candidatos quedan parejos.

Bloquea el ticket 04 (selección del mejor modelo por SKU).

## Answer

Criterio: **WAPE medio del backtest walk-forward**, calculado sobre las
mismas ventanas que ya excluyen los casos indefinidos (demanda real cero,
ver `src/forecast/metricas.py`).

- El benchmark participa en el mismo ranking que los candidatos (ETS/TSB
  vía `src/forecast/modelo.py`) — no es un caso especial aparte. Por
  construcción, si ningún candidato le gana al benchmark, el benchmark
  queda seleccionado como mejor modelo (es un modelo válido de pronóstico,
  no solo un piso de comparación).
- Selección: `mejor_modelo = argmin(WAPE_medio)` entre {benchmark, modelo}.
- Desempate (WAPE medio igual, redondeado a 3 decimales): gana el que
  tenga menor `|Bias_medio|`.
