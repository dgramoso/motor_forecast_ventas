# Motor de Forecast de Ventas

Motor que genera pronósticos de demanda por SKU, comparando varios métodos de pronóstico entre sí y seleccionando el mejor por SKU en cada corrida.

## Language

**Candidato**:
Un método de pronóstico que compite en el ranking de `comparar_modelos` para un SKU dado (`benchmark`, `ets`, `tsb`, `xgboost`, `prophet`, `random_forest`). Todos compiten en pie de igualdad — el benchmark no es un caso especial, gana por default si nadie le gana. SARIMA no es candidato — ver `docs/adr/0001-no-sarima.md`. ETS y TSB compiten como candidatos independientes, no por una regla fija de intermitencia — ver `docs/adr/0002-ets-tsb-por-backtest.md`.
_Avoid_: modelo (ambiguo — "modelo" también nombra al concepto general de método estadístico; usar "candidato" cuando el contexto es la competencia por SKU).

**Fallback**:
Cuando el ajuste de un candidato falla ante datos degenerados (poca historia, ventana casi constante) y el candidato sustituye su pronóstico por el del benchmark Seasonal Naive. No es un error: es un comportamiento defensivo esperado y auditable.
_Avoid_: error, excepción (esos son la causa; "fallback" es la respuesta).

**Motivo de fallback**:
El string que identifica por qué un candidato cayó en fallback — tipo de excepción capturada más su mensaje (`f"{type(error).__name__}: {error}"`). Se registra para poder auditar la corrida, no para mostrarlo al usuario final.

**Tasa de fallback**:
% de ventanas del backtest walk-forward donde un candidato cayó en fallback, para un SKU dado. Una tasa alta indica que el candidato "ganó" el ranking de WAPE en gran parte gracias al benchmark que lo respalda, no a su propio ajuste.

**SKU sin datos suficientes para comparar**:
Un SKU donde todas las ventanas del backtest walk-forward tuvieron demanda real total cero — el WAPE queda indefinido (`NaN`) para los 6 candidatos por igual, así que no hay una comparación real entre ellos. Distinto de "SKU con histórico insuficiente" (eso lo excluye del backtest desde antes; esto ocurre con histórico suficiente pero demanda real vacía en cada ventana).

**ADI** (Average Demand Interval):
Cantidad de períodos por cada período con demanda positiva, sobre la serie recibida (`len(serie) / períodos_con_demanda_positiva`). `inf` si la serie no tuvo ningún período con demanda positiva. Ver `diagnostico_demanda.py`.

**CV²**:
Coeficiente de variación al cuadrado de la demanda, calculado solo sobre los períodos con demanda positiva. `0.0` (no indefinido) cuando hay menos de 2 observaciones positivas. Ver `diagnostico_demanda.py`.

**Clase de demanda**:
Clasificación del patrón de demanda de un SKU según ADI y CV² (criterio SBC — Syntetos, Boylan y Croston): `sin_demanda`, `regular`, `intermitente`, `erratica` o `lumpy`. Es diagnóstico y trazabilidad, no determina el candidato ganador — eso lo decide el backtest (ver `docs/adr/0002-ets-tsb-por-backtest.md`).
_Avoid_: usar la clase de demanda como si fuera el criterio de selección del modelo — es información auditable, el ganador siempre sale del WAPE del backtest.
