# Motor de Forecast de Ventas

Motor que genera pronósticos de demanda por SKU, comparando varios métodos de pronóstico entre sí y seleccionando el mejor por SKU en cada corrida.

## Language

**Candidato**:
Un método de pronóstico que compite en el ranking de `comparar_modelos` para un SKU dado (`benchmark`, `ets_tsb`, `xgboost`, `prophet`, `random_forest`). Todos compiten en pie de igualdad — el benchmark no es un caso especial, gana por default si nadie le gana. SARIMA no es candidato — ver `docs/adr/0001-no-sarima.md`.
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
