Type: task
Status: resolved

## Question

Implementar la generación del pronóstico futuro real: dado un SKU, su serie histórica completa y un modelo ya ajustado (benchmark, ETS o TSB), producir el pronóstico hacia adelante más allá del último dato disponible, con horizonte de 90 días / 3 meses (spec.md:46).

A diferencia del backtest (que reserva datos para comparar contra lo real), acá no hay "real" contra qué comparar — es el pronóstico que se serviría hoy. Reusar las funciones de pronóstico ya existentes (`pronosticar_seasonal_naive`, `pronosticar_ets`, `pronosticar_tsb`, `pronosticar_modelo`), aplicadas sobre el histórico completo en vez de una ventana de entrenamiento parcial.

No depende de ningún otro ticket — puede arrancar ya (aunque para integrarse al pipeline completo en el ticket 07 va a necesitar el modelo que elija el ticket 04).

## Answer

Creado `src/forecast/pronosticar_futuro.py`:
- `pronosticar_futuro_sku(serie, candidato, horizonte)` — corre la función de pronóstico del candidato (`CANDIDATOS` de `comparar_modelos.py`) sobre todo el histórico, devuelve fecha + unidades pronosticadas para el horizonte (3 meses).
- `pronosticar_futuro(ventas, tabla_comparativa, horizonte)` — para todos los SKUs: selecciona el candidato ganador (reusa `seleccionar_mejor_modelo_sku`) y genera su pronóstico futuro.

Verificado sobre el dataset sintético: los 5 SKUs generan 3 períodos futuros (2026-01 a 2026-03) coherentes con su patrón — incluyendo SKU-005 (quiebre estructural), que pronostica niveles bajos acordes al régimen post-quiebre, no al nivel histórico previo.

