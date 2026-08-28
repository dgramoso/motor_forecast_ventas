Type: grilling
Status: resolved

## Question

¿Qué esquema de persistencia usamos para guardar los pronósticos generados y las "versiones ejecutadas" de los modelos (corridas del pipeline)?

Alternativas a discutir:
- Formato de almacenamiento: CSV/parquet plano vs. SQLite — a esta escala (5 SKUs, ejecuciones esporádicas) probablemente no hace falta una base de datos todavía.
- Qué campos captura una "corrida": run id, timestamp, sku_id, modelo elegido (benchmark/ETS/TSB), parámetros del modelo, métricas de backtest (WAPE/Bias/MAE), y el pronóstico generado (valores por período dentro del horizonte).
- Cómo se versiona: ¿cada corrida es inmutable y se acumulan históricos, o se sobrescribe el último pronóstico vigente por SKU (con el histórico aparte)?
- Relación con la futura persistencia real (¿este esquema debería anticipar cómo se guardaría en el DWH/API cuando eso se defina, o es explícitamente descartable?).

Bloquea el ticket 06 (persistencia de pronósticos y versiones de corrida).

## Answer

Formato: **parquet plano**, dos archivos (nada de SQLite ni DWH todavía —
sobre-ingeniería a esta escala). Versionado: **append-only inmutable**,
cada corrida agrega filas nuevas, nunca pisa una corrida anterior.

- `data/runs/corridas.parquet` — una fila por (run_id, sku_id):
  `run_id`, `timestamp_utc`, `sku_id`, `candidato`, `wape_medio`,
  `bias_medio`, `mae_medio`.
- `data/runs/pronosticos.parquet` — una fila por (run_id, sku_id, período):
  `run_id`, `sku_id`, `fecha`, `unidades_pronosticadas`.

`run_id` es un uuid4, generado por corrida (no por sku) — todos los SKUs
de una misma ejecución del pipeline comparten `run_id`. El "vigente" por
SKU se deriva como la fila de mayor `timestamp_utc` en `corridas.parquet`
para ese `sku_id` — no hay un concepto de "vigente" persistido aparte.

