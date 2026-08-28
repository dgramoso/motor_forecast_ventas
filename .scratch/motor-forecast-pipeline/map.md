## Destination

Pipeline end-to-end del motor de forecast funcionando sobre los 5 SKUs
sintéticos: ingesta de datos (fuente simulada, intercambiable) → ejecución
de modelos candidatos por SKU → comparación de resultados vs. benchmark →
selección del mejor modelo por SKU → generación del pronóstico futuro →
persistencia versionada de pronósticos y corridas de modelo.

## Notes

- Dominio: motor_forecast_ventas. Spec de referencia: specs/001-motor-forecast-sku/spec.md
- Código ya construido a reusar, no reescribir: src/datos/cargar_datos.py,
  src/forecast/metricas.py, src/forecast/benchmark.py, src/forecast/modelo_ets.py,
  src/forecast/modelo_intermitente.py, src/forecast/modelo.py, src/forecast/backtest.py
- Override: los tickets Task construyen código real, no solo documentan.
- Fuera de alcance: API, DWH real, trigger de reentrenamiento (ver Out of scope).

## Decisions so far

- [Criterio de selección del mejor modelo por SKU](issues/01-criterio-seleccion-mejor-modelo.md): WAPE medio del backtest, benchmark compite en el mismo ranking (gana por default si nadie le gana), desempate por menor |Bias medio|.
- [Pipeline de ejecución + comparación](issues/03-pipeline-ejecucion-comparacion.md): `src/forecast/comparar_modelos.py` — tabla comparativa benchmark vs. modelo, por SKU o para todos, reusable.
- [Selección del mejor modelo por SKU](issues/04-seleccion-mejor-modelo.md): `src/forecast/seleccionar_modelo.py` — aplica el criterio del ticket 01; en el dataset sintético, `modelo` le gana al `benchmark` en los 5 SKUs.
- [Esquema de persistencia](issues/02-esquema-persistencia.md): parquet plano, append-only, dos archivos (`data/runs/corridas.parquet` y `data/runs/pronosticos.parquet`), `run_id` uuid4 por corrida completa.
- [Generación del pronóstico futuro](issues/05-pronostico-futuro.md): `src/forecast/pronosticar_futuro.py` — ajusta el candidato ganador sobre todo el histórico y pronostica el horizonte (3 meses) hacia adelante.
- [Persistencia](issues/06-persistencia.md): `src/forecast/persistencia.py` — append-only sobre `data/runs/*.parquet` (gitignored, es output generado); `obtener_pronostico_vigente` deriva el vigente por timestamp.
- [Integración end-to-end](issues/07-integracion-end-to-end.md): `src/forecast/pipeline.py` — `ejecutar_pipeline()` y `main()` (`python -m src.forecast.pipeline`) corren todo el flujo sobre los 5 SKUs sintéticos.

## Not yet specified

(vacío — la fog restante se resolvió implícita en el esquema de persistencia:
no se cachea "vigente" aparte, se deriva por timestamp en cada consulta)

## Out of scope

- Exponer el pronóstico como API/servicio (Historia 1 del spec) — el foco
  de este mapa es el pipeline interno, no la capa de consumo externo.
- Conexión al DWH real (spec.md:54, sigue [NECESITA CLARIFICACIÓN]) — este
  pipeline usa la fuente simulada a propósito, intercambiable después.
- Trigger de reentrenamiento (spec.md:51) — decisión de negocio pendiente
  con el cliente, no bloquea que el pipeline corra end-to-end una vez.
- Observabilidad/logging estructurado de éxito/error por corrida
  (spec.md:126-128) — la tabla `corridas.parquet` ya deja métricas y
  timestamp por corrida, suficiente para este pipeline interno; logging
  dedicado es una mejora posterior, no bloquea el destino de este mapa.
