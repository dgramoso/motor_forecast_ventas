# Graph Report - motor_forecast_ventas  (2026-09-04)

## Corpus Check
- 75 files · ~39,199 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 760 nodes · 1396 edges · 37 communities (32 shown, 5 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 41 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `34ef03e5`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pronosticar_futuro.py
- ensemble_backtest.py
- pronosticar_directo
- pronosticar_seasonal_naive
- construir_dataset_supervisado
- map.md
- backtest_y_predicciones_lightgbm_global
- Especificación de Feature: Motor de Forecast de Ventas por SKU
- test_comparar_modelos.py
- _ajustar_prophet
- comparar_modelos.py
- pronosticar_tsb
- pipeline.py
- evaluar_ensemble_por_sku
- generar_datos_sinteticos.py
- Validación del candidato "ensemble" sobre datos reales (Online Retail II)
- Prueba de concepto: motor de forecast sobre datos reales (Online Retail II)
- Handoff — motor_forecast_ventas
- Handoff — motor_forecast_ventas
- Handoff — motor_forecast_ventas
- Domain Docs
- Issue tracker: Local Markdown
- Agent skills
- Dataset sintético de ventas
- Motor de Forecast de Ventas
- No usar SARIMA como candidato en el pipeline masivo
- 0002-ets-tsb-por-backtest.md
- 0003-ensemble-con-pesos-por-walk-forward-anidado.md
- Especificación de Feature: Reentrenamiento Programado
- Plan de Implementación: Reentrenamiento Programado
- Lista de Tareas: Reentrenamiento Programado
- Triage Labels

## God Nodes (most connected - your core abstractions)
1. `pronosticar_seasonal_naive()` - 28 edges
2. `backtest_y_predicciones_lightgbm_global()` - 20 edges
3. `evaluar_ensemble_por_sku()` - 20 edges
4. `construir_dataset_supervisado()` - 20 edges
5. `pronosticar_futuro_lightgbm_global()` - 20 edges
6. `pronosticar_tsb()` - 18 edges
7. `comparar_modelos_con_ensemble()` - 17 edges
8. `entrenar_lightgbm_global()` - 17 edges
9. `backtest_walk_forward()` - 16 edges
10. `_serie()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `_ajustar_random_forest()` --indirect_call--> `_crear_estimador()`  [INFERRED]
  src/forecast/modelo_random_forest.py → src/forecast/modelo_lightgbm_global.py
- `_ajustar_xgboost()` --indirect_call--> `_crear_estimador()`  [INFERRED]
  src/forecast/modelo_xgboost.py → src/forecast/modelo_lightgbm_global.py
- `ejecutar_pipeline()` --calls--> `cargar_ventas()`  [EXTRACTED]
  src/forecast/pipeline.py → src/datos/cargar_datos.py
- `comparar_modelos()` --calls--> `serie_por_sku()`  [EXTRACTED]
  src/forecast/comparar_modelos.py → src/datos/cargar_datos.py
- `_backtest_y_predicciones_por_candidato()` --calls--> `serie_por_sku()`  [EXTRACTED]
  src/forecast/ensemble_backtest.py → src/datos/cargar_datos.py

## Import Cycles
- None detected.

## Communities (37 total, 5 thin omitted)

### Community 0 - "pronosticar_futuro.py"
Cohesion: 0.07
Nodes (41): cargar_ventas(), DataFrame, Series, Carga el histórico de ventas por SKU. Hoy lee el CSV sintético…, serie_por_sku(), adi(), clasificar_demanda(), cv2() (+33 more)

### Community 1 - "ensemble_backtest.py"
Cohesion: 0.07
Nodes (31): FuncionPronostico, backtest_walk_forward(), DataFrame, Series, Backtest walk-forward (rolling), sin split único ni k-fold — ver spec.md:19…, Backtest walk-forward del candidato LightGBM global — `comparar_modelos_sku`…, Evaluación walk-forward del candidato "ensemble" (ETS + TSB + LightGBM global)…, combinar_pronosticos() (+23 more)

### Community 2 - "pronosticar_directo"
Cohesion: 0.07
Nodes (30): _construir_features(), _elegir_lags(), _filas_utilizables(), pronosticar_directo(), DataFrame, ndarray, Series, Utilidades compartidas por los modelos basados en árboles (XGBoost, Random… (+22 more)

### Community 3 - "pronosticar_seasonal_naive"
Cohesion: 0.05
Nodes (46): NamedTuple, Wrapper compartido de fallback — usado por modelo_ets.py y modelo_prophet.py:…, estimar_tendencia(), pronosticar_seasonal_naive(), ndarray, Series, Benchmark Seasonal Naive + drift condicional — piso de comparación del MVP…, Seasonal naive, con drift automático (pendiente OLS) si la serie tiene… (+38 more)

### Community 4 - "construir_dataset_supervisado"
Cohesion: 0.10
Nodes (27): LGBMRegressor, construir_dataset_supervisado(), construir_features_lightgbm(), DataFrame, Features para el candidato LightGBM global (ver modelo_lightgbm_global.py) —…, Una fila por (sku_id, fecha) con los lags/rolling stats calculados únicamente…, Dataset supervisado para los `horizonte` pasos, estrategia directa (ver…, _columnas_features() (+19 more)

### Community 5 - "map.md"
Cohesion: 0.05
Nodes (28): Answer, Question, Answer, Question, Answer, Question, Answer, Question (+20 more)

### Community 6 - "backtest_y_predicciones_lightgbm_global"
Cohesion: 0.08
Nodes (28): backtest_y_predicciones_lightgbm_global(), _iterar_predicciones_por_sku(), _pronosticar_origen(), DataFrame, ndarray, Corre `_iterar_predicciones_por_sku` UNA sola vez y devuelve tanto la tabla…, Predicciones out-of-sample crudas del walk-forward, por SKU: `{sku_id: (reales,…, Entrena y predice para un origen. Devuelve una tabla indexada por `sku_id` con… (+20 more)

### Community 7 - "Especificación de Feature: Motor de Forecast de Ventas por SKU"
Cohesion: 0.06
Nodes (31): 10. Checklist de Completitud, 1. Resumen Ejecutivo, 2. Contexto y Motivación, 3. Alcance, 4. Historias de Usuario, 5. Requisitos No Funcionales, 6. Casos Borde y Escenarios de Error, 7. Experiencia de Usuario (sin diseño técnico) (+23 more)

### Community 8 - "test_comparar_modelos.py"
Cohesion: 0.11
Nodes (25): comparar_modelos(), comparar_modelos_sku(), DataFrame, Una fila por candidato, con sus métricas agregadas del backtest y su tasa de…, Igual que `comparar_modelos_sku`, para todos los SKUs de `ventas`. Un SKU cuya…, Las unidades vendidas/pronosticadas nunca son negativas — un modelo puede…, _sin_negativos(), _sin_negativos_con_metadata() (+17 more)

### Community 9 - "_ajustar_prophet"
Cohesion: 0.16
Nodes (18): Exception, ajustar_con_fallback(), ndarray, Series, Corre `ajustar(serie, horizonte)`. Si lanza una de `excepciones`, cae a…, _ajustar_modelo_prophet(), _ajustar_prophet(), pronosticar_prophet() (+10 more)

### Community 10 - "comparar_modelos.py"
Cohesion: 0.12
Nodes (24): _ajustar_benchmark(), ndarray, Series, Corre el backtest walk-forward de benchmark vs. modelo por SKU y devuelve la…, El benchmark es el destino del fallback de los demás — nunca cae en fallback él…, _ajustar_ets(), _ajustar_holt_winters(), pronosticar_ets() (+16 more)

### Community 11 - "pronosticar_tsb"
Cohesion: 0.15
Nodes (14): _ajustar_tsb(), pronosticar_tsb(), ndarray, Series, TSB (Teunter-Syntetos-Babai) para demanda intermitente — SKU-003 mostró que…, Mismo contrato que `_ajustar_ets` (forecast, fallback, motivo) para que…, Series, Tests de TSB (src/forecast/modelo_intermitente.py): p_t*z_t con suavizado… (+6 more)

### Community 12 - "pipeline.py"
Cohesion: 0.11
Nodes (25): Path, _agregar(), guardar_corrida(), guardar_pronosticos(), nuevo_run_id(), obtener_pronostico_vigente(), DataFrame, Persistencia de corridas y pronósticos — esquema decidido en .scratch/motor-… (+17 more)

### Community 13 - "evaluar_ensemble_por_sku"
Cohesion: 0.11
Nodes (18): _backtest_y_predicciones_por_candidato(), comparar_modelos_con_ensemble(), evaluar_ensemble(), evaluar_ensemble_por_sku(), DataFrame, ndarray, Corre `backtest_walk_forward` de un candidato con metadata de fallback (`ets` o…, Una fila por SKU con el candidato "ensemble", mismo esquema que… (+10 more)

### Community 14 - "generar_datos_sinteticos.py"
Cohesion: 0.26
Nodes (13): _estacionalidad(), generar_dataset(), generar_dataset_escala(), main(), DataFrame, ndarray, Genera un dataset sintético de ventas mensuales por SKU. Simula la extracción…, Dataset sintético con `n_skus` SKUs, para probar escala (memoria, tiempo de… (+5 more)

### Community 15 - "Validación del candidato "ensemble" sobre datos reales (Online Retail II)"
Cohesion: 0.20
Nodes (9): Artefactos, Corrección de un hallazgo anterior, Cuando el ensemble NO gana, pierde por bastante, Interpretación y recomendación, Muestra, Pendiente, Por clase de demanda, Resultado: el ensemble gana más que cualquier candidato individual (+1 more)

### Community 16 - "Prueba de concepto: motor de forecast sobre datos reales (Online Retail II)"
Cohesion: 0.22
Nodes (8): Artefactos, Corrida final, Dataset, Distribución de candidato ganador (selección por SKU), Fallback en el pronóstico futuro final, por candidato ganador, Hallazgos y correcciones aplicadas, Prueba de concepto: motor de forecast sobre datos reales (Online Retail II), Qué queda pendiente

### Community 17 - "Handoff — motor_forecast_ventas"
Cohesion: 0.25
Nodes (7): Decisiones tomadas en esta sesión, Dónde está el trabajo, Handoff — motor_forecast_ventas, Notas de entorno, Próxima sesión — foco sugerido, Qué falta definir (todos son `[NECESITA CLARIFICACIÓN]` en el spec), Skills sugeridas para continuar

### Community 18 - "Handoff — motor_forecast_ventas"
Cohesion: 0.25
Nodes (7): Dónde está el trabajo, Handoff — motor_forecast_ventas, Notas de entorno, Próxima sesión — foco sugerido, Qué queda pendiente / sin resolver, Qué se hizo en esta sesión (orden cronológico), Skills sugeridas para continuar

### Community 19 - "Handoff — motor_forecast_ventas"
Cohesion: 0.25
Nodes (7): Decisiones que no se leen del código, Dónde está el trabajo, Estado del ensemble sobre datos reales, Handoff — motor_forecast_ventas, Notas de entorno, Qué queda pendiente, Qué se hizo

### Community 20 - "Domain Docs"
Cohesion: 0.33
Nodes (5): Before exploring, read these, Domain Docs, File structure, Flag ADR conflicts, Use the glossary's vocabulary

### Community 21 - "Issue tracker: Local Markdown"
Cohesion: 0.33
Nodes (5): Conventions, Issue tracker: Local Markdown, Wayfinding operations, When a skill says "fetch the relevant ticket", When a skill says "publish to the issue tracker"

### Community 22 - "Agent skills"
Cohesion: 0.33
Nodes (4): Agent skills, Domain docs, Issue tracker, Triage labels

### Community 23 - "Dataset sintético de ventas"
Cohesion: 0.40
Nodes (4): Archivos, Dataset sintético de ventas, Patrones por SKU, Regenerar

### Community 33 - "Especificación de Feature: Reentrenamiento Programado"
Cohesion: 0.06
Nodes (31): 10. Checklist de Completitud, 1. Resumen Ejecutivo, 2. Contexto y Motivación, 3. Alcance, 4. Historias de Usuario, 5. Requisitos No Funcionales, 6. Casos Borde y Escenarios de Error, 7. Experiencia de Usuario (sin diseño técnico) (+23 more)

### Community 34 - "Plan de Implementación: Reentrenamiento Programado"
Cohesion: 0.06
Nodes (30): 10. Guía de Validación Rápida (Quickstart), 11. Registro de Decisiones Técnicas (ADR), 1. Resumen Técnico, 2. Stack Tecnológico, 3. Arquitectura del Sistema, 4. Modelo de Datos (resumen), 5. Contrato de Invocación (reemplaza "Contratos de API" — no aplica REST), 6. Estrategia de Testing (+22 more)

### Community 35 - "Lista de Tareas: Reentrenamiento Programado"
Cohesion: 0.22
Nodes (8): Grupo 0: Fundación 🏗️, Grupo 1: Aislamiento por SKU (Historia 3) 🔨, Grupo 2: Logging y resumen agregado (Historia 2) ⚙️, Grupo Final: Producción 🚀, Leyenda, Lista de Tareas: Reentrenamiento Programado, Métricas de Progreso, Resumen de Paralelización

## Knowledge Gaps
- **153 isolated node(s):** `Question`, `Answer`, `Question`, `Answer`, `Question` (+148 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **5 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `pronosticar_seasonal_naive()` connect `pronosticar_seasonal_naive` to `pronosticar_futuro.py`, `ensemble_backtest.py`, `pronosticar_directo`, `backtest_y_predicciones_lightgbm_global`, `_ajustar_prophet`, `comparar_modelos.py`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `construir_dataset_supervisado()` connect `construir_dataset_supervisado` to `pronosticar_futuro.py`, `ensemble_backtest.py`, `backtest_y_predicciones_lightgbm_global`?**
  _High betweenness centrality (0.029) - this node is a cross-community bridge._
- **Why does `pronosticar_tsb()` connect `pronosticar_tsb` to `ensemble_backtest.py`, `comparar_modelos.py`?**
  _High betweenness centrality (0.027) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `pronosticar_seasonal_naive()` (e.g. with `comparar_modelos.py` and `.test_divergencia_futura_no_afecta_origenes_con_prefijo_comun()`) actually correct?**
  _`pronosticar_seasonal_naive()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Question`, `Answer`, `Question` to the rest of the system?**
  _153 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pronosticar_futuro.py` be split into smaller, more focused modules?**
  _Cohesion score 0.06649616368286446 - nodes in this community are weakly interconnected._
- **Should `ensemble_backtest.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07407407407407407 - nodes in this community are weakly interconnected._