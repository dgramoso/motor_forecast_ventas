# Graph Report - motor_forecast_ventas  (2026-09-02)

## Corpus Check
- 69 files · ~30,378 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 635 nodes · 1215 edges · 34 communities (30 shown, 4 thin omitted)
- Extraction: 97% EXTRACTED · 3% INFERRED · 0% AMBIGUOUS · INFERRED: 38 edges (avg confidence: 0.7)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `abd25db9`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- pronosticar_futuro_lightgbm_global
- comparar_modelos_global.py
- pronosticar_directo
- pronosticar_seasonal_naive
- construir_dataset_supervisado
- map.md
- seleccionar_mejor_modelo_sku
- Especificación de Feature: Motor de Forecast de Ventas por SKU
- comparar_modelos_sku
- _ajustar_prophet
- _ajustar_ets
- pronosticar_tsb
- _ajustar_xgboost
- comparar_modelos.py
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
- test_ensemble.py

## God Nodes (most connected - your core abstractions)
1. `pronosticar_seasonal_naive()` - 28 edges
2. `construir_dataset_supervisado()` - 20 edges
3. `pronosticar_futuro_lightgbm_global()` - 20 edges
4. `backtest_y_predicciones_lightgbm_global()` - 18 edges
5. `pronosticar_tsb()` - 18 edges
6. `entrenar_lightgbm_global()` - 17 edges
7. `backtest_walk_forward()` - 16 edges
8. `_ajustar_ets()` - 16 edges
9. `_serie()` - 16 edges
10. `clasificar_demanda()` - 15 edges

## Surprising Connections (you probably didn't know these)
- `_ajustar_random_forest()` --indirect_call--> `_crear_estimador()`  [INFERRED]
  src/forecast/modelo_random_forest.py → src/forecast/modelo_lightgbm_global.py
- `_ajustar_xgboost()` --indirect_call--> `_crear_estimador()`  [INFERRED]
  src/forecast/modelo_xgboost.py → src/forecast/modelo_lightgbm_global.py
- `pronosticar_futuro()` --calls--> `serie_por_sku()`  [EXTRACTED]
  src/forecast/pronosticar_futuro.py → src/datos/cargar_datos.py
- `pronosticar_futuro_ensemble()` --calls--> `serie_por_sku()`  [EXTRACTED]
  src/forecast/pronosticar_futuro.py → src/datos/cargar_datos.py
- `pronosticar_futuro_lightgbm_global()` --calls--> `serie_por_sku()`  [EXTRACTED]
  src/forecast/pronosticar_futuro.py → src/datos/cargar_datos.py

## Import Cycles
- None detected.

## Communities (34 total, 4 thin omitted)

### Community 0 - "pronosticar_futuro_lightgbm_global"
Cohesion: 0.08
Nodes (33): adi(), clasificar_demanda(), cv2(), Series, Diagnóstico del patrón de demanda — ADI, CV² y clasificación SBC (Syntetos,…, % de períodos con demanda cero. `nan` si la serie está vacía., Average demand interval: cantidad de períodos por cada período con demanda…, Coeficiente de variación al cuadrado de la demanda, sobre los períodos con… (+25 more)

### Community 1 - "comparar_modelos_global.py"
Cohesion: 0.07
Nodes (36): FuncionPronostico, backtest_walk_forward(), DataFrame, Series, Backtest walk-forward (rolling), sin split único ni k-fold — ver spec.md:19…, _ajustar_benchmark(), backtest_y_predicciones_lightgbm_global(), _iterar_predicciones_por_sku() (+28 more)

### Community 2 - "pronosticar_directo"
Cohesion: 0.07
Nodes (30): _construir_features(), _elegir_lags(), _filas_utilizables(), pronosticar_directo(), DataFrame, ndarray, Series, Utilidades compartidas por los modelos basados en árboles (XGBoost, Random… (+22 more)

### Community 3 - "pronosticar_seasonal_naive"
Cohesion: 0.09
Nodes (32): NamedTuple, Wrapper compartido de fallback — usado por modelo_ets.py y modelo_prophet.py:…, estimar_tendencia(), pronosticar_seasonal_naive(), ndarray, Series, Benchmark Seasonal Naive + drift condicional — piso de comparación del MVP…, Seasonal naive, con drift automático (pendiente OLS) si la serie tiene… (+24 more)

### Community 4 - "construir_dataset_supervisado"
Cohesion: 0.10
Nodes (27): LGBMRegressor, construir_dataset_supervisado(), construir_features_lightgbm(), DataFrame, Features para el candidato LightGBM global (ver modelo_lightgbm_global.py) —…, Una fila por (sku_id, fecha) con los lags/rolling stats calculados únicamente…, Dataset supervisado para los `horizonte` pasos, estrategia directa (ver…, _columnas_features() (+19 more)

### Community 5 - "map.md"
Cohesion: 0.05
Nodes (28): Answer, Question, Answer, Question, Answer, Question, Answer, Question (+20 more)

### Community 6 - "seleccionar_mejor_modelo_sku"
Cohesion: 0.20
Nodes (11): DataFrame, Selección del mejor modelo por SKU — criterio decidido en .scratch/motor-…, `tabla_comparativa_sku` es la salida de `comparar_modelos_sku` para un SKU. Si…, `tabla_comparativa` es la salida de `comparar_modelos` (todos los SKUs)., seleccionar_mejor_modelo(), seleccionar_mejor_modelo_sku(), DataFrame, Tests de seleccionar_modelo.py: criterio de selección (menor WAPE medio,… (+3 more)

### Community 7 - "Especificación de Feature: Motor de Forecast de Ventas por SKU"
Cohesion: 0.06
Nodes (31): 10. Checklist de Completitud, 1. Resumen Ejecutivo, 2. Contexto y Motivación, 3. Alcance, 4. Historias de Usuario, 5. Requisitos No Funcionales, 6. Casos Borde y Escenarios de Error, 7. Experiencia de Usuario (sin diseño técnico) (+23 more)

### Community 8 - "comparar_modelos_sku"
Cohesion: 0.16
Nodes (16): comparar_modelos_sku(), DataFrame, Las unidades vendidas/pronosticadas nunca son negativas — un modelo puede…, Una fila por candidato, con sus métricas agregadas del backtest y su tasa de…, _sin_negativos(), _sin_negativos_con_metadata(), ndarray, Series (+8 more)

### Community 9 - "_ajustar_prophet"
Cohesion: 0.16
Nodes (18): Exception, ajustar_con_fallback(), ndarray, Series, Corre `ajustar(serie, horizonte)`. Si lanza una de `excepciones`, cae a…, _ajustar_modelo_prophet(), _ajustar_prophet(), pronosticar_prophet() (+10 more)

### Community 10 - "_ajustar_ets"
Cohesion: 0.16
Nodes (18): _ajustar_ets(), _ajustar_holt_winters(), pronosticar_ets(), ndarray, Series, Modelo estadístico clásico por SKU: Holt-Winters (ETS), aditivo. Trend y…, Ajusta ETS y devuelve (forecast, fallback, motivo_fallback). `fallback=True`…, patch (+10 more)

### Community 11 - "pronosticar_tsb"
Cohesion: 0.15
Nodes (14): _ajustar_tsb(), pronosticar_tsb(), ndarray, Series, TSB (Teunter-Syntetos-Babai) para demanda intermitente — SKU-003 mostró que…, Mismo contrato que `_ajustar_ets` (forecast, fallback, motivo) para que…, Series, Tests de TSB (src/forecast/modelo_intermitente.py): p_t*z_t con suavizado… (+6 more)

### Community 12 - "_ajustar_xgboost"
Cohesion: 0.14
Nodes (14): _ajustar_xgboost(), pronosticar_xgboost(), ndarray, Series, XGBoost — candidato independiente en `comparar_modelos.CANDIDATOS`. Ver…, Ajusta XGBoost y devuelve (forecast, fallback, motivo_fallback)., _EstimadorConstante, patch (+6 more)

### Community 13 - "comparar_modelos.py"
Cohesion: 0.07
Nodes (40): Path, cargar_ventas(), DataFrame, Series, Carga el histórico de ventas por SKU. Hoy lee el CSV sintético…, serie_por_sku(), comparar_modelos(), Corre el backtest walk-forward de benchmark vs. modelo por SKU y devuelve la… (+32 more)

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
Cohesion: 0.40
Nodes (3): Agent skills, Domain docs, Issue tracker

### Community 23 - "Dataset sintético de ventas"
Cohesion: 0.40
Nodes (4): Archivos, Dataset sintético de ventas, Patrones por SKU, Regenerar

### Community 33 - "test_ensemble.py"
Cohesion: 0.13
Nodes (13): combinar_pronosticos(), optimizar_pesos(), ndarray, Ensemble de pronósticos por SKU — combinación lineal por pesos de predicciones…, ŷ = Σ w_i · ŷ_i. Exige que `pesos` cubra exactamente los mismos modelos que…, Pesos que minimizan el WAPE medio sobre las ventanas out-of-sample ya evaluadas…, DataFrame, Tests de ensemble.py: combinación por pesos fijos y ajuste de pesos óptimos… (+5 more)

## Knowledge Gaps
- **97 isolated node(s):** `Question`, `Answer`, `Question`, `Answer`, `Question` (+92 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `pronosticar_seasonal_naive()` connect `pronosticar_seasonal_naive` to `pronosticar_futuro_lightgbm_global`, `comparar_modelos_global.py`, `pronosticar_directo`, `_ajustar_prophet`, `_ajustar_xgboost`, `comparar_modelos.py`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Why does `construir_dataset_supervisado()` connect `construir_dataset_supervisado` to `pronosticar_futuro_lightgbm_global`, `comparar_modelos_global.py`, `comparar_modelos.py`?**
  _High betweenness centrality (0.038) - this node is a cross-community bridge._
- **Why does `pronosticar_tsb()` connect `pronosticar_tsb` to `test_ensemble.py`, `comparar_modelos.py`?**
  _High betweenness centrality (0.034) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `pronosticar_seasonal_naive()` (e.g. with `comparar_modelos.py` and `.test_divergencia_futura_no_afecta_origenes_con_prefijo_comun()`) actually correct?**
  _`pronosticar_seasonal_naive()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `pronosticar_tsb()` (e.g. with `comparar_modelos.py` and `.test_ajusta_pesos_con_predicciones_reales_del_backtest()`) actually correct?**
  _`pronosticar_tsb()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Question`, `Answer`, `Question` to the rest of the system?**
  _97 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `pronosticar_futuro_lightgbm_global` be split into smaller, more focused modules?**
  _Cohesion score 0.07743496672716274 - nodes in this community are weakly interconnected._