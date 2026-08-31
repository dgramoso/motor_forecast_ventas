# Prueba de concepto: motor de forecast sobre datos reales (Online Retail II)

Antecedente de la validación del motor de forecast contra un dataset de
retail real, como complemento a los tests unitarios (que usan dobles
rápidos) y al smoke test con AirPassengers (una sola serie). Objetivo:
exponer el pipeline a múltiples SKUs reales, con la intermitencia,
discontinuidad e historia dispar de un catálogo real.

## Dataset

[Online Retail II (UCI)](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — transacciones de una tienda online del Reino Unido, dic-2009 a dic-2011 (25 meses), 1.067.371 líneas.

Filtrado: se excluyen códigos que no son productos reales (`POST`, `D`, `M`, `DOT`, `BANK CHARGES`, `CRUK`, `C2`, `AMAZONFEE` — franqueo, descuentos, ajustes administrativos), y se agrega a demanda mensual por SKU.

| Filtro | SKUs |
|---|---|
| Catálogo completo (sin códigos no-producto) | 5.297 |
| Con ≥18 observaciones (mínimo del backtest: `ventana_minima=15` + `horizonte=3`) | 1.861 |
| ...con volumen total positivo (no son devoluciones netas) | **1.852** ← corrida final |
| Con historial denso, sin huecos, en los 25 meses | 702 |

La brecha entre 5.297 y 1.852 es en sí un hallazgo: ~65% del catálogo tiene menos de 18 meses de ventas (productos discontinuados a mitad de rango, lanzados tarde, o esporádicos). Un motor de forecast de producción debe asumir que la mayoría de un catálogo real no tiene historia suficiente para un modelo estadístico serio.

## Hallazgos y correcciones aplicadas

Tres bugs reales expuestos por los datos (no por los tests unitarios, que usan series sintéticas bien comportadas):

1. **Pronósticos negativos** (commit `e7c6ede`): un candidato con tendencia decreciente fuerte y poca historia podía extrapolar por debajo de cero — cantidad física inválida. Corrección: `comparar_modelos._sin_negativos` / `_sin_negativos_con_metadata` clipean a `max(forecast, 0)` para los 6 candidatos, una sola vez.
2. **XGBoost/Random Forest en fallback constante** (commit `e7c6ede`): con poca historia, el lag=12 (y su media móvil) consumía toda la historia disponible antes de dejar una fila utilizable, forzando fallback casi siempre. Corrección: `_modelo_arboles._elegir_lags` escala a un set de lags cortos (1,2,3, sin lag=12) cuando el set completo no alcanzaría `MIN_FILAS_ENTRENAMIENTO`.
3. **Discontinuidad en el umbral de escalada de lags** (commit `8fae494`): el criterio original aproximaba "historia suficiente para lags completos" como `largo_serie >= 2*PERIODO_ESTACIONAL` (24), pero la aritmética real (con `MIN_FILAS_ENTRENAMIENTO=12`, `horizonte=3`) exige `largo_serie >= 26`. Series de exactamente 24-25 observaciones —como el pronóstico futuro final sobre el histórico completo de varios SKUs de este dataset— escalaban a lags completos y fallaban su propia precondición, forzando un fallback evitable. Corrección: `_filas_utilizables` calcula la precondición real en vez de aproximarla por largo de serie.

Los tres bugs eran invisibles en el backtest (las ventanas del walk-forward, acotadas a `largo_serie - horizonte`, nunca llegan al largo completo de la serie) y sólo se manifestaban en el pronóstico futuro final (que sí entrena con el histórico completo) — ver discusión completa en el hilo de esta sesión.

## Corrida final

1.852 SKUs, `horizonte=3`, `ventana_minima=15`. Candidatos: `benchmark`, `ets_tsb`, `xgboost`, `random_forest` (se excluyen `sarima` y `prophet` por costo — ~59h estimadas para los 6 candidatos sobre el catálogo completo vs. ~74min para los 4 rápidos; quedó fuera de esta corrida, no descartado). Tiempo real: 74 minutos.

### Distribución de candidato ganador (selección por SKU)

| Candidato | SKUs | % |
|---|---|---|
| `benchmark` | 910 | 49% |
| `ets_tsb` | 647 | 35% |
| `random_forest` | 214 | 12% |
| `xgboost` | 81 | 4% |
| *(sin datos suficientes para comparar)* | 70 | 4% |

Que el benchmark gane casi la mitad de los SKUs no es una derrota de los candidatos más sofisticados: es la validación empírica de la decisión de diseño (`.scratch/motor-forecast-pipeline/issues/01-criterio-seleccion-mejor-modelo.md`) de hacerlo competir en igualdad de condiciones en vez de tratarlo como un piso a superar. Con 18-25 meses de historia y catálogos de retail intermitentes/ruidosos, un modelo simple con buen ajuste estacional generaliza mejor que uno complejo sobreajustado a poca historia, en una fracción relevante de los casos.

### Fallback en el pronóstico futuro final, por candidato ganador

| Candidato | Tasa de fallback |
|---|---|
| `benchmark` | 0% (nunca cae en fallback — es el destino del fallback de los demás) |
| `ets_tsb` | 4,0% |
| `random_forest` | 3,3% |
| `xgboost` | 11,1% |

565 de 5.556 pronósticos futuros (10,2%) quedaron en exactamente 0,00 tras el clip de negativos.

## Artefactos

- [`online_retail_ii_selecciones.csv`](online_retail_ii_selecciones.csv): una fila por SKU con el candidato ganador, sus métricas de backtest y el marcador `sin_datos_suficientes`.
- Dataset de origen y scripts de la corrida: quedaron en el scratchpad de la sesión (no versionados — son reproducibles a partir del dataset público de UCI y `src/forecast/comparar_modelos.py`), no en este repo.

## Qué queda pendiente

- Correr `sarima`/`prophet` sobre una muestra más chica para ver si desplazan el reparto de ganadores (no se hizo en esta corrida, por costo).
- El tratamiento de devoluciones netas en el histórico de entrada (demanda mensual negativa) sigue sin definición explícita — ver `specs/001-motor-forecast-sku/spec.md`, sección 6.
