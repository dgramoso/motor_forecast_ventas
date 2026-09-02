# Validación del candidato "ensemble" sobre datos reales (Online Retail II)

Continuación de `online_retail_ii_prueba_de_concepto.md`: valida si el
candidato `ensemble` (ETS + TSB + LightGBM global, pesos por walk-forward
anidado — ver `src/forecast/ensemble_backtest.py`) le gana en la práctica
a los candidatos individuales, sobre datos reales con ruido e
intermitencia, no solo sobre el dataset sintético usado en los tests.

## Muestra

180 SKUs (de los 1.852 elegibles de la POC anterior), estratificados
proporcionalmente por `clase_demanda` (ADI/CV², ver
`diagnostico_demanda.py`) — no el catálogo completo, por el costo de
correr 7 candidatos por SKU (14 min sobre 180 SKUs).

| Clase de demanda | Catálogo completo | Muestra |
|---|---|---|
| erratica | 1.175 | 114 |
| regular | 485 | 47 |
| lumpy | 152 | 15 |
| intermitente | 40 | 4 |

`horizonte=3`, `ventana_minima=15` — igual que la POC anterior (dataset
de 25 meses). Candidatos: `benchmark`, `ets`, `tsb`, `xgboost`,
`random_forest`, `lightgbm_global`, `ensemble`. Sin `prophet` (costoso,
ya excluido en la POC anterior) ni `sarima` (descartado, ver
`docs/adr/0001-no-sarima.md`).

## Resultado: el ensemble gana más que cualquier candidato individual

| Candidato | SKUs | % |
|---|---|---|
| `ensemble` | 76 | 42,2% |
| `lightgbm_global` | 27 | 15,0% |
| `ets` | 24 | 13,3% |
| `tsb` | 20 | 11,1% |
| `benchmark` | 16 | 8,9% |
| `random_forest` | 13 | 7,2% |
| `xgboost` | 4 | 2,2% |

A diferencia de la POC anterior (donde el benchmark ganaba el 49% del
catálogo completo sin ensemble), acá el ensemble desplaza claramente al
resto — casi 3 veces más ganador que el segundo candidato.

### Por clase de demanda

| Clase | % ensemble ganador |
|---|---|
| erratica | 43,0% |
| regular | 42,6% |
| lumpy | 46,7% |
| **intermitente** | **0,0%** (0/4 — ganó `tsb` en 2, `ets` en 2) |

El ensemble no le gana a TSB en demanda intermitente — esperable: TSB
está diseñado específicamente para eso, y promediarlo con ETS/LightGBM
(que no lo están) diluye su ventaja en vez de sumarla. La muestra de
intermitentes es chica (4 SKUs, el catálogo completo tiene solo 40) —
no alcanza para una conclusión firme, pero la dirección es coherente
con la intuición del método.

## Cuando el ensemble NO gana, pierde por bastante

De los 96 SKUs donde el ensemble no ganó pero sí tuvo WAPE definido:

- Brecha relativa media (ensemble vs. ganador real): **+99,0%** de WAPE
- Brecha relativa mediana: **+60,4%** de WAPE

El ensemble no es "casi tan bueno como el mejor" cuando pierde — pierde
con margen considerable. Esto es consistente con la naturaleza de una
combinación lineal: cuando un especialista (p.ej. TSB en demanda
intermitente) le acierta bien a la estructura de una serie, promediarlo
con modelos que no la capturan empeora el resultado en vez de
mejorarlo. El ensemble gana en promedio (42% de las veces, con margen)
pero no es un "seguro" contra malos pronósticos SKU por SKU.

## Interpretación y recomendación

El ensemble justifica su complejidad: en esta muestra, es el candidato
individual con mayor tasa de victorias, por un margen amplio. La
metodología de walk-forward anidado (que evita que se compare en
ventaja injusta, ver `ensemble_backtest.py`) parece estar dando una
comparación honesta y el resultado sigue siendo favorable.

No hay evidencia todavía de que convenga tratarlo como un reemplazo
universal — en demanda intermitente específicamente, sigue siendo mejor
dejar competir a TSB solo. El diseño actual (todos compiten por WAPE,
sin reglas fijas) ya maneja esto correctamente sin intervención
adicional.

## Pendiente

- Correr sobre una muestra más grande de SKUs intermitentes
  específicamente (el catálogo completo solo tiene 40, la muestra actual
  4) para confirmar si el patrón "ensemble pierde en intermitentes" se
  sostiene con más casos.
- Considerar correr sobre el catálogo completo (1.852 SKUs) si se
  necesita el dato exacto de distribución para producción — la muestra
  de 180 alcanza para validar la decisión de diseño, no para reportar
  un número de producción.
- Los scripts de descarga/preparación de Online Retail II y de esta
  validación quedaron en el scratchpad de la sesión (no versionados) —
  igual que en la POC anterior, son reproducibles a partir del dataset
  público de UCI y `src/forecast/ensemble_backtest.py`.

## Artefactos

- [`ensemble_selecciones_muestra.csv`](ensemble_selecciones_muestra.csv):
  candidato ganador por SKU de la muestra, con sus métricas de backtest.
- [`ensemble_comparacion_vs_ganador.csv`](ensemble_comparacion_vs_ganador.csv):
  WAPE del ensemble vs. WAPE del ganador real, por SKU — la base de la
  brecha relativa reportada arriba.
