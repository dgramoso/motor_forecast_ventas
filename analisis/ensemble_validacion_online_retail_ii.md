# Validación del candidato "ensemble" sobre datos reales (Online Retail II)

Continuación de `online_retail_ii_prueba_de_concepto.md`: valida si el
candidato `ensemble` (ETS + TSB + LightGBM global, pesos por walk-forward
anidado — ver `src/forecast/ensemble_backtest.py`) le gana en la práctica
a los candidatos individuales, sobre datos reales con ruido e
intermitencia, no solo sobre el dataset sintético usado en los tests.

## Muestra

216 SKUs de los 1.852 elegibles de la POC anterior: una muestra de 180
estratificada proporcionalmente por `clase_demanda` (ADI/CV², ver
`diagnostico_demanda.py`), más los 36 SKUs intermitentes del catálogo
que no habían caído en esa muestra — con solo 4 intermitentes no
alcanzaba para decir nada de esa clase (ver "Corrección" más abajo). No
el catálogo completo, por el costo de correr 7 candidatos por SKU (17
min sobre 216 SKUs).

| Clase de demanda | Catálogo completo | Corrida |
|---|---|---|
| erratica | 1.175 | 114 |
| regular | 485 | 47 |
| **intermitente** | **40** | **40 (todos)** |
| lumpy | 152 | 15 |

`horizonte=3`, `ventana_minima=15` — igual que la POC anterior (dataset
de 25 meses). Candidatos: `benchmark`, `ets`, `tsb`, `xgboost`,
`random_forest`, `lightgbm_global`, `ensemble`. Sin `prophet` (costoso,
ya excluido en la POC anterior) ni `sarima` (descartado, ver
`docs/adr/0001-no-sarima.md`).

Los intermitentes se corrieron junto al resto de la muestra, no solos:
`lightgbm_global` entrena UN modelo con todas las SKUs del DataFrame que
recibe, así que correr los 40 intermitentes aislados habría producido un
modelo global entrenado solo con series intermitentes — distinto del que
compite en el pipeline real, y por lo tanto no comparable.

## Resultado: el ensemble gana más que cualquier candidato individual

| Candidato | SKUs | % |
|---|---|---|
| `ensemble` | 87 | 40,3% |
| `lightgbm_global` | 35 | 16,2% |
| `tsb` | 32 | 14,8% |
| `ets` | 27 | 12,5% |
| `benchmark` | 17 | 7,9% |
| `random_forest` | 13 | 6,0% |
| `xgboost` | 5 | 2,3% |

A diferencia de la POC anterior (donde el benchmark ganaba el 49% del
catálogo completo, sin ensemble compitiendo), acá el ensemble desplaza
claramente al resto — más del doble que el segundo candidato.

### Por clase de demanda

| Clase | SKUs | % ensemble ganador |
|---|---|---|
| regular | 47 | 42,6% |
| erratica | 114 | 42,1% |
| lumpy | 15 | 40,0% |
| intermitente | 40 | 32,5% |

El ensemble es el candidato más ganador en las cuatro clases. En
demanda intermitente su ventaja se atenúa (32,5% contra ~42% en el
resto) y `tsb` queda muy cerca (12 de 40 SKUs, 30,0%) — coherente con
que TSB está diseñado específicamente para ese patrón, pero **no** lo
desplaza.

### Corrección de un hallazgo anterior

La primera versión de este documento reportó que el ensemble ganaba
**0%** de los SKUs intermitentes y concluía que "promediarlo con
ETS/LightGBM diluye la ventaja de TSB en vez de sumarla". Ese resultado
salía de los 4 SKUs intermitentes que habían caído en la muestra
estratificada. Corriendo los 40 del catálogo, el ensemble gana 32,5% —
la conclusión anterior era un artefacto del tamaño de muestra, no un
patrón real. Se mantiene una atenuación de su ventaja en esa clase,
mucho más leve que lo reportado.

## Cuando el ensemble NO gana, pierde por bastante

Medido sobre la muestra inicial de 180 SKUs (96 casos donde el ensemble
no ganó pero sí tuvo WAPE definido):

- Brecha relativa media (ensemble vs. ganador real): **+99,0%** de WAPE
- Brecha relativa mediana: **+60,4%** de WAPE

El ensemble no es "casi tan bueno como el mejor" cuando pierde — pierde
con margen considerable. Es consistente con la naturaleza de una
combinación lineal: cuando un especialista le acierta a la estructura de
una serie, promediarlo con modelos que no la capturan empeora el
resultado. El ensemble gana en promedio (40% de las veces, con margen)
pero no es un "seguro" contra malos pronósticos SKU por SKU.

## Interpretación y recomendación

El ensemble justifica su complejidad: es el candidato con mayor tasa de
victorias en las cuatro clases de demanda, por un margen amplio. La
metodología de walk-forward anidado (que evita compararlo en ventaja
injusta, ver `ensemble_backtest.py`) da una comparación honesta y el
resultado sigue siendo favorable.

No hay evidencia de que convenga tratarlo como reemplazo universal: en
el 60% de los SKUs sigue ganando otro candidato, y cuando pierde, pierde
por márgenes grandes. El diseño actual —todos compiten por WAPE, sin
reglas fijas por clase de demanda— maneja esto correctamente sin
intervención adicional.

## Pendiente

- Correr sobre el catálogo completo (1.852 SKUs, ~2,4 hs extrapolando de
  esta corrida) si se necesita el número exacto de distribución para
  producción — esta muestra alcanza para validar la decisión de diseño,
  no para reportar un número de producción.
- Recalcular la brecha relativa "cuando pierde" sobre la corrida de 216
  SKUs (el número reportado arriba es de la muestra inicial de 180).
- Los scripts de descarga/preparación de Online Retail II y de esta
  validación quedaron en el scratchpad de la sesión (no versionados) —
  igual que en la POC anterior, son reproducibles a partir del dataset
  público de UCI y `src/forecast/ensemble_backtest.py`.

## Artefactos

- [`ensemble_selecciones_muestra.csv`](ensemble_selecciones_muestra.csv):
  candidato ganador por SKU de la corrida de 216, con sus métricas de
  backtest y la clase de demanda.
- [`ensemble_comparacion_vs_ganador.csv`](ensemble_comparacion_vs_ganador.csv):
  WAPE del ensemble vs. WAPE del ganador real, por SKU, sobre la muestra
  inicial de 180 — la base de la brecha relativa reportada arriba.
