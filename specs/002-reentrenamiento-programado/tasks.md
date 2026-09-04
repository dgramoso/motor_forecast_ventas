# Lista de Tareas: Reentrenamiento Programado

**ID de Feature**: 002
**Plan de referencia**: `specs/002-reentrenamiento-programado/plan.md`
**Fecha de generación**: 2026-09-04
**Estado**: PENDIENTE

---

## Leyenda

- `[P]` — Tarea paralelizable (sin dependencias bloqueantes dentro del grupo)
- `[S]` — Tarea secuencial (debe ejecutarse en orden)
- `[B]` — Bloqueante para el siguiente grupo

---

## Grupo 0: Fundación 🏗️
> Prerequisito para todos los grupos siguientes. Todos los tests de este grupo deben quedar confirmados en estado FAIL antes de escribir código de implementación (Gate Test-First del plan).

```
[T000] [S][B] Test rojo: aislamiento por SKU en comparar_modelos()
  - Descripción: test que pasa un `candidatos` dict con una función que lanza una
    excepción arbitraria (no contemplada por `_ajuste_con_fallback.py`) para un
    `sku_id` específico. Confirma: el resto de los SKUs siguen apareciendo en la
    tabla resultante, ese SKU no, y queda un WARNING logueado con su `sku_id`
    y el motivo (usar `caplog` de pytest).
  - Entradas: plan.md (sección 6, Tests unitarios); T004
  - Salidas: test nuevo en `tests/forecast/test_comparar_modelos.py`, confirmado en FAIL
  - Dependencias: T004

[T001] [S][B] Test rojo: aislamiento por SKU en el tramo ETS/TSB del ensemble
  - Descripción: mismo patrón que T000, aplicado a
    `_backtest_y_predicciones_por_candidato()` en `ensemble_backtest.py:127`
    (loop por SKU independiente del de `comparar_modelos()`).
  - Entradas: plan.md (sección 3, módulo `ensemble_backtest.py`); T004
  - Salidas: test nuevo en `tests/forecast/test_ensemble_backtest.py`, confirmado en FAIL
  - Dependencias: T004

[T002] [S][B] Test rojo: abort-sin-persistir ante falla de etapa compartida
  - Descripción: mockear `cargar_ventas()` para lanzar una excepción; correr
    `ejecutar_pipeline()`; confirmar que no se agrega ninguna fila nueva a
    `corridas.parquet` ni `pronosticos.parquet`, y que la excepción se loguea a
    ERROR con traceback antes de propagar.
  - Entradas: spec.md (sección 6, casos borde); plan.md (sección 6, tests de integración)
  - Salidas: test nuevo en `tests/forecast/test_pipeline.py`, confirmado en FAIL
  - Dependencias: ninguna

[T003] [P] Test rojo: resumen agregado logueado en pipeline.py:main()
  - Descripción: correr `main()` contra el fixture de test chico; confirmar una
    entrada INFO en el log con SKUs procesados, tasa de fallback promedio y
    distribución de candidatos ganadores.
  - Entradas: spec.md (Historia 2); plan.md (sección 6, tests e2e)
  - Salidas: test nuevo en `tests/forecast/test_pipeline.py`, confirmado en FAIL
  - Dependencias: ninguna

[T004] [P] Fixture de test: excepción no contemplada por el fallback existente
  - Descripción: agregar en `tests/forecast/_helpers.py` una función de candidato
    de prueba que lanza una excepción arbitraria (ej. `ValueError`) para un
    `sku_id` dado, distinta de las que ya captura `_ajuste_con_fallback.py` —
    para poder simular una falla "no manejada" real en los tests de T000/T001.
  - Entradas: `src/forecast/_ajuste_con_fallback.py` (qué excepciones ya contempla)
  - Salidas: helper reutilizable en `tests/forecast/_helpers.py`
  - Dependencias: ninguna
```

**Estado del grupo**: [ ] Completo (continuar solo cuando T000-T003 estén confirmados en FAIL)

---

## Grupo 1: Aislamiento por SKU (Historia 3) 🔨
> Prerequisito: Grupo 0 completo.

```
[T010] [S] Implementar try/except + logging.warning en comparar_modelos()
  - Descripción: envolver cada iteración del loop por SKU (`comparar_modelos.py:135`)
    en `try/except Exception`; loguear WARNING con `sku_id` + tipo/mensaje del
    error; continuar con el resto. Sin cambiar la firma pública de la función.
  - Entradas: T000 (rojo), T004 (fixture)
  - Salidas: `comparar_modelos.py` modificado; T000 en PASS
  - Dependencias: T000, T004

[T011] [S] Implementar el mismo patrón en _backtest_y_predicciones_por_candidato()
  - Descripción: mismo `try/except` + `logging.warning` en el loop por SKU de
    `ensemble_backtest.py:140`, consistente con T010.
  - Entradas: T001 (rojo), T004; patrón validado en T010
  - Salidas: `ensemble_backtest.py` modificado; T001 en PASS
  - Dependencias: T010, T001

[T012] [S] Resolver consistencia del candidato "ensemble" cuando ETS o TSB fallaron para un SKU
  - Descripción: hoy, si `_backtest_y_predicciones_por_candidato` excluye un SKU
    (T011), `evaluar_ensemble()` (`ensemble_backtest.py:191`) lo sigue procesando
    con predicciones vacías (`.get(sku_id, ([], []))`), degradando naturalmente a
    "sin datos suficientes" (NaN) en esa fila — no lo excluye. Decidir si ese
    comportamiento ya existente es aceptable para v1 (probablemente sí: es el
    mismo criterio de "sin datos suficientes" que ya usa el resto del sistema,
    ver CONTEXT.md) o si hace falta excluir el SKU también de la fila de
    ensemble. Documentar la decisión en un comentario o, si aplica, en un ADR.
  - Entradas: T011
  - Salidas: comportamiento confirmado y, si corresponde, ajustado en
    `evaluar_ensemble()`; decisión documentada
  - Dependencias: T011

[T019] [S][B] Tests de integración del Grupo 1 en PASS
  - Descripción: correr los tests de integración de la sección 6 del plan (SKU
    puntual falla → resto persiste, log con WARNING) contra el código real.
  - Entradas: T010, T011, T012
  - Salidas: `tests/forecast/test_comparar_modelos.py` y
    `tests/forecast/test_ensemble_backtest.py` en PASS
  - Dependencias: T010, T011, T012
```

**Estado del grupo**: [ ] Completo

---

## Grupo 2: Logging y resumen agregado (Historia 2) ⚙️
> Prerequisito: Grupo 1 completo.

```
[T020] [S] Configurar logging a archivo en pipeline.py:main()
  - Descripción: agregar un `FileHandler` (ej. `logs/corridas_programadas.log`)
    antes de llamar a `ejecutar_pipeline()`. INFO para el resumen de éxito,
    ERROR con traceback si una etapa compartida lanza antes de la persistencia.
  - Entradas: T002 (rojo), T003 (rojo); plan.md (sección 3, módulo `pipeline.py`)
  - Salidas: `pipeline.py` modificado con logging configurado
  - Dependencias: T019

[T021] [P] Calcular el resumen agregado a partir de selecciones
  - Descripción: función pura (SKUs procesados, tasa de fallback promedio,
    distribución de candidatos ganadores) a partir del `DataFrame` que devuelve
    `seleccionar_mejor_modelo()`. Sin logging de por medio — testeable aislada.
  - Entradas: `src/forecast/seleccionar_modelo.py` (esquema de `selecciones`,
    ver CONTEXT.md — "Tasa de fallback")
  - Salidas: función nueva (ubicación a decidir al implementar: `pipeline.py` o
    un helper propio) + test unitario en PASS
  - Dependencias: T019

[T022] [S] Integrar el resumen agregado al logging de éxito
  - Descripción: conectar la salida de T021 con el `logging.info()` de T020 en
    `pipeline.py:main()`.
  - Entradas: T020, T021
  - Salidas: `pipeline.py` completo con resumen agregado logueado en éxito
  - Dependencias: T020, T021

[T023] [S] Confirmar abort-sin-persistir + logging ERROR ante falla de etapa compartida
  - Descripción: verificar contra T002 que, con el logging ya configurado
    (T020), la falla de una etapa compartida queda registrada en ERROR con
    traceback y que efectivamente no se persiste nada — este comportamiento ya
    existe (plan.md, ADR 4), acá solo se confirma con logging real de por medio.
  - Entradas: T002 (rojo), T020
  - Salidas: T002 en PASS
  - Dependencias: T020

[T029] [S][B] Tests e2e del Grupo 2 en PASS
  - Descripción: corrida completa contra el fixture de test chico — dispara,
    persiste, loguea resumen, y `obtener_pronostico_vigente()` refleja el
    resultado (happy path de spec.md, sección 7).
  - Entradas: T020, T021, T022, T023
  - Salidas: `tests/forecast/test_pipeline.py` en PASS (incluye T002 y T003 ya en verde)
  - Dependencias: T020, T021, T022, T023
```

**Estado del grupo**: [ ] Completo

---

## Grupo Final: Producción 🚀
> Prerequisito: Todos los grupos anteriores completos.

```
[T900] [P] Crear la tarea en Windows Task Scheduler
  - Descripción: `Register-ScheduledTask` con la cadencia que se defina al
    desplegar (spec 002 — sin default en el repo), `-StartWhenAvailable`
    (catch-up) y `-MultipleInstances IgnoreNew` (no-solape). Ver plan.md
    sección 10 (Quickstart) para el comando de referencia.
  - Entradas: plan.md (sección 10)
  - Salidas: tarea programada activa en el entorno de despliegue
  - Dependencias: T029

[T901] [P] Documentación de operación
  - Descripción: cómo crear/editar la tarea, dónde queda el log
    (`logs/corridas_programadas.log`), cómo distinguir un resumen exitoso de
    una falla (INFO vs. ERROR), y cómo forzar una corrida manual de validación.
  - Entradas: plan.md (secciones 7 y 10)
  - Salidas: documento de operación (ubicación a decidir — ej. junto a
    `docs/agents/` o en un README propio de la feature)
  - Dependencias: T029

[T902] [P] Documentar el criterio de rollback
  - Descripción: deshabilitar la tarea en Task Scheduler vuelve al estado
    100% manual, sin tocar código ni datos persistidos — dejarlo explícito en
    la misma documentación de T901.
  - Entradas: plan.md (sección 7, Fase Final)
  - Salidas: sección de rollback en el documento de T901
  - Dependencias: T901

[T999] [S] Validación final con criterios de aceptación del spec
  - Descripción: ejecutar manualmente los criterios de aceptación de las
    Historias 1, 2 y 3 (spec.md) contra una corrida real forzada por Task
    Scheduler — incluye provocar una falla de SKU puntual y una de etapa
    compartida para confirmar el comportamiento en producción, no solo en tests.
  - Entradas: spec.md (todos los Criterios de Aceptación)
  - Salidas: checklist de aceptación completado
  - Dependencias: T900, T901, T902
```

**Estado del grupo**: [ ] Completo

---

## Resumen de Paralelización

```
Grupo 0: T004 → T000,T001 (paralelo entre sí, ambos dependen de T004); T002,T003 en paralelo, sin dependencia de T004
                      ↓
Grupo 1: T010 → T011 → T012 → T019
                      ↓
Grupo 2: T020,T021 (paralelo) → T022 → T023 → T029
                      ↓
Final:   T900,T901 (paralelo) → T902 → T999
```

---

## Métricas de Progreso

| Grupo | Total tareas | Completadas | % |
|-------|-------------|-------------|---|
| Grupo 0 | 5 | 0 | 0% |
| Grupo 1 | 4 | 0 | 0% |
| Grupo 2 | 5 | 0 | 0% |
| Final | 4 | 0 | 0% |
| **Total** | **18** | **0** | **0%** |
