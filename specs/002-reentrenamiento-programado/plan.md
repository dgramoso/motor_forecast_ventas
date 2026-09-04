# Plan de Implementación: Reentrenamiento Programado

**ID de Feature**: 002
**Spec de referencia**: `specs/002-reentrenamiento-programado/spec.md`
**Fecha**: 2026-09-04
**Estado**: BORRADOR

---

## Fase -1: Gates de Pre-implementación (OBLIGATORIO)

### Gate de Simplicidad
- [x] ¿Máximo 3 módulos/proyectos? — 3 archivos existentes modificados (`comparar_modelos.py`, `ensemble_backtest.py`, `pipeline.py`), ninguno nuevo. El "scheduler" no es un módulo de código: es una tarea de Windows Task Scheduler configurada fuera del repo.
- [x] ¿Sin features especulativas o "podría necesitarse"? — no se agrega notificación push, no se agrega campo de origen en el parquet, no se agrega reintento — todo ya descartado explícitamente en el PRD.
- [x] ¿Sin over-engineering para escala no demostrada? — ver ADR 1: se descarta Airflow por ser el único job a programar, sin otros DAGs ni infra de orquestación ya desplegada en este entorno.

### Gate Anti-abstracción
- [x] ¿Se usan frameworks directamente sin wrappers innecesarios? — `logging` stdlib directo, sin envoltorio propio de "logger de corridas". Windows Task Scheduler directo, sin capa de abstracción de scheduling.
- [x] ¿Representación única por entidad (sin modelos duplicados)? — la corrida programada es la misma entidad `run_id`/`corridas.parquet` que ya existe; no se crea un modelo paralelo de "corrida programada".

### Gate Integración-primero
- [x] ¿Contratos definidos antes del código? — el contrato de invocación (`python -m src.forecast.pipeline`, sin argumentos, cwd = raíz del repo) es el mismo que ya existe hoy; no hay contrato nuevo que diseñar, solo formalizar cómo lo dispara el Task Scheduler (sección 5).
- [x] ¿Tests de contrato planificados en fase Red? — sí, ver sección 6.

### Gate Test-First
- [x] ¿Tests escritos y aprobados ANTES del código de implementación? — ver Fase 0 del plan de entrega (sección 7).
- [x] ¿Tests confirmados en estado FAIL (Red) antes de implementar? — mismo punto.

**Decisiones de complejidad**: ninguna — todos los gates pasan sin excepción.

---

## 1. Resumen Técnico

**Enfoque general**: el pipeline ya es idempotente y persiste con `run_id`/`timestamp_utc` (append-only). Lo único que falta es (a) que una falla en un SKU puntual no tumbe la corrida completa, (b) que quede un registro local legible de cada corrida, y (c) el disparador periódico. (a) y (b) se resuelven con cambios quirúrgicos al código existente; (c) se resuelve enteramente con configuración nativa de Windows Task Scheduler, sin código nuevo.

**Principales desafíos técnicos**:
1. Aislar fallas por SKU sin romper la firma pública de `comparar_modelos()` (consumida por `comparar_modelos_con_ensemble()` y por los tests existentes) → usar `logging.warning()` dentro del loop en vez de cambiar el tipo de retorno.
2. Diferenciar "falla de SKU puntual" (la corrida sigue) de "falla de etapa compartida" (se aborta todo, sin persistir nada — spec 002 sección 6) sin agregar lógica de clasificación nueva → se deriva de **dónde** ocurre la excepción: dentro del loop por SKU queda aislada; fuera de él (`cargar_ventas()`, entrenamiento de LightGBM global) ya propaga y aborta hoy — comportamiento actual, sin cambios.

---

## 2. Stack Tecnológico

| Componente | Tecnología | Justificación | Requisito del spec |
|------------|-----------|--------------|-------------------|
| Scheduler | Windows Task Scheduler | Ya disponible en el SO, sin infra nueva; catch-up ("Run task as soon as possible after a scheduled start is missed") y no-solape ("Do not start a new instance") son settings nativos, no hay que programarlos | Historia 1, RNF Disponibilidad |
| Log de corridas | `logging` (stdlib) + `FileHandler` | Framework directo, cero dependencias nuevas | Historia 2 |
| Persistencia de resultados | `persistencia.py` existente, sin cambios de esquema | Ya cumple lo pedido; el PRD restringe explícitamente no tocar el esquema | Historia 1, restricción del PRD (sección 8) |

---

## 3. Arquitectura del Sistema

### Diagrama de componentes
```
[Windows Task Scheduler] ──trigger (cadencia configurable, catch-up, no-solape)──▶ python -m src.forecast.pipeline
                                                                                            │
                                                                                   pipeline.main()
                                                                              (configura logging → archivo)
                                                                                            │
                                                                                   ejecutar_pipeline()
                                                                                            │
                                                              comparar_modelos_con_ensemble()
                                                              ├─ comparar_modelos()  ──try/except por SKU──▶ logging.warning(sku_id, motivo)
                                                              └─ backtest_y_predicciones_lightgbm_global()  (falla = etapa compartida, propaga)
                                                                                            │
                                                                                seleccionar_mejor_modelo()
                                                                                            │
                                                                                pronosticar_futuro()
                                                                     (opera sobre tabla_comparativa: los SKU
                                                                      que fallaron arriba ya no aparecen acá)
                                                                                            │
                                                            guardar_corrida() / guardar_pronosticos()   ← solo si no hubo abort
                                                                                            │
                                                                       logging.info(resumen agregado)
```

### Módulos y responsabilidades

**`src/forecast/comparar_modelos.py`** (modificado)
- Responsabilidad: `comparar_modelos()` envuelve cada iteración del loop por SKU en `try/except Exception`; ante una excepción no manejada, loggea `WARNING` con `sku_id` + tipo/mensaje del error y continúa con el resto. No cambia su firma pública — sigue devolviendo el mismo `DataFrame`, ahora sin las filas de los SKUs que fallaron.
- Interfaz pública: sin cambios (`comparar_modelos(ventas, horizonte, ventana_minima, candidatos) -> pd.DataFrame`).
- Dependencias: ninguna nueva.

**`src/forecast/ensemble_backtest.py`** (modificado, mismo patrón)
- Responsabilidad: el tramo que corre ETS/TSB por SKU dentro de `comparar_modelos_con_ensemble()` recibe el mismo aislamiento try/except + logging que `comparar_modelos()`. El punto exacto (`_backtest_y_predicciones_por_candidato` o el loop que lo invoca) se confirma al derivar tareas (Fase 3, fuera de este documento).
- Interfaz pública: sin cambios.

**`src/forecast/pipeline.py`** (modificado, mínimo)
- Responsabilidad: `main()` configura logging a archivo (`FileHandler`) antes de llamar a `ejecutar_pipeline()`. Si la corrida termina con éxito, loggea `INFO` con el resumen agregado (SKUs procesados, tasa de fallback promedio, distribución de candidatos ganadores). Si una etapa compartida lanza una excepción no manejada, la loggea a `ERROR` con traceback y la deja propagar (el proceso termina con exit code ≠ 0 — señal redundante que también queda en el historial nativo de Task Scheduler).
- Interfaz pública: `ejecutar_pipeline()` sin cambios de firma; `main()` gana el setup de logging.

**Nada nuevo**: no se crea un módulo de "scheduler", ni un wrapper de notificaciones, ni un cliente de colas. La programación vive enteramente en la configuración del Task Scheduler, documentada en la sección 10 (Quickstart), no en el repo.

---

## 4. Modelo de Datos (resumen)

Sin cambios al esquema de negocio — se reutilizan `corridas.parquet` y `pronosticos.parquet` tal cual (restricción explícita del PRD, spec 002 sección 8). El único artefacto nuevo es el archivo de log (`logs/corridas_programadas.log`), que no es una entidad de negocio — es observabilidad de la corrida, texto plano, sin schema versionado.

### Entidades principales
| Entidad | Propósito | Atributos clave |
|---------|-----------|----------------|
| Corrida (existente, sin cambios) | Una ejecución completa del pipeline | `run_id`, `timestamp_utc`, `sku_id`, `candidato`, métricas |
| Entrada de log (nueva, no persistida como entidad de negocio) | Registro consultable de una corrida programada | timestamp, nivel (INFO/WARNING/ERROR), `run_id` (cuando exista), mensaje |

### Relaciones clave
- Una entrada de log de nivel INFO (resumen) referencia el `run_id` de la Corrida que generó — permite cruzar log ↔ parquet si hace falta auditar en detalle.

---

## 5. Contrato de Invocación (reemplaza "Contratos de API" — no aplica REST)

| Acción | Comando | Propósito | Historia |
|--------|---------|-----------|---------|
| Task Scheduler → Action | `python -m src.forecast.pipeline` (Start in = raíz del repo) | Disparar una corrida completa, igual que la invocación manual de hoy | Historia 1 |

No hay eventos ni WebSockets — es un proceso batch de un solo disparo por corrida.

---

## 6. Estrategia de Testing

> Orden obligatorio: contrato → integración → e2e → unitario

### Tests de contrato
- Un test de integración que invoca `python -m src.forecast.pipeline` como subprocess (o llama a `main()` directamente capturando logs) contra un CSV de prueba chico, y verifica: exit code 0, `corridas.parquet`/`pronosticos.parquet` con un nuevo `run_id`, y una entrada de resumen en el log.

### Tests de integración
- Un candidato que lanza una excepción arbitraria (no una de las ya contempladas por el fallback existente) para un SKU puntual dentro de `comparar_modelos()`: el resto de los SKUs se procesan igual, ese SKU queda ausente de la tabla resultante y de `corridas.parquet` de esa corrida, y aparece un `WARNING` en el log con su `sku_id` y motivo.
- `cargar_ventas()` mockeada para lanzar una excepción: la corrida se aborta, no se persiste ninguna fila nueva en `corridas.parquet` ni `pronosticos.parquet`, y aparece un `ERROR` en el log.
- Usar datos reales (el CSV sintético del repo o un fixture chico derivado de él) — sin mocks del pipeline en sí, solo del punto de falla bajo prueba (regla de integración-primero: BD/archivos reales).

### Tests e2e
- Corrida completa contra el fixture de test existente (`tests/forecast/_helpers.py`), verificando el flujo happy path descripto en la spec (sección 7): corrida dispara → persiste → log con resumen → `obtener_pronostico_vigente()` refleja el resultado.

### Tests unitarios
- `comparar_modelos()` con un `candidatos` dict de prueba que incluya una función que lanza para un `sku_id` específico — confirma aislamiento y contenido del log (usar `caplog` de pytest).
- Cálculo del resumen agregado (a partir de `selecciones`) como función pura, testeada sin logging de por medio.

---

## 7. Plan de Entrega por Fases

### Fase 0: Fundación
**Prerrequisitos**: ninguno
**Entregables**:
- [ ] Test (rojo) de aislamiento por SKU en `comparar_modelos()`
- [ ] Test (rojo) que hace explícito el abort-sin-persistir ante falla de etapa compartida (formaliza el comportamiento actual, antes implícito)
- [ ] Test (rojo) del resumen agregado logueado en `pipeline.py:main()`
- [ ] Fixture de test para simular una excepción no contemplada por el fallback existente (distinta de las que ya capturan `_ajuste_con_fallback.py`)

### Fase 1: Aislamiento por SKU (Historia 3)
**Prerrequisitos**: Fase 0 completa
**Requisitos del spec**: Historia 3
**Entregables**:
- [ ] `try/except` + `logging.warning` por SKU en `comparar_modelos()`
- [ ] Mismo patrón en el tramo ETS/TSB de `ensemble_backtest.py`
- [ ] Tests de integración de la sección 6 pasando

### Fase 2: Logging y resumen agregado (Historia 2)
**Prerrequisitos**: Fase 1 completa
**Requisitos del spec**: Historia 2
**Entregables**:
- [ ] Configuración de logging a archivo en `pipeline.py:main()` (INFO = resumen de éxito, ERROR = falla de etapa compartida con traceback)
- [ ] Cálculo del resumen agregado (SKUs procesados, tasa de fallback promedio, distribución de candidatos ganadores) a partir de `selecciones`
- [ ] Tests e2e de la sección 6 pasando

### Fase Final: Producción (Historia 1, RNF Disponibilidad)
**Prerrequisitos**: Fase 1 y Fase 2 completas
**Entregables**:
- [ ] Tarea creada en Windows Task Scheduler: trigger con la cadencia que se defina al desplegar (spec 002, restricción — sin default en el repo), "Run task as soon as possible after a scheduled start is missed" activado, "If the task is already running: Do not start a new instance" activado
- [ ] Documentación de operación (cómo crear/editar la tarea, dónde queda el log, cómo distinguir un resumen exitoso de una falla)
- [ ] Criterio de rollback: deshabilitar la tarea en Task Scheduler vuelve al estado 100% manual, sin tocar código ni datos persistidos

---

## 8. Consideraciones de Seguridad

| Riesgo | Mitigación | Fase de implementación |
|--------|-----------|----------------------|
| La tarea corre con las credenciales de un usuario de Windows; si ese usuario pierde acceso a la carpeta del repo o de datos, la corrida falla y nadie lo nota si no revisa Task Scheduler | El log de `ERROR` (falla de etapa compartida) + el historial nativo de Task Scheduler (exit code ≠ 0) son dos señales redundantes de la misma falla | Fase Final |

---

## 9. Consideraciones de Rendimiento

| Requisito del spec | Estrategia técnica | Cómo se valida |
|-------------------|-------------------|----------------|
| La corrida completa (~74 min en la POC) no debe solaparse con la siguiente (spec 002, Historia 1) | Setting nativo "If the task is already running: Do not start a new instance" | Verificación manual en Fase Final: programar dos triggers cercanos entre sí y confirmar en el historial de Task Scheduler que el segundo no arrancó |

---

## 10. Guía de Validación Rápida (Quickstart)

```powershell
# 1. Confirmar que el pipeline corre manualmente hoy (baseline)
python -m src.forecast.pipeline

# 2. Crear la tarea programada (cadencia de ejemplo: diaria a las 3am — ajustar al valor real definido al desplegar)
$action = New-ScheduledTaskAction -Execute "python" -Argument "-m src.forecast.pipeline" -WorkingDirectory "C:\Users\dgramoso\Desktop\motor_forecast_ventas"
$trigger = New-ScheduledTaskTrigger -Daily -At 3am
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "MotorForecastVentas-ReentrenamientoProgramado" -Action $action -Trigger $trigger -Settings $settings

# 3. Forzar una corrida fuera de horario para validar el circuito completo
Start-ScheduledTask -TaskName "MotorForecastVentas-ReentrenamientoProgramado"

# 4. Verificar resultado
Get-Content logs\corridas_programadas.log -Tail 20
```

**Resultado esperado**: el log muestra una entrada de resumen agregado (SKUs procesados, tasa de fallback promedio, distribución de candidatos ganadores) y `data/runs/corridas.parquet` tiene un `run_id` nuevo con `timestamp_utc` reciente.

---

## 11. Registro de Decisiones Técnicas (ADR)

| # | Decisión | Alternativas consideradas | Razón de elección |
|---|---------|--------------------------|------------------|
| 1 | Windows Task Scheduler como mecanismo de scheduling | Airflow (hay skill disponible en el entorno); cron vía WSL; servicio propio con `schedule`/`APScheduler` corriendo en background | Sin infra nueva; cadencia, catch-up y no-solape son settings nativos del SO. Airflow es over-engineering para un único job sin otros DAGs ni orquestación ya desplegada en este entorno (no se encontró `dags/`, `airflow.cfg` ni `docker-compose` en el repo) — viola el Gate de Simplicidad si se elige sin evidencia de necesidad real |
| 2 | `logging` stdlib a archivo local, sin canal push | Email, Slack, u otro webhook | Ya decidido en el PRD (spec 002, pregunta abierta 1, resuelta): log local, sin push activo |
| 3 | Aislamiento de fallas por SKU vía `try/except` + `logging.warning` dentro del loop ya existente en `comparar_modelos()`, sin cambiar su firma | Cambiar `comparar_modelos()` para devolver una tupla `(tabla, fallas)`; crear un módulo "runner" nuevo que orqueste SKU por SKU desde afuera del pipeline actual | Cambiar la firma rompe callers existentes (`comparar_modelos_con_ensemble()`, los tests de `tests/forecast/`); un módulo "runner" nuevo duplicaría el loop que ya existe. El patrón try/except + logging es la intervención mínima que satisface Historia 3 sin abstracción nueva (Gate Anti-abstracción) |
| 4 | Abortar sin persistir ante falla de etapa compartida se logra "gratis": el código actual ya llama a `guardar_corrida()`/`guardar_pronosticos()` recién al final de `ejecutar_pipeline()`, después de todas las etapas compartidas | Envolver `ejecutar_pipeline()` en una transacción explícita o checkpoint intermedio | No hace falta: si una etapa compartida (`cargar_ventas()`, LightGBM global) lanza antes de llegar a la persistencia, no se persiste nada — es el comportamiento ya vigente, solo se le agrega logging explícito (spec 002, pregunta 3) |

---

## Historial de Cambios

| Versión | Fecha | Cambio | Autor |
|---------|-------|--------|-------|
| 0.1 | 2026-09-04 | Borrador inicial, vía `/diseno-proyecto` Fase 2 a partir del PRD aprobado (spec.md v0.2) | Daniel Gramoso |
