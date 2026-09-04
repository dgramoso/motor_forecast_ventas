# Especificación de Feature: Reentrenamiento Programado

**ID de Feature**: 002
**Rama**: `002-reentrenamiento-programado`
**Fecha**: 2026-09-04
**Estado**: APROBADO
**Autor**: Daniel Gramoso

---

## 1. Resumen Ejecutivo

**Problema que resuelve**: hoy el motor de forecast solo se ejecuta cuando alguien corre `python -m src.forecast.pipeline` a mano — el pronóstico vigente se queda desactualizado hasta que alguien se acuerde de reentrenar.

**Solución propuesta**: correr `ejecutar_pipeline()` en una cadencia programada, sin intervención manual, dejando cada corrida persistida igual que hoy (`corridas.parquet`, `pronosticos.parquet`) y avisando el resultado.

**Usuarios objetivo**: quien consume `obtener_pronostico_vigente()` (o los archivos de `data/runs/`) para planificación — hoy es el propio Daniel Gramoso; no hay otros consumidores identificados todavía.

**Valor de negocio**: el pronóstico deja de degradarse silenciosamente por falta de reejecución manual, sin agregar trabajo operativo recurrente.

---

## 2. Contexto y Motivación

### Situación actual

El pipeline completo (`src/forecast/pipeline.py:ejecutar_pipeline`) ya es idempotente y soporta corridas repetidas: cada corrida genera un `run_id` propio, se agrega (append-only) a `corridas.parquet`/`pronosticos.parquet`, y "vigente" se deriva por SKU como la corrida de mayor `timestamp_utc` — no hay estado que romper por correr de nuevo. Lo único que falta es el disparador: hoy es 100% manual.

`cargar_ventas()` (`src/datos/cargar_datos.py`) lee un CSV sintético fijo (`data/synthetic/ventas_historicas.csv`), no un DWH real. **Esta feature no incluye conectar una fuente de datos real** — se diseña el mecanismo de programación contra el estado actual de `cargar_ventas()`, sea cual sea su fuente en el momento de la corrida. El día que `cargar_ventas()` apunte a un DWH real (spec 001, sección 6), el reentrenamiento programado empieza a aportar valor real sin cambios en esta feature.

### Por qué ahora

El motor ya pasó por una prueba de concepto sobre datos reales (Online Retail II) y hardening de bugs — es la base estable a partir de la cual tiene sentido automatizar la reejecución, en vez de seguir corriendo el pipeline a mano cada vez.

### Métricas de éxito

| Métrica | Valor actual | Objetivo | Plazo |
|---------|-------------|---------|-------|
| Corridas manuales necesarias para mantener el pronóstico vigente | 100% | 0% (todas programadas) | Al cierre de v1 |
| Corridas programadas que terminan sin que nadie se entere del resultado | N/A (no existe programación hoy) | 0% — toda corrida notifica éxito o falla | Al cierre de v1 |

---

## 3. Alcance

### Incluido en esta versión (v1)

- Disparar `ejecutar_pipeline()` en una cadencia programada, sin acción manual.
- Persistir el resultado de cada corrida igual que hoy (sin cambios al esquema de `persistencia.py` — no se agrega campo de origen programada/manual, alcanza con `run_id`/`timestamp_utc` existentes).
- Si una corrida falla a mitad de camino en un SKU puntual, persistir lo que se pudo completar y registrar qué falló. Si falla una etapa compartida por todos los SKUs (ej. `cargar_ventas()`), abortar toda la corrida sin persistir nada y sin reintento automático (ver sección 6).
- Registrar el resultado de cada corrida programada, tanto éxito como falla, en un log/archivo de estado local consultable — con un resumen agregado (cantidad de SKUs procesados, tasa de fallback promedio, distribución de candidatos ganadores; sin detalle por SKU). No es una notificación push (email/Slack) — ver Historia 2.
- Evitar corridas superpuestas (una corrida programada no debe arrancar si la anterior sigue en curso).
- Si el scheduler no estuvo disponible en el momento programado (máquina apagada, servicio caído), ejecutar la corrida faltante apenas vuelve a estar disponible (catch-up automático).

### Explícitamente EXCLUIDO

- Conectar `cargar_ventas()` a un DWH/fuente de datos real — sigue leyendo lo que `cargar_datos.py` defina en cada momento, sin cambios en esta feature.
- Dashboard de monitoreo de fallback/WAPE a lo largo del tiempo — es un proyecto separado, ya identificado en el handoff previo.
- Ampliar cobertura de SARIMA/Prophet al catálogo completo — cuestión de costo de cómputo, no de esta feature.
- Elegir la infraestructura de scheduling (Task Scheduler, Airflow, cron, cloud) — es una decisión técnica, se resuelve en el plan de implementación (Fase 2), no en este PRD.
- Notificación push (email/Slack) — v1 usa log/archivo local, sin canal activo.
- Reintento automático ante falla de etapa compartida — se aborta directo (ver sección 6).
- Campo nuevo de origen (programada/manual) en `corridas.parquet` — no hay necesidad identificada.
- Fijar la cadencia concreta — queda como parámetro configurable del mecanismo de scheduling, sin default fijo en este PRD (ver sección 2 y pregunta abierta 6 cerrada).

### Dependencias externas

- Ninguna nueva respecto del pipeline actual — reutiliza `ejecutar_pipeline()`, `persistencia.py` y el catálogo de candidatos existentes tal cual están.

---

## 4. Historias de Usuario

### Historia 1: Reentrenamiento sin intervención manual

```
Como responsable del motor de forecast
Quiero que el pipeline se reejecute solo, en una cadencia definida
Para que el pronóstico vigente no dependa de que alguien se acuerde de correrlo a mano
```

**Criterios de aceptación**:
- [ ] Dado que llega el momento programado, cuando no hay una corrida en curso, entonces se dispara una nueva corrida completa de `ejecutar_pipeline()` sin acción manual.
- [ ] Dado que una corrida programada está en curso, cuando llega el momento de la siguiente corrida programada, entonces esa nueva corrida no arranca (no hay corridas superpuestas).
- [ ] Dado que una corrida programada termina con éxito, cuando se consulta `obtener_pronostico_vigente()`, entonces refleja los resultados de esa corrida.

**Prioridad**: MUST
**Estimación de valor**: ALTA

---

### Historia 2: Visibilidad del resultado de cada corrida

```
Como responsable del motor de forecast
Quiero poder consultar si cada corrida programada terminó bien o mal
Para poder actuar si algo falló, revisando un único lugar en vez de logs sueltos o parquet a mano
```

**Criterios de aceptación**:
- [ ] Dado que una corrida programada termina con éxito, cuando termina, entonces queda registrado en un log/archivo de estado local consultable, con un resumen agregado (cantidad de SKUs procesados, tasa de fallback promedio, distribución de candidatos ganadores) — sin detalle por SKU.
- [ ] Dado que una corrida programada falla (parcial, por un SKU puntual, o total, por una etapa compartida), cuando termina, entonces queda registrado en ese mismo log/archivo qué falló (SKU o etapa, y el motivo).
- [ ] No hay notificación push (email/Slack) en v1 — enterarse del resultado requiere consultar el log/archivo de estado, no es automático hacia una persona.

**Prioridad**: MUST
**Estimación de valor**: ALTA

---

### Historia 3: Corrida resiliente a fallas parciales

```
Como responsable del motor de forecast
Quiero que si un SKU rompe el ajuste de un candidato, el resto de la corrida no se pierda
Para no quedarme sin pronóstico actualizado por un problema puntual y aislado
```

**Criterios de aceptación**:
- [ ] Dado que un candidato falla su ajuste para un SKU dado *dentro del comportamiento de fallback ya existente* (ver `CONTEXT.md` — "Fallback"), cuando eso ocurre, entonces la corrida sigue normalmente (esto ya es el comportamiento actual del pipeline, no cambia).
- [ ] Dado que ocurre una excepción no manejada por el pipeline (fuera del fallback ya previsto) para un SKU puntual, cuando eso ocurre, entonces los SKUs que sí se pudieron procesar quedan persistidos igual, y el SKU que falló queda registrado como falla en el log de esa corrida.
- [ ] Dado que ocurre una excepción no manejada en una etapa compartida por todos los SKUs (ej. `cargar_ventas()` falla por completo), cuando eso ocurre, entonces la corrida entera se aborta sin persistir nada, sin reintento automático, y queda registrada como falla total en el log.

**Prioridad**: MUST
**Estimación de valor**: MEDIA

---

## 5. Requisitos No Funcionales

### Rendimiento
- La corrida completa sobre el catálogo (sin SARIMA/Prophet) tomó ~74 min en la POC — el scheduling debe asumir corridas de duración similar y no solapar la siguiente disparada mientras la anterior sigue corriendo (ver Historia 1).

### Seguridad
- No aplica cambio respecto del pipeline actual — no se introduce manejo de credenciales ni datos sensibles nuevos en esta feature (la conexión a DWH real queda fuera de alcance).

### Disponibilidad
- Si la máquina/infraestructura donde corre el job no está disponible en el momento programado, se ejecuta la corrida faltante apenas vuelve a estar disponible (catch-up automático), en vez de saltarla.

### Escalabilidad
- No aplica — mismo catálogo de SKUs y candidatos que corre hoy manualmente.

### Observabilidad
- No hace falta distinguir en `corridas.parquet` una corrida programada de una manual — alcanza con lo que ya persiste (`run_id`, `timestamp_utc`); no hay necesidad identificada de auditar el origen de la corrida.
- El log/archivo de estado de cada corrida (Historia 2) es el mecanismo principal de observabilidad de esta feature.

---

## 6. Casos Borde y Escenarios de Error

| Escenario | Comportamiento esperado |
|-----------|------------------------|
| Un candidato cae en fallback para un SKU (comportamiento normal ya documentado en `CONTEXT.md`) | No es una falla de la corrida — se persiste como siempre, sin generar alerta especial. |
| Una excepción no manejada rompe el procesamiento de un SKU puntual | El resto de la corrida continúa; ese SKU queda excluido de esa corrida y reportado como falla en la notificación (Historia 3). |
| Una excepción no manejada rompe una etapa compartida por todos los SKUs (ej. `cargar_ventas()` falla) | Se aborta toda la corrida, no se persiste nada, sin reintento automático; queda registrada como falla total en el log. |
| Llega el momento de una corrida programada mientras la corrida anterior sigue en curso | La nueva corrida no arranca (Historia 1) — esa ejecución programada se descarta, no se reintenta en un slot posterior (la próxima corrida sigue el calendario normal). |
| El scheduler mismo no está disponible en el momento programado (máquina apagada, servicio caído) | Se ejecuta la corrida faltante apenas vuelve a estar disponible (catch-up automático). |

---

## 7. Experiencia de Usuario (sin diseño técnico)

### Flujo principal (happy path)

1. Llega el momento programado de la cadencia definida.
2. El sistema dispara una corrida completa del pipeline (equivalente a `ejecutar_pipeline()`).
3. El sistema persiste el resultado igual que una corrida manual hoy (`corridas.parquet`, `pronosticos.parquet`).
4. El sistema registra el resultado en el log/archivo de estado local, con un resumen agregado.
5. Quien consulta `obtener_pronostico_vigente()` ve el pronóstico actualizado, sin haber tenido que correr nada a mano.

### Flujos alternativos

- **Si la corrida anterior sigue en curso al llegar el momento programado**: la nueva corrida no arranca; esa ejecución programada se descarta (Historia 1).
- **Si falla un SKU puntual**: el resto de la corrida se persiste igual; el log indica qué SKU falló y por qué (Historia 3).
- **Si falla una etapa compartida (toda la corrida)**: se aborta sin persistir nada, sin reintento; queda registrado en el log (Historia 3).
- **Si el scheduler no estuvo disponible en el momento programado**: se ejecuta apenas vuelve a estar disponible (catch-up automático).

---

## 8. Restricciones y Suposiciones

### Restricciones

- No se cambia el esquema de persistencia existente (`persistencia.py`) — no se agrega campo de origen programada/manual.
- El mecanismo de scheduling elegido (Fase 2) debe poder invocar `ejecutar_pipeline()` (o el equivalente vía CLI, `python -m src.forecast.pipeline`) sin requerir cambios en su firma.
- La cadencia queda como parámetro configurable del mecanismo elegido, sin valor default fijado en este PRD — se define al desplegar (Fase 2 o después), una vez que se conecte una fuente de datos real y se sepa su frecuencia de actualización.

### Suposiciones

- `ejecutar_pipeline()` sigue siendo idempotente y sin estado compartido entre corridas — si eso cambia, este PRD debería revisarse.
- La fuente de datos que consuma `cargar_ventas()` en el momento de correr esta feature (sintética o real) es responsabilidad de otra iniciativa — el scheduling no valida ni depende de cuál sea.
- El log/archivo de estado local (Historia 2) es suficiente como mecanismo de visibilidad para el único consumidor identificado hoy (Daniel Gramoso) — si aparecen más consumidores, puede hacer falta reabrir esta decisión hacia una notificación push.

---

## 9. Preguntas Abiertas

Todas las preguntas quedaron resueltas en la revisión del 2026-09-04 (ver Historial de Cambios). No quedan abiertas.

| # | Pregunta | Responsable | Fecha límite | Respuesta |
|---|---------|-------------|-------------|-----------|
| 1 | Canal de notificación (email, Slack, otro) | Daniel Gramoso | Antes de Fase 2 | Log/archivo local, sin push activo |
| 2 | ¿La notificación de éxito lleva detalle por SKU o alcanza con agregado? | Daniel Gramoso | Antes de Fase 2 | Solo agregado |
| 3 | Si falla una etapa compartida por todos los SKUs (ej. `cargar_ventas()`), ¿se aborta toda la corrida sin persistir nada? | Daniel Gramoso | Antes de Fase 2 | Sí, se aborta sin persistir, sin reintento automático |
| 4 | Si el scheduler no está disponible en el momento programado, ¿se reintenta al volver, o se salta esa corrida? | Daniel Gramoso | Antes de Fase 2 | Se ejecuta apenas vuelve a estar disponible (catch-up automático) |
| 5 | ¿Hace falta distinguir en `corridas.parquet` una corrida programada de una manual, o alcanza con lo que ya persiste? | Daniel Gramoso | Antes de Fase 2 | Alcanza con lo que ya hay |
| 6 | Cadencia concreta (diaria/semanal/mensual/otra) | Daniel Gramoso | Antes de Fase 2 | Parámetro configurable, sin default fijo en este PRD |
| 7 | Infraestructura de scheduling (Task Scheduler, Airflow, cron, cloud) | Daniel Gramoso | Fase 2 (plan técnico) | Se resuelve en el plan, no es pregunta de PRD |

---

## 10. Checklist de Completitud

### Antes de marcar como APROBADO:
- [x] No quedan marcadores `[NECESITA CLARIFICACIÓN]` sin resolver
- [x] Todos los criterios de aceptación son testeables
- [x] Las métricas de éxito son medibles
- [x] Los casos borde críticos están documentados
- [x] Las dependencias externas están identificadas (ninguna nueva)
- [x] No hay especificación de implementación técnica (stack, código, APIs internas)
- [ ] Al menos 2 personas del equipo han revisado — pendiente, equipo de una sola persona; se deja sin marcar a propósito

---

## Historial de Cambios

| Versión | Fecha | Cambio | Autor |
|---------|-------|--------|-------|
| 0.1 | 2026-09-04 | Borrador inicial, vía `/diseno-proyecto` a partir de discovery con el usuario | Daniel Gramoso |
| 0.2 | 2026-09-04 | Resueltas las 6 preguntas abiertas (canal de notificación, nivel de detalle, falla de etapa compartida, disponibilidad del scheduler, distinción programada/manual, cadencia); estado pasa a APROBADO | Daniel Gramoso |
