# Especificación de Feature: Motor de Forecast de Ventas por SKU

**ID de Feature**: 001
**Rama**: `001-motor-forecast-sku`
**Fecha**: 2026-08-21
**Estado**: BORRADOR
**Autor**: Daniel Gramoso

---

## 1. Resumen Ejecutivo

**Problema que resuelve**: El equipo de planificación no tiene una predicción sistemática de demanda por producto/SKU, lo que dificulta decidir cuánto y cuándo comprar o producir.

**Solución propuesta**: Un motor que lee el histórico de ventas desde la base de datos/DWH, genera una predicción de demanda futura por SKU y la expone como API/servicio para que otros sistemas la consuman.

**Usuarios objetivo**: [NECESITA CLARIFICACIÓN: ¿quién consume el forecast — supply chain, compras, comercial, finanzas? El usuario indicó que todavía no está definido]

**Valor de negocio**: Mejorar la precisión de la planificación de stock/compras reduciendo quiebres y sobrestock por SKU.

---

## 2. Contexto y Motivación

### Situación actual
[NECESITA CLARIFICACIÓN: ¿cómo se planifica la demanda hoy? ¿Excel manual, otro modelo, sin proceso formal?]

### Por qué ahora
[NECESITA CLARIFICACIÓN: ¿qué evento o necesidad de negocio dispara este proyecto ahora?]

### Métricas de éxito
| Métrica | Valor actual | Objetivo | Plazo |
|---------|-------------|---------|-------|
| WAPE del modelo vs. benchmark Seasonal Naive (con o sin drift, según si la serie del SKU tiene tendencia) | WAPE del benchmark (a medir) | Superar consistentemente al benchmark en rolling backtesting: WAPE_modelo < WAPE_naive en [NECESITA CLARIFICACIÓN: ¿qué % de ventanas?] | [NECESITA CLARIFICACIÓN] |
| Bias del modelo (no debe empeorar sistemáticamente vs. benchmark) | [NECESITA CLARIFICACIÓN] | [NECESITA CLARIFICACIÓN] | [NECESITA CLARIFICACIÓN] |
| Cobertura de SKUs con forecast automático | 0% | [NECESITA CLARIFICACIÓN] | [NECESITA CLARIFICACIÓN] |

---

## 3. Alcance

### Incluido en esta versión (v1)
- Conexión de solo lectura a la base de datos/DWH para extraer histórico de ventas por SKU
- Generación de forecast de demanda futura por SKU
- Exposición del forecast como API/servicio (consumo por otros sistemas)
- [NECESITA CLARIFICACIÓN: ¿el horizonte y la frecuencia de actualización del forecast? Ej: semanal a 4-13 semanas, o mensual a 3-12 meses]

### Explícitamente EXCLUIDO
- Forecast a nivel cliente/cuenta o agregado (fuera de alcance de v1, es SKU-level)
- Interfaz de usuario / dashboard propio (el consumo es vía API, no UI en v1)
- [NECESITA CLARIFICACIÓN: enfoque de reentrenamiento — a discutir con el cliente. Dirección propuesta: disparado por deterioro del error (objetivo y medible) en vez de cadencia fija; falta definir métrica de referencia, umbral, ventana de evaluación, y si conviene combinarlo con una cadencia base programada para cubrir el lag entre disparo y daño ya ocurrido. Ver también posible mejora posterior al MVP.]

### Dependencias externas
- Base de datos/DWH con histórico de ventas por SKU (motor y credenciales: [NECESITA CLARIFICACIÓN])
- [NECESITA CLARIFICACIÓN: ¿sistemas consumidores conocidos de la API, para definir el contrato?]

---

## 4. Historias de Usuario

### Historia 1: Consultar forecast de un SKU vía API
```
Como sistema consumidor (ERP/planificación)
Quiero solicitar el forecast de ventas de un SKU específico vía API
Para incorporarlo a mi proceso de planificación de compras/stock
```

**Criterios de aceptación**:
- [ ] Dado un SKU válido con histórico suficiente, cuando se solicita el forecast vía API, entonces se retorna la predicción por período con su horizonte definido
- [ ] Dado un SKU sin histórico suficiente, cuando se solicita el forecast, entonces el sistema responde con un estado explícito de "sin datos suficientes" en vez de una predicción no confiable
- [ ] [NECESITA CLARIFICACIÓN: ¿la API responde el forecast pre-calculado (batch) o lo calcula on-demand en el momento del request?]

**Prioridad**: MUST
**Estimación de valor**: ALTA

---

### Historia 2: Actualización periódica del forecast
```
Como sistema de planificación
Quiero que el forecast se recalcule con el histórico más reciente
Para que las predicciones reflejen la demanda actual
```

**Criterios de aceptación**:
- [ ] Dado que hay nuevo histórico de ventas en la base de datos, cuando se ejecuta el proceso de actualización, entonces el forecast disponible vía API refleja los datos más recientes
- [ ] [NECESITA CLARIFICACIÓN: frecuencia de actualización — diaria, semanal, mensual]

**Prioridad**: MUST
**Estimación de valor**: ALTA

---

### Historia 3: Consultar forecast para múltiples SKUs
```
Como sistema consumidor
Quiero solicitar el forecast de un conjunto de SKUs (o de todos) en una sola operación
Para evitar múltiples llamadas individuales en procesos batch
```

**Criterios de aceptación**:
- [ ] Dado un listado de SKUs, cuando se solicita el forecast batch vía API, entonces se retorna la predicción de cada SKU solicitado
- [ ] [NECESITA CLARIFICACIÓN: ¿hay límite de cantidad de SKUs por request?]

**Prioridad**: SHOULD
**Estimación de valor**: MEDIA

---

## 5. Requisitos No Funcionales

### Rendimiento
- [NECESITA CLARIFICACIÓN: tiempo de respuesta esperado de la API y volumen de requests concurrentes]

### Seguridad
- Acceso a la API restringido a sistemas/consumidores autorizados
- [NECESITA CLARIFICACIÓN: mecanismo de autenticación — API key, OAuth, red interna]
- Credenciales de conexión a la base de datos/DWH gestionadas de forma segura (fuera del código)

### Disponibilidad
- [NECESITA CLARIFICACIÓN: ¿la API es crítica para un proceso en tiempo real, o tolera ventanas de mantenimiento?]

### Escalabilidad
- [NECESITA CLARIFICACIÓN: cantidad total de SKUs a soportar en v1]

### Observabilidad
- Logs de cada actualización de forecast (éxito/error, SKUs procesados)
- Métrica de error de forecast (MAPE/WAPE) accesible para monitoreo de calidad del modelo

---

## 6. Casos Borde y Escenarios de Error

| Escenario | Comportamiento esperado |
|-----------|------------------------|
| SKU nuevo sin histórico suficiente | La API responde estado explícito de "sin datos suficientes", no una predicción |
| SKU discontinuado / sin ventas recientes | [NECESITA CLARIFICACIÓN: ¿se excluye del forecast o se marca con demanda cero?] |
| Falla de conexión a la base de datos/DWH | El servicio degrada devolviendo el último forecast válido calculado, con indicador de "desactualizado" |
| SKU inexistente solicitado vía API | La API responde error 404 / no encontrado |
| Histórico con outliers o valores negativos (devoluciones) | [NECESITA CLARIFICACIÓN: ¿tratamiento esperado de outliers/devoluciones en el histórico?] |

---

## 7. Experiencia de Usuario (sin diseño técnico)

### Flujo principal (happy path)
1. El proceso de actualización lee el histórico de ventas por SKU desde la base de datos/DWH
2. El sistema genera el forecast de demanda futura para cada SKU con histórico suficiente
3. El sistema publica los forecasts actualizados para consumo vía API
4. Un sistema consumidor solicita el forecast de uno o varios SKUs vía API
5. El sistema retorna la predicción por período dentro del horizonte definido

### Flujos alternativos
- **Si el SKU no tiene histórico suficiente**: la API informa explícitamente que no hay forecast disponible para ese SKU
- **Si falla la actualización del forecast**: el sistema mantiene disponible el último forecast válido y registra el error

---

## 8. Restricciones y Suposiciones

### Restricciones
- El motor debe conectarse a la base de datos/DWH existente en modo solo lectura (no debe escribir sobre las tablas fuente)
- [NECESITA CLARIFICACIÓN: ¿restricciones de infraestructura para desplegar la API — on-premise, cloud, proveedor específico?]

### Suposiciones
- La base de datos/DWH contiene histórico de ventas por SKU con calidad y granularidad suficiente para modelar
- Existe al menos un sistema consumidor identificado que integrará contra la API en v1

---

## 9. Preguntas Abiertas

| # | Pregunta | Responsable | Fecha límite | Respuesta |
|---|---------|-------------|-------------|-----------|
| 1 | ¿Quién es el usuario/consumidor final del forecast (rol y sistema)? | Daniel Gramoso | Pendiente | PENDIENTE |
| 2 | ¿Cuál es el horizonte y la frecuencia de actualización del forecast (ej. semanal a 4-13 semanas vs. mensual a 3-12 meses)? | Daniel Gramoso | Pendiente | PENDIENTE |
| 3 | ¿Qué motor de base de datos/DWH y cómo se accede (credenciales, red)? | Daniel Gramoso | Pendiente | PENDIENTE |
| 4 | ¿La API calcula on-demand o sirve forecasts pre-calculados en batch? | Daniel Gramoso | Pendiente | PENDIENTE |
| 5 | ¿Cuántos SKUs debe soportar el motor en v1 (volumen)? | Daniel Gramoso | Pendiente | PENDIENTE |
| 6 | ¿Cómo se autentican los consumidores de la API? | Daniel Gramoso | Pendiente | PENDIENTE |

---

## 10. Checklist de Completitud

### Antes de marcar como APROBADO:
- [ ] No quedan marcadores `[NECESITA CLARIFICACIÓN]` sin resolver
- [ ] Todos los criterios de aceptación son testeables
- [ ] Las métricas de éxito son medibles
- [ ] Los casos borde críticos están documentados
- [ ] Las dependencias externas están identificadas
- [ ] No hay especificación de implementación técnica (stack, código, APIs internas)
- [ ] Al menos 2 personas del equipo han revisado

---

## Historial de Cambios

| Versión | Fecha | Cambio | Autor |
|---------|-------|--------|-------|
| 0.1 | 2026-08-21 | Borrador inicial | Daniel Gramoso |
