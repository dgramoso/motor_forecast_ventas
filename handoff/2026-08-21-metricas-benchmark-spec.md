# Handoff — motor_forecast_ventas

**Fecha de cierre**: 2026-08-21
**Sesión**: definición de métricas de evaluación, benchmark y enfoque de reentrenamiento para el spec del MVP

---

## Dónde está el trabajo

Toda la decisión sustantiva quedó volcada en el spec, no acá — este documento es solo el resumen de sesión y lo que sigue.

- **Spec de referencia**: `specs/001-motor-forecast-sku/spec.md`
- Cambios de esta sesión: líneas 34-35 (métricas de éxito) y línea 51 (enfoque de reentrenamiento)
- No es un repo git todavía (se inicializa en esta misma sesión de cierre, ver más abajo)

## Decisiones tomadas en esta sesión

1. **Métricas de evaluación**: WAPE (principal, precisión ponderada por volumen) + Bias (principal, detecta sesgo direccional que WAPE solo no ve) + MAE (complementaria, unidades absolutas). Se descarta MAPE (inestable con `real≈0`, pondera igual SKUs chicos y grandes) y R² (mide ajuste vs. varianza histórica, no error operativo; se infla con estacionalidad/tendencia).
2. **Benchmark**: Seasonal Naive como piso de comparación — con drift si la serie tiene tendencia, sin drift si no. Criterio de éxito del MVP: superar consistentemente al benchmark en rolling backtesting (walk-forward, no split único ni k-fold, para evitar leakage temporal).
3. **Reentrenamiento**: dirección propuesta es "disparado por deterioro del error" (objetivo y medible) en vez de cadencia fija — pero queda **explícitamente pendiente de discutir con el cliente**, con la observación de que un enfoque puramente reactivo puede combinarse con una cadencia base programada para no depender solo del trigger.

Todo el razonamiento detrás de estas tres decisiones (por qué WAPE y no MAPE, por qué no R², qué es Seasonal Naive con drift, por qué el benchmark importa aunque "un modelo ajustado debería ganarle siempre a uno naive") está en la conversación de esta sesión — no se duplicó en el spec, solo las conclusiones.

## Qué falta definir (todos son `[NECESITA CLARIFICACIÓN]` en el spec)

Prioridad alta — bloquean poder ejecutar el criterio de éxito:
- % de ventanas del rolling backtest que el modelo debe superar al benchmark para considerar "consistentemente" (spec.md:34)
- Objetivo/umbral de Bias (spec.md:35)

Pendiente de conversación con el cliente:
- Enfoque de reentrenamiento: métrica de referencia, umbral, ventana de evaluación, si se combina con cadencia base (spec.md:51)
- Usuarios/consumidor final del forecast (spec.md:17, pregunta abierta #1)
- Horizonte y frecuencia de actualización (spec.md:46, pregunta abierta #2)
- Motor de BD/DWH y acceso (spec.md:54, pregunta abierta #3)
- On-demand vs. batch pre-calculado (spec.md:71, pregunta abierta #4)
- Volumen de SKUs a soportar (spec.md:124, pregunta abierta #5)
- Autenticación de la API (spec.md:117, pregunta abierta #6)

Secundarios (no bloquean el MVP conceptual, pero hay que cerrarlos antes de aprobar el spec):
- Tratamiento de SKU discontinuado (spec.md:137)
- Tratamiento de outliers/devoluciones en el histórico (spec.md:140)
- Infraestructura de despliegue (spec.md:163)
- Situación actual de planificación / por qué ahora (spec.md:26, 29)

Ver checklist de completitud en `specs/001-motor-forecast-sku/spec.md:184-193` — no está aprobable hasta resolver los `[NECESITA CLARIFICACIÓN]` pendientes.

## Próxima sesión — foco sugerido

El usuario todavía no indicó el foco exacto de la próxima sesión más allá de "seguir revisando" el spec. Arrancar por:
1. Repasar el resumen que se dio al usuario al final de esta sesión (definido vs. pendiente).
2. Resolver el % de ventanas del backtest y el umbral de Bias (bloquean el criterio de éxito).
3. Preparar preguntas concretas para el cliente sobre reentrenamiento, usuarios y horizonte — son decisiones de negocio, no técnicas.

## Skills sugeridas para continuar

- **`diseno-proyecto`** — para seguir completando el spec vía SDD (ya es el patrón que se venía usando en esta sesión, aunque invocado implícitamente).
- **`domain-modeling`** — cuando la terminología del proyecto (WAPE, benchmark, deterioro del error, etc.) se estabilice, para crear `CONTEXT.md` y empezar a registrar ADRs (ej. "por qué Seasonal Naive con drift como benchmark", "por qué trigger de deterioro sobre cadencia fija"). Todavía no existe `CONTEXT.md` en el repo — se crea recién cuando el domain-modeling skill lo haga, no antes.
- **`analytics-workflow`** — para cuando el proyecto pase de fase de diseño/spec a EDA y construcción real del modelo de forecast.

## Notas de entorno

- El repo no tenía `.git` al momento de este handoff.
