# Operación: Reentrenamiento Programado

**ID de Feature**: 002
**Referencias**: `spec.md`, `plan.md` (sección 10), `tasks.md` (Grupo Final)

---

## Crear la tarea programada

```powershell
.\scripts\registrar_tarea_programada.ps1
```

Registra `MotorForecastVentas-ReentrenamientoProgramado` en Windows Task
Scheduler: cadencia semanal (domingo 3am por defecto — editar el trigger
en el script antes de correrlo si hace falta otro día/hora), catch-up
automático si la máquina estaba apagada, y sin solape con una corrida
anterior todavía en curso.

Para cambiar la cadencia después de creada, no edites el script — editá
el trigger directamente en Task Scheduler (`taskschd.msc`) o volvé a
correr `Unregister-ScheduledTask` + el script con el trigger ajustado.

## Forzar una corrida manual (para validar)

```powershell
Start-ScheduledTask -TaskName "MotorForecastVentas-ReentrenamientoProgramado"
```

Corre el pipeline completo fuera de horario, igual que si hubiera
disparado la cadencia programada.

## Dónde queda el resultado

- **Log de la corrida**: `logs/corridas_programadas.log` — una línea
  `INFO` con el resumen agregado (SKUs procesados, tasa de fallback
  promedio, distribución de candidatos ganadores) si terminó bien, o una
  línea `ERROR` con traceback si abortó por una falla de etapa
  compartida (ver spec.md, sección 6).
- **Datos persistidos**: `data/runs/corridas.parquet` y
  `data/runs/pronosticos.parquet` — un `run_id` nuevo por corrida
  exitosa. Nada se agrega ahí si la corrida abortó.
- **Historial nativo de Task Scheduler**: `taskschd.msc` → la tarea →
  pestaña "History" — señal redundante del mismo resultado (exit code
  del proceso), útil si el log de la corrida no llegó a escribirse por
  algún motivo previo al propio pipeline (ej. Python no encontrado).

## Cómo distinguir éxito de falla

```powershell
Get-Content logs\corridas_programadas.log -Tail 20
```

- Línea que empieza con `INFO` y menciona "SKUs procesados" → corrida
  exitosa, ver `data/runs/corridas.parquet` para el detalle.
- Línea que empieza con `ERROR` → corrida abortada antes de persistir
  nada. El resto de esa línea (traceback) indica qué etapa compartida
  falló (`cargar_ventas()`, entrenamiento de LightGBM global, etc.).
- Un SKU puntual que rompió con algo no contemplado por el fallback
  existente no genera una línea separada de "falla de la corrida": queda
  como `WARNING` de `src.forecast.comparar_modelos` o
  `src.forecast.ensemble_backtest` en el mismo log, y el resto de la
  corrida sigue — la corrida general sigue contando como éxito.

## Rollback

Deshabilitar o borrar la tarea vuelve al estado 100% manual, sin tocar
código ni datos ya persistidos:

```powershell
# Pausar sin borrar (se puede reactivar después)
Disable-ScheduledTask -TaskName "MotorForecastVentas-ReentrenamientoProgramado"

# Borrar del todo
Unregister-ScheduledTask -TaskName "MotorForecastVentas-ReentrenamientoProgramado" -Confirm:$false
```

Ninguna de las dos opciones afecta `data/runs/*.parquet` ni el log ya
escrito — el pipeline sigue disponible para correr a mano
(`python -m src.forecast.pipeline`) igual que antes de esta feature.
