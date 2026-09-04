<#
Registra la tarea de Windows Task Scheduler que dispara el reentrenamiento
programado del motor de forecast (specs/002-reentrenamiento-programado).

No se ejecuta automáticamente por nadie — correlo a mano cuando quieras
activar el reentrenamiento programado. Ver
specs/002-reentrenamiento-programado/operacion.md para el resto de la
operación (dónde queda el log, cómo forzar una corrida, cómo revertir).
#>

$NombreTarea = "MotorForecastVentas-ReentrenamientoProgramado"
$RaizRepo = Split-Path -Parent $PSScriptRoot

$accion = New-ScheduledTaskAction -Execute "python" -Argument "-m src.forecast.pipeline" -WorkingDirectory $RaizRepo

# Cadencia semanal — ajustar el día/hora al valor real que decidas.
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At 3am

# StartWhenAvailable = catch-up automático si la máquina estaba apagada.
# MultipleInstances IgnoreNew = no dispara una corrida nueva si la
# anterior sigue en curso (ver spec.md, Historia 1).
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $NombreTarea -Action $accion -Trigger $trigger -Settings $settings
