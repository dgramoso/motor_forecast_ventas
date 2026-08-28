Type: task
Status: resolved
Blocked by: 02

## Question

Implementar el guardado de pronósticos y versiones de corrida según el esquema decidido en el ticket 02: formato de almacenamiento, campos por corrida, y política de versionado (inmutable vs. sobrescritura del vigente).

## Answer

Creado `src/forecast/persistencia.py` (requiere `pyarrow`, agregado a requirements.txt):
- `nuevo_run_id()` — uuid4 por corrida.
- `guardar_corrida(run_id, selecciones)` / `guardar_pronosticos(run_id, pronostico)` — append-only sobre `data/runs/corridas.parquet` y `data/runs/pronosticos.parquet`.
- `obtener_pronostico_vigente(sku_id=None)` — deriva el vigente como la corrida de mayor `timestamp_utc` por SKU (no hay tabla de "vigente" aparte, tal como se decidió en el ticket 02).

Verificado: dos corridas consecutivas acumulan (10 filas de corridas, 2 run_id distintos), no se pisan.

