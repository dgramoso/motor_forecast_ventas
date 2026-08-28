Type: task
Status: resolved
Blocked by: 03, 04, 05, 06

## Question

Armar el pipeline único y ejecutable que encadena: ingesta (`src/datos/cargar_datos.py`) → ejecución y comparación de modelos (ticket 03) → selección del mejor modelo por SKU (ticket 04) → generación del pronóstico futuro (ticket 05) → persistencia de pronósticos y corrida (ticket 06).

Debe correr de punta a punta sobre los 5 SKUs sintéticos con un solo comando/función, y dejar como resultado observable: qué modelo se eligió por SKU, el pronóstico generado, y dónde quedó persistido.

## Answer

Creado `src/forecast/pipeline.py`:
- `ejecutar_pipeline(horizonte, ventana_minima)` — encadena ingesta (`cargar_ventas`) → `comparar_modelos` (03) → `seleccionar_mejor_modelo` (04) → `pronosticar_futuro` (05) → `guardar_corrida`/`guardar_pronosticos` (06). Devuelve run_id, tabla comparativa, selecciones y pronóstico.
- `main()` — CLI (`python -m src.forecast.pipeline`) que corre todo y muestra el resumen: modelo elegido por SKU, pronóstico futuro, y dónde quedó persistido.

Verificado end-to-end sobre los 5 SKUs sintéticos: corre sin errores, cada corrida genera un run_id nuevo y se persiste sin pisar corridas anteriores.

