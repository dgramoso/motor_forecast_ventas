# Agregar 4 modelos candidatos: SARIMA, XGBoost, Prophet, Random Forest

## Contexto

El Paso 2 del pipeline (`comparar_modelos.py`) hoy compara dos candidatos:
`benchmark` (Seasonal Naive) y el router ETS/TSB. Se agregan 4 candidatos
nuevos para ampliar el ranking por SKU.

## Decisiones

### Arquitectura
- Los 4 modelos nuevos entran como **candidatos independientes** en
  `CANDIDATOS`, no dentro del router — compiten por WAPE igual que hoy
  compiten `benchmark` vs. el router ETS/TSB.
- La clave del router se **renombra de `"modelo"` a `"ets_tsb"`** —
  `"modelo"` queda ambiguo apenas hay 5 candidatos más que también "son
  un modelo". Verificado que `data/runs/corridas.parquet` es output
  generado y gitignored (`map.md`), no hay corridas reales que se
  invaliden con el rename.
- Archivos nuevos, mismo patrón que `modelo_ets.py` / `modelo_intermitente.py`:
  `modelo_sarima.py`, `modelo_xgboost.py`, `modelo_prophet.py`,
  `modelo_random_forest.py`. Cada uno expone
  `pronosticar_<nombre>(serie, horizonte) -> np.ndarray`.
- Los 4 siguen el mismo contrato defensivo que `modelo_ets.py`: ante
  datos degenerados en el ajuste, fallback a `pronosticar_seasonal_naive`
  con motivo registrado (mismo mecanismo que ya usa ETS —
  `_EXCEPCIONES_AJUSTE_ETS`).
- Walk-forward completo, sin recortes, mismos SKUs y ventanas para los 6
  candidatos — no se optimiza runtime de forma especulativa; si el tiempo
  de corrida se vuelve un problema real, se revisita con datos concretos.
- Sin autotuning de hiperparámetros — se descartó explícitamente al
  arrancar esta conversación.

### SARIMA (`modelo_sarima.py`)
- Sin dependencia nueva: usa `SARIMAX` de `statsmodels` (ya en
  `requirements.txt`).
- Orden elegido por grilla de AIC, recalculada en cada ventana de
  entrenamiento (nunca con información posterior al origen, mismo
  criterio que `benchmark.estimar_tendencia`): `p,q ∈ {0,1,2}`,
  `d ∈ {0,1}`, `P,Q,D ∈ {0,1}`, `periodo=12` fijo (mismo
  `PERIODO_ESTACIONAL` que el resto del proyecto). No se prueban tests de
  raíz unitaria para elegir `d`/`D` — quedan como parte de la grilla.
- Se descartó `pmdarima` (`auto_arima`) por historial de incompatibilidad
  con versiones recientes de numpy/statsmodels, y por evitar una
  dependencia nueva pudiendo resolverlo con lo que ya está instalado.

### Prophet (`modelo_prophet.py`)
- Dependencia nueva: `prophet` (trae `cmdstanpy`/Stan).
- Config: `growth="linear"` (sin cap — no hay noción de techo de demanda
  en la spec), `yearly_seasonality=True`, `weekly_seasonality=False`,
  `daily_seasonality=False` (datos mensuales, no diarios),
  `seasonality_mode="additive"` (coherente con que ETS tampoco usa
  variantes multiplicativas en el MVP).

### XGBoost y Random Forest (`modelo_xgboost.py`, `modelo_random_forest.py`)
- Dependencias nuevas: `xgboost`; `scikit-learn` (ya estaba instalado en
  el entorno, se agrega igual a `requirements.txt` para que quede
  declarado).
- **Pronóstico directo**, no recursivo: 3 modelos independientes, uno por
  paso del horizonte (`HORIZONTE=3`) — evita que el error de t+1
  contamine t+2/t+3. Se aceptó el costo de 3x entrenamiento porque el
  horizonte es corto.
- Features fijas por fila: lags 1, 2, 3, 12 (últimos 3 meses + mismo mes
  año anterior, mismo `PERIODO_ESTACIONAL=12`), medias móviles de 3 y 12
  meses, mes calendario como dummy (mismo patrón que
  `benchmark.estimar_tendencia`, no codificación cíclica — un solo
  criterio de "cómo se representa el mes" en todo el proyecto).
- Precondición explícita: si tras construir los lags quedan menos de 12
  filas utilizables, se lanza `ValueError` a propósito para que caiga en
  el mismo mecanismo de fallback (a diferencia de `statsmodels`,
  `scikit-learn`/`xgboost` no fallan solos ante datos escasos, solo
  ajustan mal).
- Hiperparámetros fijos, sin tuning: `n_estimators=200`, `max_depth=3`
  (poca profundidad — pocas filas de entrenamiento, evitar sobreajuste
  inmediato); XGBoost además `learning_rate=0.1`.
- `random_state` fijo (semilla explícita) para que el backtest sea
  reproducible corrida a corrida — el resto del pipeline ya es
  determinístico dado el mismo histórico.

## Riesgos conocidos, no bloqueantes

- `prophet` requiere además instalar el binario de CmdStan (compilación
  aparte, no solo `pip install`) — puede ser lento y requiere toolchain
  de C++ en Windows. No hay venv de proyecto: la instalación es al
  Python global del sistema.
- Con 6 candidatos por SKU por ventana, `comparar_modelos()` va a tardar
  sensiblemente más que hoy — aceptado, ver "Walk-forward completo"
  arriba.

## Estado de validación (2026-08-28)

- `scikit-learn`, `xgboost`, `prophet` instalados vía pip en el Python
  global (no hay venv de proyecto).
- `python -m cmdstanpy.install_cmdstan` reportó un fallo de `make build`
  en un intento intermedio (faltaba `mingw32-make` en el PATH en ese
  momento), pero el proceso completo terminó instalando CmdStan 2.39.0
  igual (`cmdstanpy.cmdstan_path()` lo confirma).
- Los 6 candidatos (`benchmark`, `ets_tsb`, `sarima`, `xgboost`,
  `prophet`, `random_forest`) fueron validados end-to-end con
  `comparar_modelos_sku` sobre SKU-001 real — corren y devuelven
  resultados sin excepciones.
- Suite de tests existente (`tests/`, 23 tests con `unittest`) sigue
  pasando sin cambios — el rename de `"modelo"` a `"ets_tsb"` y los
  candidatos nuevos no la afectan (no hay tests de
  `comparar_modelos.py`/`pipeline.py` todavía).
