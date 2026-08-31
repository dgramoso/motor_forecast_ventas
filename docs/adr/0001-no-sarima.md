# No usar SARIMA como candidato en el pipeline masivo

SARIMA competía como candidato en `comparar_modelos` (`modelo_sarima.py`), pero se decidió sacarlo: en un pipeline que corre sobre miles de SKUs, el orden (p,d,q)(P,D,Q) se recalcula por grilla de AIC en cada ventana y por cada serie, lo que es lento y frágil de auditar a escala, y se degrada mal con series cortas o intermitentes (el caso intermitente ya lo cubre `ets_tsb` vía TSB, ver `modelo_intermitente.py`). SARIMA sigue siendo una opción válida para análisis puntuales ad-hoc sobre una serie individual, fuera de este pipeline — solo se eliminó como candidato competitivo automatizado.

## Considered Options

- Mantener SARIMA como candidato de bajo peso (grilla más chica): igual requiere ajuste por SKU y no resuelve la fragilidad ante intermitencia.
- Sacarlo del pipeline y dejarlo disponible solo para uso manual/exploratorio: opción elegida.
