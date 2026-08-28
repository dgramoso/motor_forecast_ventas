# Dataset sintético de ventas

Generado por [`src/datos/generar_datos_sinteticos.py`](../../src/datos/generar_datos_sinteticos.py). Simula la extracción mensual de ventas por SKU que en producción vendría del DWH (spec.md:54, todavía sin definir). Se usa como sandbox de desarrollo — se trata como si fuera el histórico real hasta que haya acceso a la base de datos del cliente.

## Archivos

- `ventas_historicas.csv` — `sku_id`, `fecha` (mensual, `MS`), `unidades_vendidas`. 5 SKUs × 60 meses (2021-01 a 2025-12).
- `skus.csv` — catálogo de los 5 SKUs con su patrón de demanda.

## Patrones por SKU

| SKU | Patrón | Por qué |
|-----|--------|---------|
| SKU-001 | Tendencia creciente + estacionalidad marcada | Caso "fácil": ambos componentes presentes y limpios |
| SKU-002 | Estacionalidad marcada, sin tendencia | Aísla si el modelo captura estacionalidad sin confundirla con tendencia |
| SKU-003 | Bajo volumen / demanda intermitente | Ejercita el caso de histórico ruidoso/discontinuo (spec.md:70, "sin datos suficientes") |
| SKU-004 | Nivel + estacionalidad leve, con picos de promoción y devoluciones (valores negativos) | Ejercita el tratamiento de outliers/devoluciones (spec.md:140) |
| SKU-005 | Nivel estable con quiebre estructural (caída fuerte en el mes 42) | Simula un SKU discontinuado a mitad de serie (spec.md:137) |

## Regenerar

```bash
python src/datos/generar_datos_sinteticos.py
```

Semilla fija (42) — reproducible. Ejecutar solo si se cambia el generador.
