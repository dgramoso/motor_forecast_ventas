"""CLI: imprime la tabla comparativa de benchmark vs. modelo, por SKU,
sobre el dataset sintético. La lógica reusable vive en comparar_modelos.py.
"""

from src.datos.cargar_datos import cargar_ventas
from src.forecast.comparar_modelos import comparar_modelos


def main() -> None:
    ventas = cargar_ventas()
    tabla = comparar_modelos(ventas).round(3)
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
