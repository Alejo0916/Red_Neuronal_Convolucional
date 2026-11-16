"""Comprueba si alguna celda del notebook tiene salidas de error.

Uso:
    python tools/check_notebook_errors.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "experimentacion_cifar10_cnn.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    fallos = 0
    for indice, celda in enumerate(notebook.cells):
        if celda.cell_type != "code":
            continue
        for salida in celda.get("outputs", []):
            if salida.get("output_type") == "error":
                fallos += 1
                print(f"=== ERROR en la celda {indice}: {salida.get('ename')}: {salida.get('evalue')}")
                print("\n".join(salida.get("traceback", [])[-12:]))
                print()
    total = sum(1 for celda in notebook.cells if celda.cell_type == "code")
    con_salida = sum(
        1
        for celda in notebook.cells
        if celda.cell_type == "code" and celda.get("outputs")
    )
    print(f"{total} celdas de código, {con_salida} con salida, {fallos} con error.")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
