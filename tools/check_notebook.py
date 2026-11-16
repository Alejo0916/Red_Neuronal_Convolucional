"""Comprueba la sintaxis de cada celda de código del notebook generado.

Uso:
    python tools/check_notebook.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "experimentacion_cifar10_cnn.ipynb"


def main() -> int:
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    errores = 0
    for indice, celda in enumerate(notebook.cells):
        if celda.cell_type != "code":
            continue
        fuente = celda.source
        # Las celdas de Colab empiezan con comentarios de título: se eliminan.
        lineas = [linea for linea in fuente.splitlines() if not linea.startswith("# @title")]
        codigo = "\n".join(lineas)
        try:
            ast.parse(codigo)
        except SyntaxError as error:
            errores += 1
            print(f"[celda {indice}] SyntaxError: {error}")
            numero = error.lineno or 1
            contexto = codigo.splitlines()[max(0, numero - 3) : numero + 2]
            for linea in contexto:
                print(f"    {linea}")
    total = sum(1 for celda in notebook.cells if celda.cell_type == "code")
    print(f"{total} celdas de código comprobadas, {errores} con errores de sintaxis.")
    return 1 if errores else 0


if __name__ == "__main__":
    sys.exit(main())
