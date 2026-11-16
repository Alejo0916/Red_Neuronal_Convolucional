"""Muestra un resumen de las salidas del notebook ejecutado (para verificación).

Uso:
    python tools/notebook_summary.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "experimentacion_cifar10_cnn.ipynb"

PALABRAS = (
    "Accuracy",
    "accuracy",
    "Parámetros entrenables",
    "División",
    "Reproducible",
    "Pesos cargados",
    "Mejor época",
    "Real:",
    "Mostradas",
    "Entrenamiento",
    "Desviación",
    "Forma del tensor",
)


def main() -> int:
    salida = io.StringIO()
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    imagenes = sum(
        1
        for celda in notebook.cells
        for salida_celda in celda.get("outputs", [])
        if "image/png" in salida_celda.get("data", {})
    )
    lineas = [
        f"Celdas: {len(notebook.cells)} "
        f"(código: {sum(1 for c in notebook.cells if c.cell_type == 'code')})",
        f"Imágenes embebidas: {imagenes}",
        "",
    ]
    for indice, celda in enumerate(notebook.cells):
        if celda.cell_type != "code":
            continue
        for salida_celda in celda.get("outputs", []):
            if salida_celda.get("output_type") != "stream":
                continue
            texto = salida_celda.get("text", "")
            lineas.append(f"--- celda {indice} ---")
            lineas.extend(texto.rstrip().splitlines()[:14])
            lineas.append("")

    salida.write("\n".join(lineas))
    sys.stdout.buffer.write(salida.getvalue().encode("utf-8", errors="replace"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
