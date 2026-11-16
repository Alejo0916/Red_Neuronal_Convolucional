"""Exporta imágenes de ejemplo del dataset a `app/assets/examples/` para la app.

Además de las imágenes genera `manifest.json`, que asocia cada archivo con su
clase verdadera. Esto evita depender de nombres de archivo con acentos.

Uso:
    python tools/export_examples.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.config import CLASSES, DATA_DIR
from src.data import load_cifar10

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "app" / "assets" / "examples"

#: Clases y cantidad de ejemplos a exportar.
SELECTION = ["avión", "automóvil", "gato", "perro", "barco", "camión"]


def main() -> None:
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_cifar10(root=DATA_DIR, download=False, train=False)

    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(dataset.targets):
        by_class.setdefault(int(label), []).append(index)

    manifest: list[dict[str, object]] = []
    for class_name in SELECTION:
        class_index = CLASSES.index(class_name)
        indices = by_class.get(class_index, [])[:2]
        for number, index in enumerate(indices, start=1):
            image = Image.fromarray(dataset.data[index]).convert("RGB")
            filename = f"{class_index:02d}_{number}.png"
            image.save(EXAMPLES_DIR / filename)
            manifest.append(
                {
                    "file": filename,
                    "class_index": class_index,
                    "class_name": class_name,
                    "dataset_index": int(index),
                }
            )
            print(f"  {filename}  (clase real: {class_name})")

    (EXAMPLES_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"{len(manifest)} imágenes exportadas a {EXAMPLES_DIR} (+ manifest.json)")


if __name__ == "__main__":
    main()
