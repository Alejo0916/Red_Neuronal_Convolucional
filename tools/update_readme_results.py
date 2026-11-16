"""Actualiza el bloque de resultados del README a partir de `reports/metrics.json`.

Uso:
    python tools/update_readme_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README_PATH = PROJECT_ROOT / "README.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
HISTORY_PATH = PROJECT_ROOT / "reports" / "history.json"

INICIO = "<!-- RESULTADOS:METRICAS -->"
FIN = "<!-- RESULTADOS:FIN -->"


def construir_bloque(metrics: dict, history: list[dict]) -> str:
    """Construye el bloque Markdown con los resultados reales."""
    met = metrics["metrics"]
    por_clase = met["per_class"]
    confusiones = metrics.get("top_confusions", [])[:5]
    epocas = len(history)
    mejor = max(history, key=lambda registro: registro["val_accuracy"])
    ultima = history[-1]

    lineas_clase = [
        f"| {nombre} | {datos['precision']:.4f} | {datos['recall']:.4f} | {datos['f1']:.4f} | {datos['support']:,} |"
        for nombre, datos in por_clase.items()
    ]
    lineas_confusion = [
        f"| {item['real']} | {item['predicho']} | {item['casos']} |" for item in confusiones
    ]

    return f"""
### Métricas en el conjunto de prueba (10 000 imágenes)

| Métrica | Valor |
|---|---:|
| **Accuracy** | **{met['accuracy']:.4f}** ({met['accuracy']:.2%}) |
| **Precision** (macro) | {met['precision_macro']:.4f} |
| **Recall** (macro) | {met['recall_macro']:.4f} |
| **F1** (macro) | {met['f1_macro']:.4f} |
| Imágenes evaluadas | {met['num_samples']:,} |

Mejor época de validación: **época {mejor['epoch']}** con accuracy **{mejor['val_accuracy']:.4f}**
(pérdida de validación {mejor['val_loss']:.4f}). Entrenamiento completo: {epocas} épocas.

### Métricas por clase

| Clase | Precision | Recall | F1 | Soporte |
|---|---:|---:|---:|---:|
{chr(10).join(lineas_clase)}

### Pares de clases más confundidos

| Clase real | Clase predicha | Casos |
|---|---|---:|
{chr(10).join(lineas_confusion)}

### Curvas del entrenamiento

| Pérdida | Accuracy |
|---|---|
| ![Curva de pérdida](reports/figures/04_perdida_entrenamiento.png) | ![Curva de accuracy](reports/figures/05_accuracy_entrenamiento.png) |

### Matriz de confusión

![Matriz de confusión](reports/figures/06_matriz_confusion.png)

### Predicciones de ejemplo

![Predicciones de ejemplo](reports/figures/08_predicciones_ejemplo.png)

### Exploración del dataset

![Muestras por clase](reports/figures/01_muestras_por_clase.png)

![Distribución de clases](reports/figures/02_distribucion_clases.png)

![Preprocesamiento](reports/figures/03_preprocesamiento.png)
""".strip()


def main() -> int:
    if not METRICS_PATH.exists():
        print(f"No existe {METRICS_PATH}; ejecuta primero `python -m src.evaluate`.")
        return 1

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    history = json.loads(HISTORY_PATH.read_text(encoding="utf-8")) if HISTORY_PATH.exists() else []
    contenido = README_PATH.read_text(encoding="utf-8")

    if INICIO not in contenido or FIN not in contenido:
        print("El README no contiene los marcadores de resultados.")
        return 1

    bloque = construir_bloque(metrics, history)
    antes, _, despues = contenido.partition(INICIO)
    _, _, despues = despues.partition(FIN)
    README_PATH.write_text(f"{antes}{INICIO}\n\n{bloque}\n\n{FIN}{despues}", encoding="utf-8")
    print(f"README actualizado con los resultados (accuracy {metrics['metrics']['accuracy']:.4f}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
