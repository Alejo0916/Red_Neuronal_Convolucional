"""Evaluación del modelo: métricas, matriz de confusión, gráficas y predicciones.

Uso:
    python -m src.evaluate
    python -m src.evaluate --checkpoint models/cnn_cifar10.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from .config import (
    CHECKPOINT_PATH,
    CLASSES,
    FIGURES_DIR,
    METRICS_PATH,
    PREDICTIONS_PATH,
    PROJECT_ROOT,
)
from .data import build_dataloaders, denormalize
from .model import build_model
from .predict import predict_batch, save_predictions
from .utils import get_logger, load_json, resolve_device, save_json, set_seed

LOGGER = get_logger()


# --------------------------------------------------------------------------- #
# Inferencia sobre el conjunto de prueba
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (etiquetas reales, etiquetas predichas, probabilidades) del loader."""
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        images = images.contiguous(memory_format=torch.channels_last)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1)

        all_targets.append(labels.numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    return (
        np.concatenate(all_targets),
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def compute_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    """Accuracy, precision, recall y F1 macro + tabla por clase."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    accuracy = float(accuracy_score(targets, predictions))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions, average="macro", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        targets, predictions, average=None, labels=list(range(len(CLASSES))), zero_division=0
    )

    report = classification_report(
        targets,
        predictions,
        target_names=list(CLASSES),
        zero_division=0,
        output_dict=True,
    )

    per_class = {
        name: {
            "precision": float(per_class_precision[index]),
            "recall": float(per_class_recall[index]),
            "f1": float(per_class_f1[index]),
            "support": int(per_class_support[index]),
        }
        for index, name in enumerate(CLASSES)
    }

    return {
        "accuracy": accuracy,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "num_samples": int(len(targets)),
        "per_class": per_class,
        "classification_report": report,
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(len(CLASSES)))).tolist(),
    }


# --------------------------------------------------------------------------- #
# Figuras
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    matrix: np.ndarray,
    path: Path,
    title: str = "Matriz de confusión (conjunto de prueba)",
) -> Path:
    """Matriz de confusión con conteos y su versión normalizada por fila."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    normalized = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1, None)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for axis, values, subtitle, fmt in (
        (axes[0], matrix, "Conteos absolutos", "{:.0f}"),
        (axes[1], normalized, "Normalizada por fila (recall)", "{:.2f}"),
    ):
        image = axis.imshow(values, cmap="Blues", interpolation="nearest", vmin=0)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_title(subtitle, fontsize=12, fontweight="bold")
        axis.set_xticks(np.arange(len(CLASSES)))
        axis.set_yticks(np.arange(len(CLASSES)))
        axis.set_xticklabels(CLASSES, rotation=45, ha="right", fontsize=9)
        axis.set_yticklabels(CLASSES, fontsize=9)
        axis.set_xlabel("Clase predicha")
        axis.set_ylabel("Clase real")
        threshold = values.max() / 2.0 if values.max() > 0 else 0.5
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                axis.text(
                    j,
                    i,
                    fmt.format(values[i, j]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if values[i, j] > threshold else "black",
                )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Matriz de confusión guardada en %s", path)
    return path


def plot_training_curves(history: list[dict[str, float]], path: Path) -> list[Path]:
    """Curvas de pérdida y accuracy de entrenamiento y validación."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [record["epoch"] for record in history]

    paths = []
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(epochs, [record["train_loss"] for record in history], marker="o", label="Entrenamiento")
    axis.plot(epochs, [record["val_loss"] for record in history], marker="s", label="Validación")
    axis.set_xlabel("Época")
    axis.set_ylabel("Pérdida (CrossEntropy)")
    axis.set_title("Evolución de la pérdida", fontsize=13, fontweight="bold")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    loss_path = path.parent / "04_perdida_entrenamiento.png"
    fig.savefig(loss_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(loss_path)

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(epochs, [record["train_accuracy"] for record in history], marker="o", label="Entrenamiento")
    axis.plot(epochs, [record["val_accuracy"] for record in history], marker="s", label="Validación")
    axis.set_xlabel("Época")
    axis.set_ylabel("Accuracy")
    axis.set_title("Evolución de la exactitud", fontsize=13, fontweight="bold")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    acc_path = path.parent / "05_accuracy_entrenamiento.png"
    fig.savefig(acc_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    paths.append(acc_path)

    LOGGER.info("Curvas de entrenamiento guardadas en %s", [str(p) for p in paths])
    return paths


def plot_sample_predictions(
    images: torch.Tensor,
    targets: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    path: Path,
    mean: Sequence[float],
    std: Sequence[float],
    count: int = 12,
) -> Path:
    """Rejilla con la imagen, la clase real, la predicción y su probabilidad."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    correct = predictions == targets
    correct_indices = np.flatnonzero(correct)
    wrong_indices = np.flatnonzero(~correct)

    selected: list[int] = []
    half = count // 2
    selected.extend(correct_indices[:half].tolist())
    selected.extend(wrong_indices[: count - len(selected)].tolist())
    selected.extend(correct_indices[half : half + max(0, count - len(selected))].tolist())

    columns = 4
    rows = int(np.ceil(len(selected) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3.1, rows * 3.5))
    axes = np.atleast_1d(axes).ravel()

    for axis, index in zip(axes, selected):
        image = denormalize(images[index], mean, std)
        axis.imshow(image)
        real = CLASSES[int(targets[index])]
        predicted = CLASSES[int(predictions[index])]
        confidence = float(probabilities[index][int(predictions[index])])
        color = "seagreen" if real == predicted else "crimson"
        axis.set_title(
            f"Real: {real}\nPred: {predicted} ({confidence:.1%})",
            fontsize=8.5,
            color=color,
            fontweight="bold",
        )
        axis.axis("off")

    for axis in axes[len(selected) :]:
        axis.axis("off")

    fig.suptitle(
        f"Predicciones de ejemplo — {len(selected)} imágenes del conjunto de prueba",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Rejilla de predicciones guardada en %s", path)
    return path


# --------------------------------------------------------------------------- #
# Pipeline de evaluación
# --------------------------------------------------------------------------- #
def evaluate(
    checkpoint_path: Path = CHECKPOINT_PATH,
    metrics_path: Path = METRICS_PATH,
    device: str = "auto",
    seed: int = 42,
    num_workers: int = 2,
) -> dict[str, object]:
    """Evalúa el checkpoint sobre el conjunto de prueba y genera todas las figuras."""
    from .predict import load_model

    set_seed(seed)
    resolved_device = resolve_device(device)
    LOGGER.info("Evaluando %s en %s", checkpoint_path, resolved_device)

    model, metadata = load_model(checkpoint_path, device=resolved_device)

    bundle = build_dataloaders(num_workers=num_workers, seed=seed)
    targets, predictions, probabilities = collect_predictions(model, bundle.test_loader, resolved_device)

    metrics = compute_metrics(targets, predictions)
    LOGGER.info(
        "Test -> accuracy %.4f | precision %.4f | recall %.4f | f1_macro %.4f",
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
    )

    matrix = np.asarray(metrics["confusion_matrix"])
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    confusions = []
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j and off_diagonal[i, j] > 0:
                confusions.append(
                    {"real": CLASSES[i], "predicho": CLASSES[j], "casos": int(off_diagonal[i, j])}
                )
    confusions.sort(key=lambda item: item["casos"], reverse=True)

    figures: dict[str, str] = {}
    figures["matriz_confusion"] = str(
        plot_confusion_matrix(matrix, FIGURES_DIR / "06_matriz_confusion.png")
    )

    history_path = PROJECT_ROOT / "reports" / "history.json"
    if history_path.exists():
        history = load_json(history_path)
        curves = plot_training_curves(history, FIGURES_DIR / "07_curvas_entrenamiento.png")
        figures["curvas_entrenamiento"] = str(curves[0])

    sample_images = torch.stack([bundle.test_dataset[i][0] for i in range(len(bundle.test_dataset))])
    figures["predicciones_ejemplo"] = str(
        plot_sample_predictions(
            sample_images,
            targets,
            predictions,
            probabilities,
            FIGURES_DIR / "08_predicciones_ejemplo.png",
            mean=bundle.mean,
            std=bundle.std,
        )
    )

    payload = {
        "checkpoint": str(checkpoint_path),
        "device": str(resolved_device),
        "torch_version": torch.__version__,
        "model_config": metadata.get("model_config", {}),
        "classes": list(CLASSES),
        "metrics": metrics,
        "top_confusions": confusions[:10],
        "figures": figures,
    }
    save_json(payload, metrics_path)
    LOGGER.info("Métricas guardadas en %s", metrics_path)

    sample_records = []
    correct = predictions == targets
    chosen = list(np.flatnonzero(correct)[:6]) + list(np.flatnonzero(~correct)[:6])

    pil_images = []
    for index in chosen:
        array = denormalize(bundle.test_dataset[int(index)][0], bundle.mean, bundle.std)
        pil_images.append(Image.fromarray((array * 255).astype("uint8")))
    batch_predictions = predict_batch(
        model, pil_images, device=resolved_device, mean=bundle.mean, std=bundle.std
    )
    for index, prediction in zip(chosen, batch_predictions):
        sample_records.append(
            {
                "indice": int(index),
                "real": CLASSES[int(targets[index])],
                "predicho": prediction.class_name,
                "probabilidad": prediction.confidence,
                "acierto": bool(correct[index]),
            }
        )
    save_predictions(sample_records, PREDICTIONS_PATH)
    LOGGER.info("Predicciones de ejemplo guardadas en %s", PREDICTIONS_PATH)

    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evalúa la CNN sobre el conjunto de prueba")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluate(
        checkpoint_path=args.checkpoint,
        metrics_path=args.metrics,
        device=args.device,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_parser",
    "collect_predictions",
    "compute_metrics",
    "evaluate",
    "main",
    "plot_confusion_matrix",
    "plot_sample_predictions",
    "plot_training_curves",
]
