"""Pipeline de datos para CIFAR-10: carga, división, preprocesamiento y exploración.

Responsabilidades:
- descargar y cachear CIFAR-10 de forma idempotente;
- dividir el conjunto de entrenamiento oficial en *train* (45 000) y *validación*
  (5 000) de forma **estratificada** y reproducible;
- calcular las estadísticas de normalización **solo sobre el train split**;
- definir los transforms: augmentation únicamente en entrenamiento;
- construir los `DataLoader` reproducibles;
- generar la exploración visual del dataset (rejilla por clase, distribución y
  comparación antes/después del preprocesamiento).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .config import (
    CLASSES,
    DATA_DIR,
    FIGURES_DIR,
    IMG_SIZE,
    SEED,
    VAL_SIZE,
)
from .utils import get_logger, seed_worker

LOGGER = get_logger()


# --------------------------------------------------------------------------- #
# Carga del dataset
# --------------------------------------------------------------------------- #
def load_cifar10(
    root: str | Path = DATA_DIR,
    download: bool = True,
    train: bool = True,
    transform: transforms.Compose | None = None,
) -> CIFAR10:
    """Carga (y descarga si hace falta) CIFAR-10 con el transform opcional."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = CIFAR10(root=str(root), train=train, download=download, transform=transform)
    LOGGER.info(
        "CIFAR-10 cargado: %d imágenes (train=%s)", len(dataset), train
    )
    return dataset


def stratified_split(
    targets: Iterable[int],
    val_size: int = VAL_SIZE,
    seed: int = SEED,
) -> tuple[list[int], list[int]]:
    """Divide los índices del train set en train/val manteniendo la proporción de clases."""
    from sklearn.model_selection import StratifiedShuffleSplit

    targets_array = np.asarray(list(targets))
    indices = np.arange(len(targets_array))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, targets_array))
    return train_idx.tolist(), val_idx.tolist()


# --------------------------------------------------------------------------- #
# Preprocesamiento
# --------------------------------------------------------------------------- #
def compute_channel_stats(
    dataset: CIFAR10, indices: list[int]
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Calcula media y desviación estándar por canal usando solo `indices`."""
    LOGGER.info("Calculando estadísticas de normalización sobre %d imágenes...", len(indices))
    data = dataset.data[indices].astype(np.float64) / 255.0  # (N, H, W, C)
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    std = np.where(std < 1e-8, 1.0, std)
    return tuple(float(v) for v in mean), tuple(float(v) for v in std)


def build_transforms(
    mean: Sequence[float],
    std: Sequence[float],
) -> tuple[transforms.Compose, transforms.Compose, transforms.Compose]:
    """Devuelve los transforms de train (con augmentation), val y test (sin augmentation)."""
    normalize = transforms.Normalize(mean=list(mean), std=list(std))

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, eval_transform, eval_transform


def denormalize(
    tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """Convierte un tensor CHW normalizado en un array HWC con valores en [0, 1]."""
    array = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, -1)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, -1)
    return np.clip((array * std_array + mean_array), 0.0, 1.0)


@dataclass
class DataBundle:
    """Contiene los tres subconjuntos y sus `DataLoader`."""

    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    mean: tuple[float, ...]
    std: tuple[float, ...]
    train_indices: list[int]
    val_indices: list[int]
    class_counts: dict[str, dict[str, int]]

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "train": len(self.train_dataset),  # type: ignore[arg-type]
            "val": len(self.val_dataset),  # type: ignore[arg-type]
            "test": len(self.test_dataset),  # type: ignore[arg-type]
        }


def build_datasets(
    root: str | Path = DATA_DIR,
    seed: int = SEED,
    download: bool = True,
) -> tuple[Dataset, Dataset, Dataset, tuple[float, ...], tuple[float, ...], list[int], list[int]]:
    """Construye los tres subconjuntos junto con las estadísticas de normalización."""
    raw_train = load_cifar10(root=root, download=download, train=True)
    train_idx, val_idx = stratified_split(raw_train.targets, val_size=VAL_SIZE, seed=seed)
    mean, std = compute_channel_stats(raw_train, train_idx)

    train_transform, val_transform, test_transform = build_transforms(mean, std)

    train_full = load_cifar10(root=root, download=False, train=True, transform=train_transform)
    val_full = load_cifar10(root=root, download=False, train=True, transform=val_transform)
    test_set = load_cifar10(root=root, download=False, train=False, transform=test_transform)

    train_subset = Subset(train_full, train_idx)
    val_subset = Subset(val_full, val_idx)

    LOGGER.info("División: train=%d val=%d test=%d", len(train_subset), len(val_subset), len(test_set))
    LOGGER.info("Normalización: mean=%s std=%s", mean, std)
    return train_subset, val_subset, test_set, mean, std, train_idx, val_idx


def build_dataloaders(
    root: str | Path = DATA_DIR,
    batch_size: int = 128,
    num_workers: int = 4,
    seed: int = SEED,
    download: bool = True,
) -> DataBundle:
    """Construye datasets y `DataLoader` reproducibles para train/val/test."""
    train_ds, val_ds, test_ds, mean, std, train_idx, val_idx = build_datasets(
        root=root, seed=seed, download=download
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "generator": generator,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **common)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common)

    raw_train = load_cifar10(root=root, download=False, train=True)
    raw_test = load_cifar10(root=root, download=False, train=False)
    class_counts = _class_counts(raw_train.targets, train_idx, val_idx, raw_test.targets)

    return DataBundle(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        mean=mean,
        std=std,
        train_indices=train_idx,
        val_indices=val_idx,
        class_counts=class_counts,
    )


def _class_counts(
    train_targets: Sequence[int],
    train_idx: list[int],
    val_idx: list[int],
    test_targets: Sequence[int],
) -> dict[str, dict[str, int]]:
    """Recuento de imágenes por clase en cada subconjunto."""
    targets = np.asarray(train_targets)
    train_counts = np.bincount(targets[train_idx], minlength=len(CLASSES))
    val_counts = np.bincount(targets[val_idx], minlength=len(CLASSES))
    test_counts = np.bincount(np.asarray(test_targets), minlength=len(CLASSES))
    return {
        "train": {name: int(count) for name, count in zip(CLASSES, train_counts)},
        "val": {name: int(count) for name, count in zip(CLASSES, val_counts)},
        "test": {name: int(count) for name, count in zip(CLASSES, test_counts)},
    }


# --------------------------------------------------------------------------- #
# Exploración y visualización
# --------------------------------------------------------------------------- #
def plot_class_samples(dataset: CIFAR10, path: Path, samples_per_class: int = 8) -> Path:
    """Rejilla con `samples_per_class` imágenes de cada clase."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_class: dict[int, list[int]] = {}
    for index, label in enumerate(dataset.targets):
        by_class.setdefault(int(label), []).append(index)

    rows = len(CLASSES)
    fig, axes = plt.subplots(rows, samples_per_class, figsize=(samples_per_class * 1.35, rows * 1.5))
    for row in range(rows):
        indices = by_class.get(row, [])[:samples_per_class]
        for column in range(samples_per_class):
            axis = axes[row, column]
            axis.axis("off")
            if column < len(indices):
                axis.imshow(dataset.data[indices[column]])
                if column == 0:
                    axis.set_title(CLASSES[row], fontsize=9, loc="left", fontweight="bold")
    fig.suptitle("Muestras de CIFAR-10 por clase", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figura de muestras guardada en %s", path)
    return path


def plot_class_distribution(class_counts: dict[str, dict[str, int]], path: Path) -> Path:
    """Gráfico de barras con la distribución de clases de cada subconjunto."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    splits = ["train", "val", "test"]
    positions = np.arange(len(CLASSES))
    width = 0.26

    fig, axis = plt.subplots(figsize=(12, 5))
    for index, split in enumerate(splits):
        values = [class_counts[split][name] for name in CLASSES]
        axis.bar(positions + (index - 1) * width, values, width=width, label=split)
    axis.set_xticks(positions)
    axis.set_xticklabels(CLASSES, rotation=20, ha="right")
    axis.set_ylabel("Número de imágenes")
    axis.set_title("Distribución de clases por subconjunto", fontsize=13, fontweight="bold")
    axis.legend()
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figura de distribución guardada en %s", path)
    return path


def plot_preprocessing_comparison(
    dataset: CIFAR10,
    index: int,
    mean: Sequence[float],
    std: Sequence[float],
    path: Path,
) -> Path:
    """Compara la imagen original, el tensor en [0, 1] y el tensor normalizado."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    raw = dataset.data[index]
    tensor_01 = transforms.ToTensor()(raw)
    train_transform, _, _ = build_transforms(mean, std)
    normalized = train_transform(raw)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2))
    axes[0].imshow(raw)
    axes[0].set_title("Original (0-255)")
    axes[1].imshow(np.clip(tensor_01.numpy().transpose(1, 2, 0), 0, 1))
    axes[1].set_title("ToTensor() → [0, 1]")
    axes[2].imshow(denormalize(normalized, mean, std))
    axes[2].set_title(f"Normalizado + augmentation\nmean={tuple(round(v, 3) for v in mean)}")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"Preprocesamiento — clase real: {CLASSES[int(dataset.targets[index])]}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Figura de preprocesamiento guardada en %s", path)
    return path


def explore_dataset(root: str | Path = DATA_DIR, seed: int = SEED) -> dict[str, object]:
    """Genera toda la exploración visual e informacional del dataset."""
    raw_train = load_cifar10(root=root, download=True, train=True)
    raw_test = load_cifar10(root=root, download=False, train=False)
    train_idx, val_idx = stratified_split(raw_train.targets, val_size=VAL_SIZE, seed=seed)
    mean, std = compute_channel_stats(raw_train, train_idx)
    class_counts = _class_counts(raw_train.targets, train_idx, val_idx, raw_test.targets)

    figures = {
        "muestras_por_clase": plot_class_samples(raw_train, FIGURES_DIR / "01_muestras_por_clase.png"),
        "distribucion_clases": plot_class_distribution(
            class_counts, FIGURES_DIR / "02_distribucion_clases.png"
        ),
        "preprocesamiento": plot_preprocessing_comparison(
            raw_train, train_idx[0], mean, std, FIGURES_DIR / "03_preprocesamiento.png"
        ),
    }

    totals = {split: sum(counts.values()) for split, counts in class_counts.items()}
    balanced = all(
        max(counts.values()) - min(counts.values()) <= max(2, int(0.02 * totals[split]))
        for split, counts in class_counts.items()
    )

    report = {
        "dataset": "CIFAR-10",
        "num_clases": len(CLASSES),
        "clases": list(CLASSES),
        "tamano_imagen": [IMG_SIZE, IMG_SIZE, 3],
        "split": {"train": totals["train"], "val": totals["val"], "test": totals["test"]},
        "class_counts": class_counts,
        "balanceado": balanced,
        "media_normalizacion": list(mean),
        "desviacion_normalizacion": list(std),
        "figuras": {key: str(value) for key, value in figures.items()},
    }
    LOGGER.info("Dataset balanceado: %s", balanced)
    return report


__all__ = [
    "DataBundle",
    "build_dataloaders",
    "build_datasets",
    "build_transforms",
    "compute_channel_stats",
    "denormalize",
    "explore_dataset",
    "load_cifar10",
    "plot_class_distribution",
    "plot_class_samples",
    "plot_preprocessing_comparison",
    "stratified_split",
]
