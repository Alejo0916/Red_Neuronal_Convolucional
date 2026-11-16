"""Configuración central del proyecto CNN sobre CIFAR-10.

Todas las rutas, clases y hyperparameters por defecto viven aquí para que el
notebook de Colab, el entrenamiento por terminal y la aplicación de Streamlit
compartan exactamente la misma definición.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

CHECKPOINT_PATH = MODELS_DIR / "cnn_cifar10.pt"
METADATA_PATH = MODELS_DIR / "metadata.json"
HISTORY_PATH = REPORTS_DIR / "history.json"
METRICS_PATH = REPORTS_DIR / "metrics.json"
PREDICTIONS_PATH = REPORTS_DIR / "predictions.json"

for _directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
DATASET_NAME = "CIFAR-10"
NUM_CLASSES = 10
IMG_SIZE = 32
IMG_CHANNELS = 3

#: Nombres de las clases de CIFAR-10 en el mismo orden que las etiquetas del dataset.
CLASSES: tuple[str, ...] = (
    "avión",
    "automóvil",
    "pájaro",
    "gato",
    "ciervo",
    "perro",
    "rana",
    "caballo",
    "barco",
    "camión",
)

#: Alias en inglés (etiquetas originales de torchvision) por si se necesitan.
CLASSES_EN: tuple[str, ...] = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

#: Descripción extendida del dataset, usada en la documentación.
DATASET_DESCRIPTION = (
    "CIFAR-10 (Krizhevsky, 2009) es un corpus de 60 000 imágenes RGB de 32x32 píxeles "
    "distribuidas en 10 clases de objetos: avión, automóvil, pájaro, gato, ciervo, perro, "
    "rana, caballo, barco y camión. Contiene 50 000 imágenes de entrenamiento y 10 000 "
    "de prueba, balanceadas con 6 000 / 1 000 imágenes por clase. Es uno de los "
    "benchmarks de referencia para Redes Neuronales Convolucionales."
)

DATASET_SOURCE = "https://www.cs.toronto.edu/~kriz/cifar.html"

# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# División de datos
# --------------------------------------------------------------------------- #
VAL_SIZE = 5_000
TEST_SIZE = 10_000
TRAIN_SIZE = 45_000

# --------------------------------------------------------------------------- #
# Arquitectura de la CNN
# --------------------------------------------------------------------------- #
MODEL_CONFIG: dict[str, object] = {
    "num_classes": NUM_CLASSES,
    "in_channels": IMG_CHANNELS,
    "width": 32,
    "channels": (64, 128, 256),
    "dropout2d": 0.15,
    "dropout_head": 0.50,
}

# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
TRAIN_CONFIG: dict[str, object] = {
    "epochs": 20,
    "batch_size": 128,
    "learning_rate": 1e-3,
    "weight_decay": 5e-4,
    "max_lr_ratio": 10.0,
    "grad_clip": 5.0,
    "num_workers": 2,
    "seed": SEED,
}

#: Estadísticas de normalización por canal (calculadas sobre el train split).
#: Se recalculan en `src.data.compute_channel_stats` y se guardan en los metadatos.
DEFAULT_MEAN: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
DEFAULT_STD: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "CHECKPOINT_PATH",
    "METADATA_PATH",
    "HISTORY_PATH",
    "METRICS_PATH",
    "PREDICTIONS_PATH",
    "DATASET_NAME",
    "DATASET_DESCRIPTION",
    "DATASET_SOURCE",
    "NUM_CLASSES",
    "IMG_SIZE",
    "IMG_CHANNELS",
    "CLASSES",
    "CLASSES_EN",
    "SEED",
    "VAL_SIZE",
    "TEST_SIZE",
    "TRAIN_SIZE",
    "MODEL_CONFIG",
    "TRAIN_CONFIG",
    "DEFAULT_MEAN",
    "DEFAULT_STD",
]
