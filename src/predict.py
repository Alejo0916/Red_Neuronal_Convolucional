"""Inferencia con el modelo entrenado: carga del checkpoint y predicción de imágenes.

Este módulo es la única dependencia de la aplicación de Streamlit: no importa
el pipeline de entrenamiento ni el dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from .config import (
    CHECKPOINT_PATH,
    CLASSES,
    DEFAULT_MEAN,
    DEFAULT_STD,
    IMG_SIZE,
    METADATA_PATH,
)
from .model import CNN, build_model
from .utils import get_logger

LOGGER = get_logger()


@dataclass
class Prediction:
    """Resultado de una predicción."""

    class_index: int
    class_name: str
    confidence: float
    probabilities: list[float]
    all_probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def top_k(self) -> list[tuple[str, float]]:
        """Clases ordenadas de mayor a menor probabilidad."""
        return sorted(self.all_probabilities.items(), key=lambda item: item[1], reverse=True)


class ModelNotAvailableError(RuntimeError):
    """Se lanza cuando el checkpoint no existe o no se puede cargar."""


def _read_metadata(metadata_path: Path) -> dict[str, Any]:
    if Path(metadata_path).exists():
        from .utils import load_json

        return load_json(metadata_path)
    return {}


def load_model(
    checkpoint_path: Path | str = CHECKPOINT_PATH,
    metadata_path: Path | str = METADATA_PATH,
    device: torch.device | str = "auto",
) -> tuple[CNN, dict[str, Any]]:
    """Carga el modelo entrenado y sus metadatos.

    Lanza `ModelNotAvailableError` con un mensaje accionable si el checkpoint
    no existe o es incompatible.
    """
    checkpoint_path = Path(checkpoint_path)
    metadata_path = Path(metadata_path)

    if not checkpoint_path.exists():
        raise ModelNotAvailableError(
            f"No se encontró el modelo entrenado en '{checkpoint_path}'. "
            "Entrénalo con 'python -m src.train' o descarga el checkpoint del repositorio."
        )

    from .utils import resolve_device

    if isinstance(device, torch.device):
        resolved_device = device
    elif str(device) == "auto":
        resolved_device = resolve_device("auto")
    else:
        resolved_device = torch.device(device)

    metadata = _read_metadata(metadata_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise ModelNotAvailableError(
            f"No se pudo leer el checkpoint '{checkpoint_path}': {exc}"
        ) from exc

    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ModelNotAvailableError(
            f"El archivo '{checkpoint_path}' no es un checkpoint válido (se esperaba un 'state_dict')."
        )

    model_config = checkpoint.get("model_config") or metadata.get("model_config") or {}
    if "normalization" in checkpoint and "normalization" not in metadata:
        metadata = {**metadata, "normalization": checkpoint["normalization"]}
    try:
        model = build_model(model_config)
    except Exception as exc:  # noqa: BLE001
        raise ModelNotAvailableError(
            f"La configuración del modelo no es válida: {exc}"
        ) from exc

    try:
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    except Exception as exc:  # noqa: BLE001
        raise ModelNotAvailableError(
            f"El checkpoint no es compatible con la arquitectura actual: {exc}"
        ) from exc

    model.eval()
    model.to(resolved_device)
    LOGGER.info("Modelo cargado desde %s", checkpoint_path)
    return model, metadata


def build_inference_transform(
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> transforms.Compose:
    """Transform de inferencia: redimensiona a 32x32, tensor y normaliza."""
    mean = tuple(mean) if mean is not None else DEFAULT_MEAN
    std = tuple(std) if std is not None else DEFAULT_STD
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def preprocess_image(
    image: Image.Image,
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> torch.Tensor:
    """Preprocesa una imagen PIL y devuelve un tensor `(1, 3, 32, 32)`."""
    image = image.convert("RGB")
    transform = build_inference_transform(mean, std)
    return transform(image).unsqueeze(0)


def load_image(path: Path | str) -> Image.Image:
    """Carga una imagen desde disco, launching un error claro si no es válida."""
    try:
        with Image.open(path) as handle:
            return handle.convert("RGB")
    except FileNotFoundError as exc:
        raise ModelNotAvailableError(f"No se encontró la imagen '{path}'.") from exc
    except UnidentifiedImageError as exc:
        raise ModelNotAvailableError(
            f"El archivo '{path}' no es una imagen válida o su formato no está soportado."
        ) from exc


@torch.no_grad()
def predict_image(
    model: CNN,
    image: Image.Image,
    device: torch.device | str = "cpu",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Prediction:
    """Predice la clase de una imagen PIL."""
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    model.eval()
    tensor = preprocess_image(image, mean=mean, std=std).to(resolved_device)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    class_index = int(probabilities.argmax())
    return Prediction(
        class_index=class_index,
        class_name=CLASSES[class_index],
        confidence=float(probabilities[class_index]),
        probabilities=[float(value) for value in probabilities],
        all_probabilities={
            name: float(probabilities[index]) for index, name in enumerate(CLASSES)
        },
    )


def predict_array(
    model: CNN,
    array: np.ndarray,
    device: torch.device | str = "cpu",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> Prediction:
    """Predice la clase de un array HWC uint8."""
    return predict_image(model, Image.fromarray(array.astype("uint8")), device=device, mean=mean, std=std)


@torch.no_grad()
def predict_batch(
    model: CNN,
    images: Sequence[Image.Image],
    device: torch.device | str = "cpu",
    mean: Sequence[float] | None = None,
    std: Sequence[float] | None = None,
) -> list[Prediction]:
    """Predice un lote de imágenes PIL en una sola pasada."""
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    model.eval()
    if not images:
        return []

    tensor = torch.stack([preprocess_image(image, mean=mean, std=std)[0] for image in images])
    tensor = tensor.to(resolved_device)
    probabilities = torch.softmax(model(tensor), dim=1).cpu().numpy()

    predicciones = []
    for fila in probabilities:
        indice = int(fila.argmax())
        predicciones.append(
            Prediction(
                class_index=indice,
                class_name=CLASSES[indice],
                confidence=float(fila[indice]),
                probabilities=[float(valor) for valor in fila],
                all_probabilities={
                    nombre: float(fila[position]) for position, nombre in enumerate(CLASSES)
                },
            )
        )
    return predicciones


def save_predictions(records: Sequence[dict[str, Any]], path: Path) -> Path:
    """Guarda una lista de predicciones de ejemplo en JSON."""
    from .utils import save_json

    return save_json(list(records), Path(path))


__all__ = [
    "ModelNotAvailableError",
    "Prediction",
    "build_inference_transform",
    "load_image",
    "load_model",
    "predict_array",
    "predict_batch",
    "predict_image",
    "preprocess_image",
    "save_predictions",
]
