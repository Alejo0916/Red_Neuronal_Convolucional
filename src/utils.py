"""Utilidades compartidas: semillas, dispositivos, JSON y registro de progreso."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER_NAME = "cnn_cifar10"


# --------------------------------------------------------------------------- #
# Registro
# --------------------------------------------------------------------------- #
def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Devuelve un logger con formato consistente en todo el proyecto."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s | %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


# --------------------------------------------------------------------------- #
# Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Fija las semillas de Python, NumPy y PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:  # noqa: ARG001
    """`worker_init_fn` para que cada worker de DataLoader tenga su propia semilla."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------------------------------- #
# Dispositivo
# --------------------------------------------------------------------------- #
def resolve_device(requested: str = "auto") -> torch.device:
    """Resuelve el dispositivo de cómputo solicitado."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        get_logger().warning("CUDA no está disponible; se usará CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def describe_device(device: torch.device) -> str:
    """Nombre legible del dispositivo."""
    if device.type == "cuda":
        return f"CUDA ({torch.cuda.get_device_name(device)})"
    return "CPU"


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def save_json(payload: Any, path: Path) -> Path:
    """Guarda `payload` como JSON indentado y UTF-8."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
    return path


def load_json(path: Path) -> Any:
    """Carga un JSON desde disco."""
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Objeto no serializable: {type(value)!r}")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """Número de parámetros del modelo."""
    parameters = model.parameters()
    if trainable_only:
        parameters = (p for p in parameters if p.requires_grad)
    return sum(p.numel() for p in parameters)


__all__ = [
    "count_parameters",
    "describe_device",
    "get_logger",
    "load_json",
    "resolve_device",
    "save_json",
    "seed_worker",
    "set_seed",
]
