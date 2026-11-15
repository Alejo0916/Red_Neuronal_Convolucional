"""Arquitectura de la Red Neuronal Convolucional para CIFAR-10.

Red de tres bloques convolucionales (64 → 128 → 256 canales) con BatchNorm,
ReLU, Max Pooling y Dropout2D, seguida de Global Average Pooling, una capa
densa con dropout y la salida de 10 logits.

    entrada (N, 3, 32, 32)
      bloque 1 → (N, 32, 16, 16)
      bloque 2 → (N, 64,  8,  8)
      bloque 3 → (N, 128, 4,  4)
      global avg pool → (N, 128)
      dense → (N, 10)
"""

from __future__ import annotations

import torch
from torch import nn

from .config import MODEL_CONFIG, NUM_CLASSES


class ConvBlock(nn.Module):
    """Bloque convolucional: 2x(Conv3x3 + BN + ReLU) + MaxPool + Dropout2D."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout2d(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN(nn.Module):
    """CNN para clasificación de imágenes de 32x32 en 10 clases."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = 3,
        channels: tuple[int, int, int] = (32, 64, 128),
        dropout2d: float = 0.10,
        dropout_head: float = 0.50,
    ) -> None:
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            ConvBlock(in_channels, c1, dropout=dropout2d),
            ConvBlock(c1, c2, dropout=dropout2d),
            ConvBlock(c2, c3, dropout=dropout2d),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_head)
        self.classifier = nn.Linear(c3, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve los logits `(N, num_classes)`; el softmax se aplica fuera."""
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.classifier(x)


def build_model(config: dict[str, object] | None = None) -> CNN:
    """Construye la CNN a partir de un diccionario de configuración."""
    cfg = dict(MODEL_CONFIG)
    if config:
        cfg.update({k: v for k, v in config.items() if k in cfg})
    channels = tuple(cfg["channels"])  # type: ignore[arg-type]
    return CNN(
        num_classes=int(cfg["num_classes"]),  # type: ignore[arg-type]
        in_channels=int(cfg["in_channels"]),  # type: ignore[arg-type]
        channels=channels,  # type: ignore[arg-type]
        dropout2d=float(cfg["dropout2d"]),  # type: ignore[arg-type]
        dropout_head=float(cfg["dropout_head"]),  # type: ignore[arg-type]
    )


def model_config(model: CNN) -> dict[str, object]:
    """Extrae la configuración serializable de una instancia de `CNN`."""
    first_conv = model.features[0].block[0]
    return {
        "num_classes": model.classifier.out_features,
        "in_channels": first_conv.in_channels,
        "channels": [
            model.features[0].block[0].out_channels,
            model.features[1].block[0].out_channels,
            model.features[2].block[0].out_channels,
        ],
        "dropout2d": float(model.features[0].block[-1].p),
        "dropout_head": float(model.dropout.p),
    }


__all__ = ["CNN", "ConvBlock", "build_model", "model_config"]
