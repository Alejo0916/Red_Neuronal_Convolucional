"""Paquete `src` — Red Neuronal Convolucional para clasificación de imágenes (CIFAR-10).

Submódulos:
- `config`: constantes, rutas y hyperparameters.
- `data`: carga, división estratificada, preprocesamiento y exploración.
- `model`: arquitectura CNN.
- `train`: bucle de entrenamiento y persistencia del checkpoint.
- `evaluate`: métricas, matriz de confusión, gráficas y predicciones de ejemplo.
- `predict`: inferencia sobre imágenes arbitrarias.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
