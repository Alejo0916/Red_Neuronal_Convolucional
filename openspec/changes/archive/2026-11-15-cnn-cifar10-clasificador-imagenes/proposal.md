# Proposal

## Why

El proyecto debe demostrar de forma reproducible y auditable un flujo completo de Deep Learning para clasificación de imágenes con redes neuronales convolucionales (CNN): desde la selección y exploración del dataset hasta la entrega de un modelo entrenado y una aplicación web que consume ese modelo para clasificar imágenes arbitrarias.

Hoy no existe en el repositorio ningún pipeline de datos, modelo, evaluación ni interfaz de inferencia; sin una especificación compartida, el trabajo queda disperso, no es reproducible y no se puede auditar si la app Streamlit usa realmente el modelo entrenado.

## What Changes

- Se añade un pipeline de datos reproducible sobre **CIFAR-10** (10 clases, 60 000 imágenes 32x32 RGB) con descarga automática, normalización, división estratificada en *train/val/test* y *data augmentation* aplicado solo al conjunto de entrenamiento.
- Se implementa una **CNN** en PyTorch (bloques convolucionales con BatchNorm, pooling, dropout y cabeza de clasificación) entrenada con optimizador AdamW y scheduler OneCycleLR, con checkpointing del mejor modelo según exactitud de validación.
- Se añade un módulo de **evaluación** que calcula exactitud, precisión, recall, macro-F1 y matriz de confusión, y genera las gráficas del entrenamiento (pérdida/accuracy, matriz de confusión) y una rejilla con **mínimo 10 predicciones** sobre imágenes de prueba.
- Se **guarda el modelo entrenado** (pesos + metadatos de configuración y métricas) en `models/` para que pueda consumirse sin reentrenar.
- Se añade una **aplicación Streamlit** que carga el checkpoint del repositorio, permite subir una imagen (o usar ejemplos del dataset), y devuelve la clase predicha junto con su probabilidad/confianza y el detalle completo de probabilidades por clase.
- Se añade un **notebook de experimentación** (Google Colab, `.ipynb`) con celdas ejecutadas que replican todo el pipeline: dataset, exploración, preprocesamiento, entrenamiento, evaluación y despliegue; el código del notebook es el mismo que el de `src/`.
- Se añade documentación de ejecución (`README.md`), `requirements.txt`, `.gitignore`, LICENSE y un flujo de CI que valida imports, modelo y aplicación.
- Se adoptan **OpenSpec** como capa de especificación: `openspec/specs/` publica los requisitos duraderos y `openspec/changes/` documenta el historial de cambios.

## Capabilities

### New Capabilities
- `data-pipeline`: Selección, descarga, exploración y visualización del dataset CIFAR-10; preprocesamiento, normalización, división estratificada train/val/test y política de data augmentation.
- `cnn-training`: Definición de la arquitectura CNN, configuración del optimizador/scheduler, bucle de entrenamiento con validación y persistencia del checkpoint del mejor modelo.
- `model-evaluation`: Cálculo de métricas (accuracy, precision, recall, macro-F1), generación de gráficas del entrenamiento, matriz de confusión y rejilla de predicciones de ejemplo.
- `inference-app`: Aplicación Streamlit que carga el modelo entrenado y realiza la predicción de una imagen mostrando clase y nivel de confianza.
- `experiment-notebook`: Notebook de experimentación ejecutable en Google Colab que replica el pipeline completo con salidas y figuras reales.
- `project-docs`: Documentación del repositorio: README con instrucciones de ejecución, `requirements.txt`, `.gitignore` y automatización de CI.

### Modified Capabilities
<!-- Ninguna: el repositorio parte sin especificaciones publicadas. -->

## Impact

- **Código nuevo**: `src/` (config, data, model, train, evaluate, predict), `app/` (Streamlit), `notebooks/` (Colab), `tools/` (generación del notebook).
- **Artefactos generados**: `models/` (checkpoint `.pt` + `metadata.json`), `reports/` (figuras PNG, `metrics.json`, `history.json`, predicciones de ejemplo).
- **Dependencias**: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `pillow`, `streamlit`, `tqdm`, `jupyter`/`nbformat`/`nbconvert`.
- **Entorno**: CPU y GPU (CUDA) soportados; el entrenamiento por defecto es CPU-compatible y reproduce en Colab con GPU automáticamente.
- **Git**: se añaden el historial de OpenSpec, `.gitignore` y el commit del modelo entrenado (~1–3 MB).
