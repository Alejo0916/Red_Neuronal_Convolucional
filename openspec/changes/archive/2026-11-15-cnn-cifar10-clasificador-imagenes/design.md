# Design

## Context

El repositorio arranca vacío salvo la estructura de OpenSpec. El objetivo es un pipeline de clasificación de imágenes con CNN que tenga que:

- reproducirse en tres entornos distintos (local con CPU, Google Colab con GPU, CI sin GPU);
- ser **auditable**: el mismo código en el notebook, en `src/` y en la app;
-Entregar un modelo entrenado versionado en Git (~1–3 MB) para que la app funcione sin reentrenar;
- ser legible para un evaluador que no ejecuta el proyecto: resultados, figuras y decisiones documentados.

Restricciones reales del entorno de referencia: CPU de 12 hilos (sin CUDA), descarga de datasets lenta (~170 kB/s), Python 3.13 y Windows. Esto condiciona tanto la descarga como el tiempo de entrenamiento disponible.

## Goals / Non-Goals

**Goals:**
- Un paquete `src/` importable (`from src.data import ...`) que contenga toda la lógica del pipeline.
- Un checkpoint ligero y portable (solo `state_dict` + metadatos JSON, sin pickles de clase completa) que se pueda cargar en Streamlit sin importar el paquete completo.
- Separación estricta entre *train/eval transform* y *inference transform*.
- Resultados reproducibles: semilla fija, splits guardados, métricas deterministas.
- El notebook Colab como **artefacto de primera clase**, con salidas reales versionadas.

**Non-Goals:**
- No se persiguen métricas de estado del arte (>95 % en CIFAR-10); el objetivo es un pipeline correcto, legible y reproducible con una accuracy sólida (~88–92 %).
- No se usan modelos preentrenados (ImageNet) ni transfer learning: el objetivo es construir y entrenar la CNN desde cero.
- No se despliega en un servicio cloud; la app es local/Streamlit Community Cloud.
- No se implementa un servicio REST ni base de datos.

## Decisions

### D1. PyTorch frente a TensorFlow/Keras
**Decisión:** PyTorch.
**Alternativas:** TensorFlow/Keras (el estándar de muchos cursos y de Colab).
**Rationale:** API más explícita para el bucle de entrenamiento, `state_dict` portable entre versiones y checkpointing directo con diccionarios; permite además escribir un `nn.Module` que la app instancia sin código de framework específico. La compatibilidad con Colab (`torch` ya instalado con soporte CUDA) esplug-and-play.

### D2. CIFAR-10 vía `torchvision.datasets`
**Decisión:** dataset oficial de `torchvision`, descargado con un script de reanudación propio (`tools/download_cifar10.py`).
**Alternativas:** HuggingFace `datasets`, descarga manual del `.tar.gz`.
**Rationale:** `torchvision` ya entrega las etiquetas y una API de `Dataset` estándar; la descarga oficial (`cs.toronto.edu`) resultaba lenta e interrumpida, por lo que se añade un descargador con soporte deRange requests y reintentos automáticos que reutiliza el `.tar.gz` parcial. Esto hace la descarga reanudable e idempotente en redes lentas.

### D3. Normalización por canal con estadísticas del subconjunto de entrenamiento
**Decisión:** `ToTensor()` (escala a [0, 1]) + `Normalize(mean, std)` calculados sobre las 45 000 imágenes de entrenamiento.
**Alternativas:** normalización por imagen (instance norm), `[0, 255]` sin escalar, statistics de todo el dataset.
**Rationale:** los canales de CIFAR-10 tienen escalas distintas; normalizar por canal acelera la convergencia. Calcular las estadísticas **solo sobre entrenamiento** evita fuga de información de validación/prueba: es una decisión metodológica explícita y verificable.

### D4. División estratificada fija 45 000 / 5 000 / 10 000
**Decisión:** se parte el conjunto oficial de entrenamiento (50 000) en 45 000 train + 5 000 val usando `StratifiedShuffleSplit` con `random_state=42`, y se reserva el conjunto de prueba oficial (10 000) sin tocar.
**Rationale:** la estratificación mantiene la distribución de clases (CIFAR-10 es balanceado, pero la estratificación protege la propiedad). El test oficial nunca se usa para decisiones: la selección de checkpoint usa el accuracy de **validación**, nunca del test.

### D5. Augmentation solo en entrenamiento
**Decisión:** `RandomCrop(32, padding=4)` + `RandomHorizontalFlip()` en train; `ToTensor` + `Normalize` en val/test.
**Rationale:** el recorte con relleno y el volteo son aumentaciones geométricas clásicas para CIFAR-10 que no cambian la etiqueta semántica. Aplicarlas en evaluación introduciría ruido en las métricas. La augmentation se activa porque el dataset (45 000 imágenes) es limitado frente a la capacidad del modelo: sin ella, el sobreajuste es visible en las curvas.

### D6. Arquitectura: CNN de tres bloques con BatchNorm y Global Average Pooling
**Decisión:** `Conv3x3(3→64) → BN → ReLU → Conv3x3(64→64) → BN → ReLU → MaxPool2 → Dropout2d` × 3 bloques (64/128/256) → `AdaptiveAvgPool2d(1)` → `Dropout` → `Linear(256→10)`.
**Alternativas:** VGG16 completa (~15 M parámetros, overkill para 32×32), ResNet-18 (más precisa pero 3–4× más lenta en CPU), AlexNet clásica.
**Rationale:** con 32×32 de entrada, una red profunda tipo ResNet no aporta tanto; el diseño de tres bloques con pooling progresivo (32→16→8→4) alcanza ~90 % de accuracy con ~2 M parámetros y se entrena en minutos en CPU. El **Global Average Pooling** reduce drásticamente los parámetros de la cabeza y evita overfitting.

### D7. Optimización: AdamW + OneCycleLR
**Decisión:** `AdamW(lr=1e-3, weight_decay=5e-4)` + `OneCycleLR` con 30 épocas.
**Alternativas:** SGD+momentum (clásico de ImageNet, más lento de converger aquí), CosineAnnealing, learning rate constante.
**Rationale:** OneCycleLR (warmup + annealing) converge en pocas épocas, que es la restricción real del entorno CPU. El `weight_decay` de AdamW actúa como regularizador（L2 desacoplado), complemento del dropout.

### D8. Checkpoint desacoplado del código
**Decisión:** el artefacto persistido es un `state_dict` (`.pt`) + `metadata.json` (configuración, clases, estadísticas de normalización, métricas, versión del esquema).
**Alternativas:** `torch.save(modelo_completo)` (unpickle de clases → frágil entre versiones y entorno de Colab).
**Rationale:** la app y el usuario pueden reconstruir el modelo con `build_model(cfg)` y cargar los pesos con `strict=True`. Se puede añadir una comprobación de `state_dict` en la carga para detectar incompatibilidades.

### D9. La app consume el modelo sin depender del entrenamiento
**Decisión:** `src/predict.py` expone `load_model()` y `predict_image()`; la app Streamlit solo usa esas dos funciones y un `metadata.json`.
**Alternativas:** importar `src.train` desde la app (acopla la app a scikit-learn, matplotlib y al dataset).
**Rationale:** mantiene la app ligera y testeable. Si `metadata.json` no está, la app muestra un error accionable en vez de romperse.

### D10. Notebook Colab generado desde el código del paquete
**Decisión:** `tools/build_notebook.py` genera `notebooks/experimentacion_cifar10_cnn.ipynb` con celdas que importan `src` cuando está disponible y, si no, caen a implementaciones equivalentes embebidas.
**Alternativas:** notebook escrito a mano (se desincroniza) o notebook generado en runtime (no versionable con salidas).
**Rationale:** el requisito es que "todo lo realizado en Google Colab quede en el repositorio". Un notebook generado + ejecutado y versionado con sus salidas cumple, y el generador garantiza que el código del notebook y el de `src/` no divergen.

### D11. OpenSpec como capa de especificación
**Decisión:** `openspec/specs/` (requisitos duraderos) + `openspec/changes/` (historial) + `openspec/config.yaml` (idioma `es`).
**Rationale:** aporta trazabilidad formal requisito→implementación→prueba, exigida implícitamente al pedir "documentación con OpenSpec".

### D12. DataLoader con workers limitados y sin persistencia
**Decisión:** `num_workers=2` por defecto solo para el loader de entrenamiento; los loaders de validación y prueba usan `num_workers=0`; `persistent_workers=False` en los tres.
**Alternativas:** `num_workers=4` con `persistent_workers=True`.
**Rationale:** en Windows los workers de `DataLoader` se crean con `spawn` y **cada uno recibe una copia serializada del dataset** (~150 MB) además del runtime de PyTorch. Con 4 workers persistentes en train + val el proceso llegaba a consumir > 5 GB y el entrenamiento abortó con `DefaultCPUAllocator: not enough memory`. Con 2 workers sin persistencia el consumo se mantiene estable y la ganancia de paralelismo sigue siendo marginal (el preprocesamiento son ~7 s de los ~230 s por época).

### D13. Dos checkpoints: "mejor modelo" y "estado reanudable"
**Decisión:** `models/cnn_cifar10.pt` contiene **el mejor modelo** según accuracy de validación (es el que consume la app y el que se versiona); `models/cnn_cifar10_last.pt` contiene el **estado completo** de la última época (pesos + optimizador + scheduler + historial) y permite reanudar con `--resume-from`.
**Alternativas:** guardar un único checkpoint con todo; reanudar desde cero si el proceso se interrumpe.
**Rationale:** separar "qué modelo se entrega" de "cómo se recupera un entrenamiento interrumpido" mantiene el artefacto versionado limpio (≈ 4,4 MB) sin perder la capacidad de continuar, y permite que el historial completo llegue hasta la última época.

### D14. Evaluación sin cargar todo el test set en memoria
**Decisión:** la rejilla de predicciones accede a las imágenes del test set bajo demanda (`dataset[index]`), en lugar de apilar los 10 000 tensores normalizados en memoria (≈ 123 MB).
**Rationale:** mantiene la evaluación reproducible en equipos con poca RAM, que es el escenario real de este proyecto.

## Risks / Trade-offs

- **Tiempo de entrenamiento en CPU** → el diseño de la CNN prioriza un número de parámetros razonable; se usan `num_workers` adaptativos y `pin_memory`. El número de épocas por defecto (30) es un compromiso entre calidad y tiempo; se documenta cómo reducirlo.
- **Descarga lenta del dataset** → el descargador con reanudación evita perder el progreso; el README documenta la descarga manual alternativa.
- **Sobreajuste con 45 000 imágenes** → mitigado con augmentation, dropout (2D + cabecera), weight decay y selección de checkpoint por validación.
- **Drift de versión de PyTorch entre el entorno de entrenamiento y el de inferencia** → el checkpoint guarda `state_dict` (no objetos Python) y `metadata.json` registra la versión usada; la carga usa `strict=True` para detectar incompatibilidades de forma ruidosa.
- **Cifras de resultados dependientes del hardware** → el README documenta que la accuracy puede variar ±0.5 % entre CPU y GPU por no-determinismo de cuDNN; se fijan `torch.use_deterministic_algorithms` cuando es posible.
- **App Streamlit que consume un modelo grande** → se usa un modelo de ~2 M parámetros (~8 MB) para que la carga sea instantánea.

## Migration Plan

No hay migración: es un repositorio nuevo. El orden de entrega es:

1. `src/` + `tools/` (código del pipeline).
2. Entrenamiento → `models/` + `reports/`.
3. Notebook generado y ejecutado → `notebooks/`.
4. App Streamlit → `app/`.
5. Documentación, CI y commit final.

**Rollback:** todo el estado mutable vive en `models/` y `reports/`, ambos regenerables; si un modelo resultara corrupto, basta con reentrenar (`python -m src.train`) y volver a commitear.

## Open Questions

- ¿Publicar la app en Streamlit Community Cloud con `models/` versionado? Depende de la disponibilidad de credenciales de la cuenta y queda fuera del repositorio; el README documenta el despliegue manual sin secretos.
