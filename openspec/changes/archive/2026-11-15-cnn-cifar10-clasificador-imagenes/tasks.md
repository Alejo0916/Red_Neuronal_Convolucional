# Tasks

## 1. Entorno y estructura del repositorio

- [x] 1.1 Crear el esqueleto del repositorio (`src/`, `app/`, `notebooks/`, `models/`, `reports/`, `tools/`, `tests/`) y verificar que `git status` lista los directorios esperados
- [x] 1.2 Generar `requirements.txt` con versiones fijadas (torch, torchvision, numpy, pandas, scikit-learn, matplotlib, seaborn, pillow, streamlit, tqdm, jupyter) y verificar que `pip install -r requirements.txt` completa sin errores
- [x] 1.3 Añadir `.gitignore` que excluya `data/`, `.venv/`, `__pycache__/`, `*.log` y checkpoints temporales, y verificar que `models/` y `reports/` siguen versionables
- [x] 1.4 Implementar `tools/download_cifar10.py` con reanudación por rangos y reintentos, y verificar que completa la descarga de `cifar-10-python.tar.gz` y extrae el dataset

## 2. Pipeline de datos (`src/data.py`)

- [x] 2.1 Implementar `CLASSES` (10 clases CIFAR-10 en español), `SEED` y las funciones de carga de dataset con descarga automática idempotente; verificar que `CIFAR10(root="data", download=True)` devuelve 50 000 y 10 000 muestras
- [x] 2.2 Implementar la división estratificada 45 000 / 5 000 / 10 000 con `StratifiedShuffleSplit` y `random_state` fijo; verificar que los índices son disjuntos, reproducible y con desviación de clases ≤ 1 %
- [x] 2.3 Implementar `compute_channel_stats` sobre el subconjunto de entrenamiento y `build_transforms` con augmentation solo en train; verificar que el tensor de salida es `(3, 32, 32)` float32 normalizado y que val/test no incluyen recortes ni volteo
- [x] 2.4 Implementar `build_dataloaders` con `num_workers` adaptativo, `pin_memory` y semilla por worker; verificar que los DataLoader iteran sin error y reproducen el mismo orden con la misma semilla
- [x] 2.5 Implementar `explore_dataset` que genere la rejilla de muestras por clase, la distribución de clases por subconjunto y el informe de balance; verificar que las figuras PNG se escriben en `reports/figures/`

## 3. Modelo, entrenamiento y persistencia (`src/model.py`, `src/train.py`)

- [x] 3.1 Implementar la CNN de tres bloques (64/128/256) con BatchNorm, ReLU, MaxPool y Dropout2D, cabeza con Global Average Pooling + Dropout + Linear(256→10); verificar que la salida es `(N, 10)` y que el recuento de parámetros entrenables está entre 1 M y 10 M
- [x] 3.2 Implementar `build_model`, `count_parameters` y el guardado/carga de checkpoint (`state_dict` + `metadata.json`); verificar que un checkpoint guardado y recargado produce predicciones idénticas
- [x] 3.3 Implementar el bucle de entrenamiento con AdamW, OneCycleLR, CrossEntropyLoss, best-checkpoint por accuracy de validación y registro del historial por época; verificar que `history.json` contiene una entrada por época con loss y accuracy de train y val
- [x] 3.4 Implementar el comando `python -m src.train` con argumentos (épocas, batch size, seed, dispositivo, salida) y su escritura de `models/*.pt`, `models/metadata.json` y `reports/history.json`; verificar que termina con código 0 en CPU
- [x] 3.5 Añadir reproducción de semilla (Python, NumPy, PyTorch) y registro del dispositivo y de la versión de PyTorch en los metadatos; verificar que dos ejecuciones con la misma semilla dan la misma accuracy inicial
- [x] 3.6 Limitar los `num_workers` del DataLoader (2 en train, 0 en val/test) y desactivar `persistent_workers` para que el entrenamiento no agote la RAM en Windows; verificar que el consumo de memoria se mantiene estable durante las 20 épocas
- [x] 3.7 Implementar el checkpoint de estado completo `models/cnn_cifar10_last.pt` (pesos + optimizador + scheduler + historial) y la opción `--resume-from`; verificar que el entrenamiento se reanuda desde la última época guardada
- [x] 3.8 Separar el checkpoint versionado (mejor modelo según validación) del checkpoint de reanudación; verificar que `models/cnn_cifar10.pt` contiene el modelo con mayor accuracy de validación

## 4. Evaluación (`src/evaluate.py`)

- [x] 4.1 Implementar el cálculo de accuracy, precision, recall y f1 macro con `sklearn`, más la tabla de métricas por clase; verificar que los valores están en [0, 1] y que el informe registra el número de imágenes evaluadas
- [x] 4.2 Implementar la figura de matriz de confusión (conteos y versión normalizada por fila) y la tabla de pares de clases más confundidos; verificar que la figura es 10×10 y se guarda en `reports/figures/`
- [x] 4.3 Implementar las gráficas del entrenamiento (pérdida y accuracy frente a épocas); verificar que ambos PNG se generan con las curvas de train y val
- [x] 4.4 Implementar la rejilla de predicciones con al menos 10 imágenes que muestren etiqueta real, predicha y probabilidad, incluyendo aciertos y fallos; verificar que el PNG contiene ≥ 10 paneles
- [x] 4.5 Implementar `python -m src.evaluate --checkpoint ...` que reevalúe sin reentrenar y escriba `reports/metrics.json`; verificar que dos ejecuciones dan métricas idénticas

## 5. Predicción e inferencia (`src/predict.py`)

- [x] 5.1 Implementar `load_model(checkpoint_path)` que reconstruya la red desde los metadatos y cargue el `state_dict` con `strict=True`; verificar que carga el modelo del repositorio y que un checkpoint corrupto lanza un error legible
- [x] 5.2 Implementar `preprocess_image` (redimensionado a 32×32, tensor normalizado con las estadísticas guardadas) y `predict_image` que devuelva clase, confianza y vector completo de probabilidades; verificar que la forma del tensor es `(1, 3, 32, 32)`
- [x] 5.3 Implementar `predict_batch` para evaluar un conjunto y guardar predicciones de ejemplo en `reports/predictions.json`; verificar que el archivo contiene al menos 10 registros con etiqueta real, predicha y probabilidad

## 6. Aplicación Streamlit (`app/`)

- [x] 6.1 Implementar `app/app.py` con carga del modelo cacheada, `st.file_uploader` y manejo de errores; verificar que arranca con `streamlit run app/app.py` y que muestra un mensaje claro si falta el checkpoint
- [x] 6.2 Implementar la visualización del resultado: clase predicha, nivel de confianza en porcentaje, barra de probabilidades por clase y distinción entre confianza alta y baja; verificar que se renderizan los 10 valores de probabilidad
- [x] 6.3 Añadir imágenes de ejemplo del dataset como opción de prueba sin fichero propio y su etiqueta verdadera; verificar que al seleccionar un ejemplo se muestra la comparación predicción vs. verdad
- [x] 6.4 Añadir una sección de "información del modelo" con accuracy de validación/prueba y el número de parámetros, leída de `models/metadata.json`; verificar que los valores coinciden con `reports/metrics.json`

## 7. Notebook de experimentación (`notebooks/`)

- [x] 7.1 Implementar `tools/build_notebook.py` que genere el `.ipynb` con celdas Colab (instalación de dependencias, detección de GPU) y las etapas del pipeline; verificar que el notebook se genera sin errores y es un JSON válido de nbformat 4
- [x] 7.2 Ejecutar el notebook completo con `nbconvert` para incrustar salidas y figuras reales; verificar que ninguna celda termina en error y que el `.ipynb` versionado contiene salidas visibles
- [x] 7.3 Verificar que el notebook cubre dataset, exploración, preprocesamiento, división, augmentation, entrenamiento, métricas, gráficas, matriz de confusión, ≥ 10 predicciones y guardado del modelo, con la opción de reentrenar desde cero

## 8. Documentación y automatización

- [x] 8.1 Escribir `README.md` en español con descripción del proyecto y del dataset, arquitectura, instrucciones de instalación, comandos de entrenamiento/evaluación/predicción, instrucciones de Colab, resultados obtenidos y troubleshooting; verificar que cada comando corresponde a un archivo existente
- [x] 8.2 Añadir `LICENSE` (MIT) y `CITATION.cff`; verificar que el README enlaza la licencia
- [x] 8.3 Añadir `.github/workflows/ci.yml` que instale dependencias, verifique imports de `src`, cargue el checkpoint y compruebe la sintaxis de la app; verificar que el workflow está bien formado
- [x] 8.4 Documentar en el README cómo usar OpenSpec en el repositorio (`openspec list`, `openspec validate`, flujo propose/apply/archive); verificar que `openspec validate` no reporta errores

## 9. Verificación integral

- [x] 9.1 Ejecutar el pipeline completo de principio a fin en un entorno limpio y verificar que modelo, métricas, figuras y predicciones quedan regenerados
- [x] 9.2 Verificar que la app Streamlit predice correctamente imágenes de ejemplo del conjunto de prueba y que la confianza mostrada coincide con la calculada offline
- [x] 9.3 Revisar `git status` para confirmar que no se versionan artefactos pesados no deseados y que el modelo, las figuras y el notebook con salidas sí están presentes
