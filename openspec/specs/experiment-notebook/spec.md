# experiment-notebook Specification

## Purpose
Define el notebook de experimentación que replica el pipeline completo en Google Colab, de modo que el trabajo realizado en Colab quede disponible y reproducible dentro del repositorio.

## Requirements

### Requirement: Notebook de experimentación ejecutable en Colab
El repositorio SHALL incluir un notebook `.ipynb` que pueda abrirse y ejecutarse en Google Colab sin dependencias del repositorio local, incluyendo la instalación de dependencias y la detección de GPU.

#### Scenario: Ejecución en Colab
- **WHEN** un usuario abre el notebook en Google Colab y ejecuta todas las celdas
- **THEN** las celdas se completan sin errores y producen la misma salida documentada en el repositorio

#### Scenario: Detección de hardware
- **WHEN** se ejecuta la celda de configuración
- **THEN** el notebook informa si dispone de GPU CUDA y ajusta automáticamente el dispositivo de entrenamiento

### Requirement: Cobertura completa del pipeline en el notebook
El notebook SHALL cubrir explícitamente la selección y descripción del dataset, la exploración y visualización de imágenes, el preprocesamiento y la normalización, la división en entrenamiento/validación/prueba, la construcción y el entrenamiento de la CNN, el uso de data augmentation, la evaluación con métricas, las gráficas del entrenamiento, la matriz de confusión, al menos 10 predicciones con imágenes y el guardado del modelo entrenado.

#### Scenario: Todas las etapas presentes
- **WHEN** se recorre el notebook de principio a fin
- **THEN** cada una de las etapas listadas tiene su propia celda con código y una salida visible

### Requirement: Notebook y código del repositorio alineados
El notebook SHALL ejecutar el mismo código que los módulos del paquete `src/` cuando esté disponible en el entorno, de modo que no existan dos implementaciones divergentes del pipeline.

#### Scenario: Uso del paquete del repositorio
- **WHEN** el notebook se ejecuta en un entorno donde el paquete `src` está disponible
- **THEN** importa los módulos del repositorio en lugar de redefinir la arquitectura y el entrenamiento

### Requirement: Salidas reales versionadas
El notebook SHALL estar versionado con sus salidas ejecutadas (figuras, métricas y tablas visibles) para que el repositorio documente los resultados obtenidos.

#### Scenario: Notebook con salidas
- **WHEN** se abre el notebook en GitHub
- **THEN** se muestran las figuras generadas y los valores de las métricas sin necesidad de reejecutarlo

### Requirement: Opción de reentrenamiento desde cero
El notebook SHALL ofrecer una opción explícita para reentrenar el modelo desde cero en Colab y SHALL indicar claramente cuándo se está reutilizando el checkpoint preentrenado del repositorio.

#### Scenario: Reentrenamiento en Colab
- **WHEN** el usuario activa la bandera de reentrenamiento y ejecuta la celda de entrenamiento
- **THEN** el modelo se entrena de nuevo en el entorno de Colab y el checkpoint resultante se guarda

#### Scenario: Reutilización del checkpoint
- **WHEN** la bandera de reentrenamiento está desactivada y existe un checkpoint en el repositorio
- **THEN** el notebook carga ese checkpoint y lo indica en la salida de la celda
