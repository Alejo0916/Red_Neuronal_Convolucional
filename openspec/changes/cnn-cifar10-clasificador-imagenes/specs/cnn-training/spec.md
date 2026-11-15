# Spec Delta

## Purpose

Define la arquitectura de la red neuronal convolucional, la estrategia de optimización y entrenamiento, y las garantías de reproducibilidad y persistencia del modelo entrenado.

## ADDED Requirements

### Requirement: Arquitectura de la CNN
El proyecto SHALL implementar una red neuronal convolucional en PyTorch compuesta por tres bloques de convolución (con normalización por lotes y activación ReLU) con reducción progresiva de resolución y aumento de canales, seguida de pooling global average, una capa densa con dropout y una capa de salida con 10 neuronas logits.

#### Scenario: Forma de salida
- **WHEN** se pasa un lote de imágenes de forma (N, 3, 32, 32) por la red
- **THEN** la salida es un tensor de forma (N, 10) con valores logits sin softmax aplicado

#### Scenario: Número de parámetros
- **WHEN** se instancia la red y se cuentan sus parámetros entrenables
- **THEN** el total está en el orden de 1 a 10 millones de parámetros

### Requirement: Entrenamiento con validación y regularización
El entrenamiento SHALL usar la función de pérdida de entropía cruzada, el optimizador AdamW con regularización de pesos, un scheduler de tasa de aprendizaje con warmup y SHALL evaluar el modelo en cada época sobre el conjunto de validación.

#### Scenario: Optimizador y scheduler configurados
- **WHEN** se prepara el entrenamiento
- **THEN** el optimizador es AdamW con decay de pesos configurable y existe un scheduler que modifica la tasa de aprendizaje a lo largo de las épocas

#### Scenario: Métricas por época
- **WHEN** termina cada época
- **THEN** se registran pérdida de entrenamiento, exactitud de entrenamiento, pérdida de validación y exactitud de validación

#### Scenario: Dropout activo solo en entrenamiento
- **WHEN** el modelo está en modo de evaluación
- **THEN** la capa de dropout queda desactivada y la predicción es determinista

### Requirement: Seleccionar y guardar el mejor modelo
El entrenamiento SHALL conservar el checkpoint con mayor exactitud de validación y SHALL guardarlo en disco junto con la configuración de entrenamiento, las métricas obtenidas y el historial completo de épocas.

#### Scenario: Checkpoint del mejor modelo
- **WHEN** una época alcanza una exactitud de validación mayor que la de todas las épocas anteriores
- **THEN** el checkpoint del modelo se actualiza en disco

#### Scenario: Artefactos del modelo guardados
- **WHEN** finaliza el entrenamiento
- **THEN** existen en el directorio de modelos un archivo de pesos, un archivo de metadatos con configuración y métricas, y un archivo de historial de entrenamiento

### Requirement: Entrenamiento reproducible
El entrenamiento SHALL ser reproducible a partir de una semilla fija, configurando las semillas de Python, NumPy y PyTorch y eludiendo la dependencia de GPU cuando no esté disponible.

#### Scenario: Mismas semillas produces la misma inicialización
- **WHEN** se instancia la red dos veces con la misma semilla
- **THEN** los pesos iniciales son idénticos

#### Scenario: Ejecución en CPU sin GPU
- **WHEN** no hay GPU disponible
- **THEN** el entrenamiento se ejecuta en CPU sin errores y el dispositivo se registra en los metadatos

### Requirement: Entrenamiento por línea de comandos
El entrenamiento SHALL poder invocarse desde la terminal con un comando documentado que ejecute el pipeline completo (datos, entrenamiento, evaluación y persistencia) sin intervención manual.

#### Scenario: Ejecución documentada
- **WHEN** un usuario ejecuta el comando de entrenamiento indicado en el README
- **THEN** el proceso termina con código de salida 0 y deja el modelo, las métricas y las figuras en el repositorio
