# data-pipeline Specification

## Purpose
Define el dataset de referencia (CIFAR-10), su exploración y visualización, el preprocesamiento y normalización de las imágenes, la división estratificada en entrenamiento/validación/prueba y la política de data augmentation del proyecto.

## Requirements

### Requirement: Dataset de referencia CIFAR-10
El proyecto SHALL usar CIFAR-10 como dataset de trabajo, con 10 clases (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco, camión) y 60 000 imágenes RGB de 32x32 píxeles, y SHALL documentar su descripción, tamaño por clase, origen y licencia en la documentación del repositorio.

#### Scenario: Descarga automática del dataset
- **WHEN** un usuario ejecuta el pipeline de datos en una máquina sin el dataset descargado
- **THEN** el dataset se descarga automáticamente a una caché local versionable por el usuario y la descarga es idempotente (no se repite si ya existe)

#### Scenario: Metadatos del dataset documentados
- **WHEN** se consulta la documentación del repositorio
- **THEN** aparece una sección con el nombre del dataset, el número de clases, el número de imágenes por clase, el tamaño de las imágenes y la fuente de descarga

### Requirement: Exploración y visualización de imágenes
El pipeline SHALL producir una exploración del dataset que incluya una muestra visual de imágenes por clase, una distribución de ejemplos por clase y una muestra del preprocesamiento aplicado.

#### Scenario: Rejilla de muestras por clase
- **WHEN** se ejecuta la exploración del dataset
- **THEN** se genera una figura con imágenes de ejemplo de cada una de las 10 clases etiquetadas con su nombre en español

#### Scenario: Distribución de clases
- **WHEN** se ejecuta la exploración del dataset
- **THEN** se genera una figura con el número de ejemplos de cada clase en los conjuntos de entrenamiento, validación y prueba

### Requirement: Preprocesamiento y normalización
Las imágenes SHALL convertirse a tensores de punto flotante en rango [0, 1] y SHALL normalizarse por canal usando las medias y desviaciones estándar calculadas sobre el conjunto de entrenamiento; el mismo par de estadísticas SHALL aplicarse a validación, prueba e inferencia.

#### Scenario: Tensor normalizado
- **WHEN** se transforman las imágenes del conjunto de entrenamiento
- **THEN** cada imagen devuelta es un tensor de forma (3, 32, 32) de tipo float32 con valores normalizados según las medias y desviaciones del conjunto de entrenamiento

#### Scenario: Estadísticas consistentes entre conjuntos
- **WHEN** se comparan las transformaciones de entrenamiento y de prueba
- **THEN** ambas usan exactamente el mismo par de medias y desviaciones, sin recalcular estadísticas sobre los datos de prueba

### Requirement: División estratificada de datos
El dataset SHALL dividirse en tres subconjuntos disjuntos —entrenamiento, validación y prueba— con proporción 45 000 / 5 000 / 10 000 imágenes, conservando la distribución de clases de forma estratificada, y los subconjuntos SHALL ser reproducibles a partir de una semilla fija.

#### Scenario: Proporciones y estratificación
- **WHEN** se genera la división
- **THEN** los tres subconjuntos contienen 45 000, 5 000 y 10 000 imágenes respectivamente y la proporción de cada clase se mantiene dentro de ±1 % respecto al conjunto original

#### Scenario: Reproducibilidad
- **WHEN** se ejecuta la división dos veces con la misma semilla
- **THEN** los índices de los tres subconjuntos son idénticos

#### Scenario: Disjointness
- **WHEN** se comprueban los índices de los tres subconjuntos
- **THEN** no existe ningún índice presente en más de un subconjunto

### Requirement: Data augmentation en entrenamiento
El pipeline SHALL aplicar data augmentation únicamente a las imágenes del conjunto de entrenamiento y SHALL NOT aplicarlo a los conjuntos de validación y prueba; la augmentation SHALL consistir en recorte aleatorio con relleno, volteo horizontal y normalización por canal.

#### Scenario: Augmentation solo en entrenamiento
- **WHEN** se construyen los transformadores de entrenamiento, validación y prueba
- **THEN** solo el transformador de entrenamiento incluye recorte aleatorio con relleno y volteo horizontal

#### Scenario: Determinismo en evaluación
- **WHEN** se evalúa el modelo sobre el conjunto de prueba
- **THEN** la imagen que entra al modelo es idéntica a la almacenada en el dataset, sin recortes ni volteos aleatorios

### Requirement: Balance de clases del dataset
El pipeline SHALL comprobar el balance de clases y SHALL exponer la distribución resultante en la salida de exploración, de modo que el usuario pueda detectar desbalances antes de entrenar.

#### Scenario: Informe de balance
- **WHEN** se completa la exploración del dataset
- **THEN** la salida incluye el recuento de imágenes por clase para cada subconjunto y una indicación explícita de si el dataset está balanceado
