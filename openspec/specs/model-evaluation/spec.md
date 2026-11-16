# model-evaluation Specification

## Purpose
Define cómo se evalúa el modelo entrenado: las métricas calculadas sobre el conjunto de prueba, las gráficas del entrenamiento, la matriz de confusión y la rejilla de predicciones de ejemplo que evidencia el comportamiento del clasificador.

## Requirements

### Requirement: Métricas de evaluación
El módulo de evaluación SHALL calcular sobre el conjunto de prueba la exactitud (accuracy), la precisión (precision), el recall y el F1 macro, todos en escala 0-1, y SHALL persistirlos en un archivo de métricas legible por máquina.

#### Scenario: Métricas calculadas sobre el set de prueba
- **WHEN** se evalúa el modelo entrenado sobre las 10 000 imágenes de prueba
- **THEN** el informe contiene accuracy, precision, recall y f1_macro con valores entre 0 y 1, además del número de imágenes evaluadas

#### Scenario: Métricas por clase
- **WHEN** se genera el informe de evaluación
- **THEN** incluye una tabla con precision, recall y f1-score por cada una de las 10 clases

### Requirement: Gráficas del entrenamiento
El pipeline SHALL generar una figura con la evolución de la pérdida (entrenamiento y validación) y otra con la evolución de la exactitud (entrenamiento y validación) frente a las épocas.

#### Scenario: Figuras de curvas
- **WHEN** finaliza el entrenamiento
- **THEN** existen dos figuras PNG de curvas con el eje horizontal en número de época y las curvas de entrenamiento y validación diferenciadas

### Requirement: Matriz de confusión
El pipeline SHALL generar una matriz de confusión de 10x10 sobre el conjunto de prueba, con el conteo de aciertos por clase en la diagonal, y SHALL reportar las clases con mayor confusión.

#### Scenario: Matriz de confusión normalizada y con conteos
- **WHEN** se evalúa el modelo
- **THEN** se produce una figura con la matriz de confusión y su versión normalizada por fila, y una tabla con los pares de clases más confundidos

### Requirement: Predicciones de ejemplo
El pipeline SHALL generar una rejilla con un mínimo de 10 imágenes del conjunto de prueba, cada una mostrando la imagen original, la clase verdadera, la clase predicha y la probabilidad de la predicción.

#### Scenario: Rejilla de al menos 10 predicciones
- **WHEN** se completa la evaluación
- **THEN** se guarda una figura con al menos 10 imágenes anotadas con su etiqueta real, su etiqueta predicha y su confianza

#### Scenario: Selección representativa
- **WHEN** se eligen las imágenes de ejemplo
- **THEN** la rejilla incluye casos acertados y casos fallidos para que el error sea visible

### Requirement: Informe de evaluación reproducible
La evaluación SHALL poder ejecutarse de forma independiente sobre un checkpoint guardado, sin necesidad de reentrenar, y SHALL regenerar todas las figuras y métricas de forma determinista.

#### Scenario: Reevaluación desde checkpoint
- **WHEN** un usuario ejecuta el comando de evaluación con el modelo ya entrenado
- **THEN** se regeneran las métricas y figuras sin repetir el entrenamiento

#### Scenario: Determinismo de la evaluación
- **WHEN** se evalúa el mismo checkpoint dos veces
- **THEN** las métricas obtenidas son idénticas
