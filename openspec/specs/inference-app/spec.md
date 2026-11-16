# inference-app Specification

## Purpose
Define la aplicación web que consume el modelo entrenado: permite subir una imagen arbitraria y devuelve la clase predicha junto con su probabilidad, sin requerir entrenamiento previo.

## Requirements

### Requirement: Carga del modelo entrenado
La aplicación SHALL cargar el checkpoint entrenado incluido en el repositorio al iniciarse, en modo de evaluación y sobre el dispositivo disponible, y SHALL mostrar un error claro si el archivo del modelo no existe o es incompatible.

#### Scenario: Arranque con modelo disponible
- **WHEN** se inicia la aplicación con el checkpoint presente en el repositorio
- **THEN** el modelo se carga correctamente y la interfaz queda lista para predecir

#### Scenario: Modelo ausente
- **WHEN** se inicia la aplicación sin el archivo de modelo
- **THEN** la interfaz muestra un mensaje de error que indica la ruta esperada del checkpoint y no se bloquea la aplicación

### Requirement: Predicción sobre imagen subida
La aplicación SHALL permitir cargar una imagen desde el navegador, aplicarle el mismo preprocesamiento que al entrenamiento y SHALL devolver la clase predicha junto con su probabilidad.

#### Scenario: Clasificación de una imagen
- **WHEN** el usuario sube una imagen y se ejecuta la predicción
- **THEN** la interfaz muestra el nombre de la clase predicha, el nivel de confianza en porcentaje y la imagen cargada

#### Scenario: Preprocesamiento idéntico al entrenamiento
- **WHEN** se procesa la imagen subida
- **THEN** se redimensiona a 32x32, se normaliza con las medias y desviaciones guardadas en los metadatos del modelo y se convierte a un tensor de forma (1, 3, 32, 32)

#### Scenario: Imagen en formato no soportado
- **WHEN** el usuario sube un archivo que no es una imagen decodificable
- **THEN** la interfaz muestra un mensaje de error descriptivo en lugar de fallar

### Requirement: Visualización de la confianza
La aplicación SHALL mostrar la distribución completa de probabilidades por clase y SHALL destacar visualmente la clase de mayor probabilidad.

#### Scenario: Barra de probabilidades
- **WHEN** se muestra el resultado de una predicción
- **THEN** se visualizan las 10 clases con su probabilidad y la clase predicha aparece destacada como la de mayor probabilidad

#### Scenario: Nivel de confianza cualificado
- **WHEN** la probabilidad de la clase predicha es alta
- **THEN** la interfaz indica un nivel de confianza alto; cuando es baja, indica un nivel de confianza bajo con una advertencia visual

### Requirement: Ejemplos integrados
La aplicación SHALL ofrecer imágenes de ejemplo del dataset para que el usuario pueda probar la interfaz sin tener una imagen propia.

#### Scenario: Predicción con imagen de ejemplo
- **WHEN** el usuario selecciona una imagen de ejemplo incorporada
- **THEN** la aplicación la clasifica y muestra el resultado junto con la etiqueta verdadera de la imagen

### Requirement: Arranque documentado
La aplicación SHALL poder iniciarse con un comando documentado en el README y SHALL ser compatible con la instalación mediante `requirements.txt`.

#### Scenario: Ejecución local
- **WHEN** un usuario instala las dependencias y ejecuta el comando indicado
- **THEN** la aplicación sirve la interfaz en el puerto por defecto y responde correctamente
