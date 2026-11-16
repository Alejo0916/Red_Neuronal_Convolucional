# project-docs Specification

## Purpose
Define los artefactos de documentación y automatización del repositorio: instrucciones de ejecución, dependencias declaradas, control de versiones y verificación continua del proyecto.

## Requirements

### Requirement: README con instrucciones de ejecución
El repositorio SHALL incluir un `README.md` en español que describa el proyecto, el dataset, la arquitectura, los resultados obtenidos y los comandos exactos para instalar, entrenar, evaluar y ejecutar la aplicación.

#### Scenario: Comandos verificables
- **WHEN** un usuario sigue los comandos del README
- **THEN** cada comando indicado corresponde a un archivo existente y es ejecutable desde la raíz del repositorio

#### Scenario: Sección de resultados
- **WHEN** se consulta el README
- **THEN** aparecen las métricas finales del modelo y enlaces a las figuras de evaluación generadas

### Requirement: Dependencias declaradas
El repositorio SHALL incluir un `requirements.txt` con las versiones exactas de las librerías necesarias para instalar el entorno, el entrenamiento y la aplicación.

#### Scenario: Instalación desde requirements
- **WHEN** se crea un entorno virtual y se instalan las dependencias del archivo
- **THEN** el entorno permite importar el paquete `src` y arrancar la aplicación sin instalar librerías adicionales

### Requirement: Exclusión de artefactos pesados
El repositorio SHALL excluir del control de versiones la caché del dataset, los entornos virtuales y los archivos de log, pero SHALL versionar el modelo entrenado, las métricas y las figuras.

#### Scenario: Clone limpio del repositorio
- **WHEN** un usuario clona el repositorio
- **THEN** el modelo entrenado y las figuras están presentes y la caché del dataset no lo está

### Requirement: Automatización de verificación
El repositorio SHALL incluir una integración continua que compruebe que el paquete `src` es importable, que el checkpoint del modelo carga correctamente y que la aplicación de Streamlit arranca sin errores de sintaxis.

#### Scenario: Ejecución continua correcta
- **WHEN** se envía un commit a la rama principal
- **THEN** la integración continua ejecuta las verificaciones y el resultado queda registrado en el repositorio

### Requirement: Documentación de la especificación
El repositorio SHALL publicar las especificaciones duraderas en `openspec/specs/` y SHALL conservar el historial de cambios en `openspec/changes/`, de modo que la documentación del proyecto y su especificación se mantengan alineadas.

#### Scenario: Especificaciones validadas
- **WHEN** se ejecuta la validación de OpenSpec sobre el cambio archivado
- **THEN** la validación reporta cero errores y los requisitos quedan publicados en `openspec/specs/`
