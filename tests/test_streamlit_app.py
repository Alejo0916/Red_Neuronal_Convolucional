"""Pruebas de la aplicación de Streamlit.

Se usa `streamlit.testing.v1.AppTest` para ejecutar la app sin navegador y
comprobar que arranca, carga el modelo y responde a la interacción.

Ejecución:
    python -m pytest tests/test_streamlit_app.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CHECKPOINT_PATH, CLASSES  # noqa: E402

AppTest = pytest.importorskip("streamlit.testing.v1", reason="streamlit no está instalado").AppTest

APP_PATH = PROJECT_ROOT / "app" / "app.py"

requiere_modelo = pytest.mark.skipif(
    not CHECKPOINT_PATH.exists(), reason="No hay modelo entrenado en models/"
)


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP_PATH), default_timeout=180)


def test_la_app_arranca_sin_excepciones(app):
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]


@requiere_modelo
def test_la_app_muestra_el_titulo_y_las_clases(app):
    app.run()
    titulos = [element.value for element in app.title]
    assert any("Clasificador" in titulo for titulo in titulos)

    markdown = "\n".join(element.value for element in app.markdown)
    for clase in ("gato", "perro", "camión"):
        assert clase in markdown


@requiere_modelo
def test_la_app_ofrece_imagenes_de_ejemplo(app):
    app.run()
    assert len(app.radio) >= 1
    opciones = app.radio[0].options
    assert any("ejemplo" in opcion.lower() for opcion in opciones)

    indice = next(i for i, opcion in enumerate(opciones) if "ejemplo" in opcion.lower())
    app.radio[0].set_value(opciones[indice]).run()
    assert len(app.selectbox) >= 1
    assert len(app.selectbox[0].options) >= 1


@requiere_modelo
def test_la_app_predice_una_imagen_de_ejemplo(app):
    app.run()
    opciones = app.radio[0].options
    indice = next(i for i, opcion in enumerate(opciones) if "ejemplo" in opcion.lower())
    app.radio[0].set_value(opciones[indice]).run()

    if not app.selectbox:
        pytest.skip("No hay imágenes de ejemplo disponibles")

    app.selectbox[0].select(app.selectbox[0].options[0]).run()
    assert len(app.metric) >= 1
    assert app.metric[0].label == "Clase predicha"
    assert app.metric[0].value


def test_el_manifiesto_de_ejemplos_tiene_clases_validas():
    manifest_path = PROJECT_ROOT / "app" / "assets" / "examples" / "manifest.json"
    if not manifest_path.exists():
        pytest.skip("No hay imágenes de ejemplo exportadas")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest) >= 10
    for entrada in manifest:
        assert (PROJECT_ROOT / "app" / "assets" / "examples" / entrada["file"]).exists()
        assert entrada["class_name"] in CLASSES
        assert 0 <= entrada["class_index"] < len(CLASSES)
