"""Aplicación Streamlit: clasificación de imágenes con la CNN entrenada en CIFAR-10.

Arranque:
    streamlit run app/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

from src.config import CHECKPOINT_PATH, CLASSES, METADATA_PATH
from src.predict import ModelNotAvailableError, load_model, predict_image
from src.utils import load_json, resolve_device

st.set_page_config(
    page_title="Clasificador de imágenes · CNN sobre CIFAR-10",
    page_icon="🧠",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR / "assets" / "examples"

# --------------------------------------------------------------------------- #
# Estado
# --------------------------------------------------------------------------- #
if "last_prediction" not in st.session_state:
    st.session_state["last_prediction"] = None


@st.cache_resource(show_spinner=False)
def get_model():
    """Carga el modelo una sola vez y lo cachea entre recargas de la app."""
    device = resolve_device("auto")
    model, metadata = load_model(CHECKPOINT_PATH, METADATA_PATH, device=device)
    return model, metadata, device


@st.cache_data(show_spinner=False)
def load_metrics():
    """Carga las métricas de evaluación si están disponibles."""
    metrics_path = PROJECT_ROOT / "reports" / "metrics.json"
    if metrics_path.exists():
        return load_json(metrics_path)
    return None


@st.cache_data(show_spinner=False)
def list_examples() -> list[dict[str, str]]:
    """Lista las imágenes de ejemplo embebidas en la app."""
    if not EXAMPLES_DIR.exists():
        return []
    examples = []
    for path in sorted(EXAMPLES_DIR.glob("*.png")):
        label = path.stem.replace("_", " ")
        true_class = label.split(" - ")[-1].strip()
        examples.append({"path": path, "name": path.name, "true_class": true_class})
    return examples


# --------------------------------------------------------------------------- #
# Encabezado
# --------------------------------------------------------------------------- #
st.title("Clasificador de imágenes con CNN")
st.caption(
    "Red Neuronal Convolucional entrenada desde cero sobre **CIFAR-10** (10 clases, "
    "imágenes 32×32 px). Sube una imagen y el modelo te dirá qué objeto cree que es "
    "y con cuánta confianza."
)

try:
    model, metadata, device = get_model()
except ModelNotAvailableError as error:
    st.error(str(error))
    st.info(
        "**Cómo solucionarlo:**\n\n"
        "```bash\n"
        "python -m src.train            # entrena y guarda models/cnn_cifar10.pt\n"
        "```\n\n"
        "o descarga el checkpoint desde el repositorio en `models/cnn_cifar10.pt`."
    )
    st.stop()

# --------------------------------------------------------------------------- #
# Barra lateral: información del modelo
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("ℹ️ Información del modelo")
    parametros = metadata.get("parameters")
    parametros_texto = f"{parametros:,}" if isinstance(parametros, int) else "—"
    mejor_validacion = metadata.get("best_val_accuracy")
    validacion_texto = f"{mejor_validacion:.2%}" if isinstance(mejor_validacion, (int, float)) else "—"
    st.markdown(
        f"""
- **Arquitectura:** CNN convolucional (3 bloques + GAP + densa)
- **Parámetros entrenables:** {parametros_texto}
- **Clases:** {len(CLASSES)}
- **Dispositivo de inferencia:** {device}
- **Época del mejor modelo:** {metadata.get("best_epoch", "—")}
- **Accuracy de validación:** {validacion_texto}
"""
    )

    metrics = load_metrics()
    if metrics:
        test_metrics = metrics.get("metrics", {})
        st.markdown(
            f"""
**Rendimiento en el conjunto de prueba**
- Accuracy: {test_metrics.get("accuracy", 0):.2%}
- Precision (macro): {test_metrics.get("precision_macro", 0):.2%}
- Recall (macro): {test_metrics.get("recall_macro", 0):.2%}
- F1 (macro): {test_metrics.get("f1_macro", 0):.2%}
"""
        )
    st.divider()
    st.caption("Clases del modelo")
    st.markdown("\n".join(f"- {name}" for name in CLASSES))

# --------------------------------------------------------------------------- #
# Selector de origen de la imagen
# --------------------------------------------------------------------------- #
st.subheader("1. Elige una imagen")

mode = st.radio(
    "Origen de la imagen",
    ["📤 Subir imagen", "🖼️ Usar ejemplo del dataset"],
    horizontal=True,
)

image: Image.Image | None = None
source_label = ""

examples = list_examples()

if mode == "📤 Subir imagen":
    uploaded = st.file_uploader(
        "Sube una imagen (JPG, PNG o WEBP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help="La imagen se redimensiona a 32×32 px antes de clasificarse.",
    )
    if uploaded is not None:
        try:
            image = Image.open(uploaded).convert("RGB")
            source_label = f"Archivo subido: {uploaded.name}"
        except Exception as error:  # noqa: BLE001
            st.error(f"No se pudo leer la imagen subida: {error}")
            image = None
else:
    if not examples:
        st.info("No hay imágenes de ejemplo disponibles en `app/assets/examples/`.")
    else:
        columns = st.columns(len(examples))
        for column, example in zip(columns, examples):
            with column:
                st.image(str(example["path"]), width=110)
                st.caption(example["true_class"])

        names = [example["name"] for example in examples]
        selection = st.selectbox("Ejemplo", names, index=0)
        selected = next(item for item in examples if item["name"] == selection)
        image = Image.open(selected["path"]).convert("RGB")
        source_label = f"Ejemplo: {selected['true_class']}"

# --------------------------------------------------------------------------- #
# Predicción
# --------------------------------------------------------------------------- #
st.subheader("2. Predicción del modelo")

if image is None:
    st.info("👆 Selecciona o sube una imagen para ver la clasificación.")
    st.stop()

left, right = st.columns([1, 1.35])

with left:
    st.image(image, width=320)
    st.caption(source_label)

normalization = metadata.get("normalization", {})
mean = normalization.get("mean")
std = normalization.get("std")

with right:
    with st.spinner("Clasificando..."):
        prediction = predict_image(
            model,
            image,
            device=device,
            mean=mean,
            std=std,
        )
    st.session_state["last_prediction"] = prediction

    confidence = prediction.confidence
    st.metric(
        label="Clase predicha",
        value=prediction.class_name.capitalize(),
        delta=f"Confianza: {confidence:.2%}",
        delta_color="normal",
    )

    if confidence >= 0.8:
        st.success(f"✅ Confianza **alta** ({confidence:.2%}).")
    elif confidence >= 0.5:
        st.warning(f"⚠️ Confianza **media** ({confidence:.2%}).")
    else:
        st.error(f"❌ Confianza **baja** ({confidence:.2%}). La imagen podría no pertenecer a CIFAR-10.")

    st.markdown("**Probabilidad por clase**")
    ordered = prediction.top_k
    for class_name, probability in ordered:
        bar = st.progress(min(float(probability), 1.0))
        bar.progress(min(float(probability), 1.0), text=f"{class_name}: {probability:.2%}")

    st.caption(
        "La imagen se redimensiona a 32×32 px, se normaliza con la media y la desviación "
        "estándar calculadas sobre el conjunto de entrenamiento y se pasa por la red."
    )
