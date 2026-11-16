"""Pruebas del pipeline de la CNN sobre CIFAR-10.

Ejecución:
    python -m pytest tests -v

Las pruebas que necesitan el dataset se omiten automáticamente si `data/` no
contiene CIFAR-10 descargado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    CHECKPOINT_PATH,
    CLASSES,
    IMG_SIZE,
    METADATA_PATH,
    NUM_CLASSES,
    SEED,
)
from src.data import (  # noqa: E402
    build_transforms,
    compute_channel_stats,
    denormalize,
    stratified_split,
)
from src.model import build_model, model_config  # noqa: E402
from src.predict import ModelNotAvailableError, load_model, predict_image, preprocess_image  # noqa: E402
from src.utils import count_parameters, set_seed  # noqa: E402

DATASET_DISPONIBLE = (PROJECT_ROOT / "data" / "cifar-10-batches-py").exists()
requiere_dataset = pytest.mark.skipif(
    not DATASET_DISPONIBLE, reason="CIFAR-10 no está descargado en data/"
)


# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
def test_clases_cifar10():
    assert len(CLASSES) == NUM_CLASSES == 10
    assert CLASSES[0] == "avión"
    assert CLASSES[-1] == "camión"


def test_semilla_por_defecto():
    assert SEED == 42


# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
def test_forward_devuelve_10_logits():
    model = build_model()
    salida = model(torch.randn(4, 3, IMG_SIZE, IMG_SIZE))
    assert salida.shape == (4, NUM_CLASSES)


def test_numero_de_parametros_en_rango():
    total = count_parameters(build_model())
    assert 1_000_000 <= total <= 10_000_000, total


def test_config_del_modelo_es_serializable():
    config = model_config(build_model())
    json.dumps(config)
    assert config["channels"] == [64, 128, 256]


def test_dropout_desactivado_en_evaluacion():
    model = build_model().eval()
    imagen = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(imagen), model(imagen))


def test_pesos_reproducibles_con_la_misma_semilla():
    set_seed(SEED)
    primero = build_model()
    set_seed(SEED)
    segundo = build_model()
    for a, b in zip(primero.state_dict().values(), segundo.state_dict().values()):
        assert torch.equal(a, b)


# --------------------------------------------------------------------------- #
# Datos
# --------------------------------------------------------------------------- #
def test_division_estratificada_reproducible():
    targets = list(range(10)) * 100
    train_a, val_a = stratified_split(targets, val_size=200, seed=SEED)
    train_b, val_b = stratified_split(targets, val_size=200, seed=SEED)
    assert train_a == train_b
    assert val_a == val_b


def test_division_es_disjunta_y_cubre_todo():
    targets = list(range(10)) * 100
    train_idx, val_idx = stratified_split(targets, val_size=200, seed=SEED)
    assert set(train_idx).isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == len(targets)


def test_transforms_generan_tensor_correcto():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_t, val_t, test_t = build_transforms(mean, std)
    imagen = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    from PIL import Image

    tensor = train_t(Image.fromarray(imagen))
    assert tensor.shape == (3, IMG_SIZE, IMG_SIZE)
    assert tensor.dtype == torch.float32
    assert val_t(Image.fromarray(imagen)).shape == (3, IMG_SIZE, IMG_SIZE)
    assert test_t(Image.fromarray(imagen)).shape == (3, IMG_SIZE, IMG_SIZE)


def test_augmentation_solo_en_entrenamiento():
    from torchvision import transforms

    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_t, val_t, test_t = build_transforms(mean, std)
    assert any(isinstance(t, transforms.RandomCrop) for t in train_t.transforms)
    assert any(isinstance(t, transforms.RandomHorizontalFlip) for t in train_t.transforms)
    assert not any(isinstance(t, transforms.RandomCrop) for t in val_t.transforms)
    assert not any(isinstance(t, transforms.RandomHorizontalFlip) for t in test_t.transforms)
    assert not any(isinstance(t, transforms.RandomCrop) for t in test_t.transforms)


def transforms_normalize(tensor: torch.Tensor, mean, std) -> torch.Tensor:
    media = torch.tensor(mean).view(-1, 1, 1)
    desviacion = torch.tensor(std).view(-1, 1, 1)
    return (tensor - media) / desviacion


def test_denormalize_invierte_normalizacion():
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    original = torch.rand(3, IMG_SIZE, IMG_SIZE)
    tensor = transforms_normalize(original, mean, std)
    recuperado = denormalize(tensor, mean, std)
    # `denormalize` devuelve HWC; se reordena a CHW para comparar.
    assert recuperado.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert np.allclose(recuperado.transpose(2, 0, 1), original.numpy(), atol=1e-5)


@requiere_dataset
def test_estadisticas_solo_sobre_entrenamiento():
    from src.data import load_cifar10

    dataset = load_cifar10(download=False)
    idx_train, _ = stratified_split(dataset.targets)
    mean, std = compute_channel_stats(dataset, idx_train)
    assert len(mean) == len(std) == 3
    assert all(0.0 < valor < 1.0 for valor in mean)
    assert all(valor > 0.0 for valor in std)


# --------------------------------------------------------------------------- #
# Inferencia
# --------------------------------------------------------------------------- #
def test_preprocess_devuelve_tensor_con_lote_1():
    from PIL import Image

    imagen = Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
    tensor = preprocess_image(imagen)
    assert tensor.shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_load_model_error_claro_si_falta_checkpoint(tmp_path):
    with pytest.raises(ModelNotAvailableError) as error:
        load_model(tmp_path / "no_existe.pt", metadata_path=tmp_path / "meta.json")
    assert "no_existe.pt" in str(error.value)


def test_load_model_error_claro_si_checkpoint_incompatible(tmp_path):
    ruta = tmp_path / "roto.pt"
    torch.save({"state_dict": {"pesos": torch.zeros(1)}}, ruta)
    with pytest.raises(ModelNotAvailableError):
        load_model(ruta, metadata_path=tmp_path / "meta.json")


def test_prediccion_devuelve_diez_probabilidades():
    from PIL import Image

    model = build_model().eval()
    imagen = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    prediccion = predict_image(model, imagen, device="cpu")
    assert len(prediccion.probabilities) == NUM_CLASSES
    assert 0.0 <= prediccion.confidence <= 1.0
    assert abs(sum(prediccion.probabilities) - 1.0) < 1e-5
    assert prediccion.class_name in CLASSES


def test_checkpoint_guardado_se_puede_recargar(tmp_path):
    model = build_model().eval()
    ruta = tmp_path / "modelo.pt"
    torch.save({"state_dict": model.state_dict(), "model_config": model_config(model)}, ruta)

    cargado, _ = load_model(ruta, metadata_path=tmp_path / "meta.json")
    imagen = torch.randn(2, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        assert torch.allclose(model(imagen), cargado(imagen), atol=1e-5)


# --------------------------------------------------------------------------- #
# Artefactos del repositorio
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="No hay modelo entrenado en models/")
def test_modelo_del_repositorio_carga_y_predice():
    from PIL import Image

    model, metadata = load_model(CHECKPOINT_PATH, METADATA_PATH, device="cpu")
    imagen = Image.fromarray(np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8))
    prediccion = predict_image(model, imagen, device="cpu")
    assert prediccion.class_name in CLASSES
    assert metadata["classes"] == list(CLASSES)
