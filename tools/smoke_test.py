"""Prueba de humo: valida el pipeline completo con unos pocos batches."""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CHECKPOINT_PATH, METADATA_PATH
from src.data import build_dataloaders
from src.evaluate import collect_predictions, compute_metrics
from src.model import build_model
from src.predict import load_model, predict_image
from src.train import run_epoch
from src.utils import count_parameters, set_seed


def main() -> None:
    set_seed(42)
    torch.set_num_threads(12)

    bundle = build_dataloaders(batch_size=128, num_workers=2, seed=42)
    print("train/val/test:", len(bundle.train_dataset), len(bundle.val_dataset), len(bundle.test_dataset))
    print("mean", bundle.mean, "std", bundle.std, flush=True)
    print("counts train", bundle.class_counts["train"], flush=True)

    model = build_model()
    print("params", count_parameters(model), flush=True)
    x = torch.randn(4, 3, 32, 32)
    print("forward:", model(x).shape, flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    small_train = Subset(bundle.train_dataset, list(range(256)))
    small_val = Subset(bundle.val_dataset, list(range(256)))
    small_test = Subset(bundle.test_dataset, list(range(256)))
    train_loader = DataLoader(small_train, batch_size=128)
    val_loader = DataLoader(small_val, batch_size=128)
    test_loader = DataLoader(small_test, batch_size=128)

    m = run_epoch(model, train_loader, criterion, torch.device("cpu"), optimizer=optimizer, scheduler=None)
    v = run_epoch(model, val_loader, criterion, torch.device("cpu"))
    print("train metrics", m, "val metrics", v, flush=True)

    targets, predictions, probs = collect_predictions(model, test_loader, torch.device("cpu"))
    metrics = compute_metrics(targets, predictions)
    print("metrics", {k: metrics[k] for k in ("accuracy", "precision_macro", "recall_macro", "f1_macro")}, flush=True)

    tmp = Path("scratch/smoke.pt")
    tmp.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "num_classes": 10,
                "in_channels": 3,
                "channels": [64, 128, 256],
                "dropout2d": 0.15,
                "dropout_head": 0.5,
            },
        },
        tmp,
    )
    loaded, meta = load_model(tmp, metadata_path=Path("nope.json"))
    loaded.eval()
    print("reload ok, meta:", meta, flush=True)

    img = Image.fromarray((np.random.rand(32, 32, 3) * 255).astype("uint8"))
    pred = predict_image(loaded, img)
    print("pred:", pred.class_name, pred.confidence, len(pred.probabilities), flush=True)
    print("SMOKE OK", flush=True)


if __name__ == "__main__":
    main()
