"""Entrenamiento de la CNN sobre CIFAR-10.

Uso:
    python -m src.train --epochs 20
    python -m src.train --epochs 5 --device cpu --num-workers 2
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch import nn

from .config import (
    CHECKPOINT_PATH,
    CLASSES,
    HISTORY_PATH,
    METADATA_PATH,
    PROJECT_ROOT,
    SEED,
    TRAIN_CONFIG,
)
from .data import build_dataloaders
from .model import build_model, model_config
from .utils import (
    count_parameters,
    describe_device,
    get_logger,
    load_json,
    resolve_device,
    save_json,
    set_seed,
)

LOGGER = get_logger()


# --------------------------------------------------------------------------- #
# Utilidades de época
# --------------------------------------------------------------------------- #
def _to_channels_last(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous(memory_format=torch.channels_last)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    grad_clip: float | None = None,
) -> dict[str, float]:
    """Ejecuta una época de entrenamiento (`optimizer` no nulo) o de evaluación."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        images = _to_channels_last(images)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if training:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def train(
    epochs: int = int(TRAIN_CONFIG["epochs"]),  # type: ignore[arg-type]
    batch_size: int = int(TRAIN_CONFIG["batch_size"]),  # type: ignore[arg-type]
    learning_rate: float = float(TRAIN_CONFIG["learning_rate"]),  # type: ignore[arg-type]
    weight_decay: float = float(TRAIN_CONFIG["weight_decay"]),  # type: ignore[arg-type]
    num_workers: int = int(TRAIN_CONFIG["num_workers"]),  # type: ignore[arg-type]
    seed: int = SEED,
    device: str = "auto",
    checkpoint_path: Path = CHECKPOINT_PATH,
    metadata_path: Path = METADATA_PATH,
    history_path: Path = HISTORY_PATH,
    resume_from: Path | None = None,
) -> dict[str, object]:
    """Entrena el modelo y persiste checkpoint, metadatos e historial.

    Si `resume_from` apunta a un checkpoint con estado completo (optimizador,
    scheduler e historial), el entrenamiento continúa desde la última época
    guardada en lugar de empezar de cero.
    """
    set_seed(seed)
    resolved_device = resolve_device(device)
    LOGGER.info("Dispositivo: %s", describe_device(resolved_device))
    LOGGER.info("Semilla: %d", seed)

    checkpoint_path = Path(checkpoint_path)
    last_checkpoint_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_last{checkpoint_path.suffix}")

    bundle = build_dataloaders(batch_size=batch_size, num_workers=num_workers, seed=seed)
    mean, std = bundle.mean, bundle.std

    model = build_model().to(resolved_device)
    model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate * float(TRAIN_CONFIG["max_lr_ratio"]),
        epochs=epochs,
        steps_per_epoch=len(bundle.train_loader),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    history: list[dict[str, float]] = []
    start_epoch = 1
    best_val_accuracy = -1.0
    best_epoch = -1

    if resume_from is not None and Path(resume_from).exists():
        estado = torch.load(resume_from, map_location=resolved_device, weights_only=False)
        model.load_state_dict(estado["state_dict"], strict=True)
        if "optimizer_state" in estado:
            optimizer.load_state_dict(estado["optimizer_state"])
        if "scheduler_state" in estado:
            scheduler.load_state_dict(estado["scheduler_state"])
        history = list(estado.get("history", []))
        start_epoch = int(estado.get("epoch", 0)) + 1
        best_val_accuracy = float(estado.get("val_accuracy", -1.0))
        best_epoch = int(estado.get("best_epoch", estado.get("epoch", 0)))
        LOGGER.info(
            "Reanudando desde %s (época %d, mejor val acc %.4f)",
            resume_from,
            start_epoch,
            best_val_accuracy,
        )
    else:
        LOGGER.info("Entrenamiento desde cero: %d épocas", epochs)

    total_params = count_parameters(model)
    LOGGER.info("Parámetros entrenables: %s", f"{total_params:,}")
    LOGGER.info("Épocas: %d | batch: %d | lr: %g | weight_decay: %g", epochs, batch_size, learning_rate, weight_decay)

    started = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_started = time.time()
        train_metrics = run_epoch(
            model,
            bundle.train_loader,
            criterion,
            resolved_device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=float(TRAIN_CONFIG["grad_clip"]),  # type: ignore[arg-type]
        )
        val_metrics = run_epoch(model, bundle.val_loader, criterion, resolved_device)
        epoch_time = time.time() - epoch_started

        current_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": current_lr,
            "seconds": epoch_time,
        }
        history.append(record)

        LOGGER.info(
            "Época %02d/%02d | train loss %.4f acc %.4f | val loss %.4f acc %.4f | lr %.5f | %.1fs",
            epoch,
            epochs,
            record["train_loss"],
            record["train_accuracy"],
            record["val_loss"],
            record["val_accuracy"],
            current_lr,
            epoch_time,
        )

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_config": model_config(model),
                    "normalization": {"mean": list(mean), "std": list(std)},
                    "epoch": epoch,
                    "val_accuracy": best_val_accuracy,
                },
                checkpoint_path,
            )
            LOGGER.info("  -> Nuevo mejor modelo (val acc %.4f) guardado en %s", best_val_accuracy, checkpoint_path)

        # Estado completo de la última época: permite reanudar el entrenamiento
        # si se interrumpe. No se versiona (ver .gitignore).
        torch.save(
            {
                "state_dict": model.state_dict(),
                "model_config": model_config(model),
                "normalization": {"mean": list(mean), "std": list(std)},
                "epoch": epoch,
                "val_accuracy": best_val_accuracy,
                "best_epoch": best_epoch,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "history": history,
            },
            last_checkpoint_path,
        )

    total_time = time.time() - started
    LOGGER.info("Entrenamiento finalizado en %.1f min", total_time / 60)

    metadata: dict[str, object] = {
        "model_config": model_config(model),
        "train_config": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "optimizer": "AdamW",
            "scheduler": "OneCycleLR",
            "criterion": "CrossEntropyLoss",
            "num_workers": num_workers,
            "seed": seed,
        },
        "normalization": {"mean": list(mean), "std": list(std)},
        "classes": list(CLASSES),
        "dataset": {
            "name": "CIFAR-10",
            "split": {
                "train": len(bundle.train_dataset),
                "val": len(bundle.val_dataset),
                "test": len(bundle.test_dataset),
            },
        },
        "parameters": total_params,
        "device": describe_device(resolved_device),
        "torch_version": torch.__version__,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "training_seconds": total_time,
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
    }

    save_json(metadata, metadata_path)
    save_json(history, history_path)
    LOGGER.info("Metadatos guardados en %s", metadata_path)
    LOGGER.info("Historial guardado en %s", history_path)

    return {"metadata": metadata, "history": history}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena la CNN sobre CIFAR-10")
    parser.add_argument("--epochs", type=int, default=int(TRAIN_CONFIG["epochs"]))
    parser.add_argument("--batch-size", type=int, default=int(TRAIN_CONFIG["batch_size"]))
    parser.add_argument("--lr", type=float, default=float(TRAIN_CONFIG["learning_rate"]))
    parser.add_argument("--weight-decay", type=float, default=float(TRAIN_CONFIG["weight_decay"]))
    parser.add_argument("--num-workers", type=int, default=int(TRAIN_CONFIG["num_workers"]))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--history", type=Path, default=HISTORY_PATH)
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Checkpoint con estado completo desde el que continuar el entrenamiento",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        checkpoint_path=args.checkpoint,
        metadata_path=args.metadata,
        history_path=args.history,
        resume_from=args.resume_from,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_epoch", "train"]
