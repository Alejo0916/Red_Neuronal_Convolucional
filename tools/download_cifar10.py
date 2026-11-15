"""Descarga reanudable de CIFAR-10 (formato torchvision)."""

from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
ARCHIVE_NAME = "cifar-10-python.tar.gz"
EXPECTED_BYTES = 170_498_071


def download_with_resume(url: str, dest: Path, retries: int = 200) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        actual = dest.stat().st_size if dest.exists() else 0
        if actual >= EXPECTED_BYTES:
            return dest
        headers = {"User-Agent": "Mozilla/5.0"}
        if actual:
            headers["Range"] = f"bytes={actual}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                mode = "ab" if actual and response.status == 206 else "wb"
                if mode == "wb":
                    actual = 0
                total = actual + int(response.headers.get("Content-Length", 0))
                print(
                    f"[intento {attempt}] reanudando en {actual / 1e6:.1f} MB / {EXPECTED_BYTES / 1e6:.1f} MB",
                    flush=True,
                )
                with open(dest, mode) as handle:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                        actual += len(chunk)
                        if actual % (10 << 20) < (1 << 20):
                            print(f"  -> {actual / 1e6:.1f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  error de red: {exc}. Reintentando en 3 s...", flush=True)
            time.sleep(3)
    raise RuntimeError("No se pudo completar la descarga de CIFAR-10")


def main() -> int:
    parser = argparse.ArgumentParser(description="Descarga CIFAR-10 con reanudación")
    parser.add_argument("--root", default="data", help="Directorio destino del dataset")
    args = parser.parse_args()

    root = Path(args.root)
    archive = root / ARCHIVE_NAME
    download_with_resume(CIFAR_URL, archive)
    print("Descarga completa; extrayendo...", flush=True)

    import tarfile

    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)
    print("Dataset listo en", root, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
