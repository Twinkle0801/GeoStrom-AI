"""Phase 6 compute-safety check: run BEFORE any real training.

Per the task's explicit instruction: "Before starting expensive training:
inspect available CPU/GPU resources, estimate training time, run a small
smoke test first." This script does all three, and exits non-zero if
something looks unsafe (e.g. no working forward/backward pass) rather than
letting a real training script fail expensively.

Usage:
    python ml/scripts/dl_smoke_test.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> int:
    try:
        import torch
    except ImportError:
        print("torch is not installed -- install ml/requirements-deep-learning.txt first.",
              file=sys.stderr)
        return 1

    print(f"torch version: {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"  device: {torch.cuda.get_device_name(0)}")
        print(f"  total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("  No GPU detected -- falling back to CPU-safe path (small batch, small model).")
    device = "cuda" if cuda_ok else "cpu"

    from ml.geostrom_ml.classification.deep.models import build_model, count_trainable_parameters

    for model_name in ("small_cnn", "resnet18"):
        model = build_model(model_name).to(device)
        trainable, total = count_trainable_parameters(model)
        print(f"\n{model_name}: {trainable:,} trainable / {total:,} total parameters")

        batch = torch.randn(4, 1, 224, 224, device=device)
        t0 = time.time()
        with torch.no_grad():
            out = model(batch)
        forward_s = time.time() - t0
        assert out.shape == (4, 4), f"unexpected output shape {out.shape}"
        print(f"  forward pass (batch=4): {forward_s * 1000:.1f} ms, output shape {tuple(out.shape)}")

        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        labels = torch.randint(0, 4, (4,), device=device)
        t0 = time.time()
        optimizer.zero_grad()
        loss = criterion(model(batch), labels)
        loss.backward()
        optimizer.step()
        backward_s = time.time() - t0
        print(f"  backward pass (batch=4): {backward_s * 1000:.1f} ms, loss={loss.item():.3f}")

        # 353 real training samples -> ~22 batches/epoch at batch_size=16
        est_epoch_s = (forward_s + backward_s) * (353 / 4)
        print(f"  estimated time per epoch (353 train samples, extrapolated): {est_epoch_s:.1f} s")
        print(f"  estimated time for 40 epochs: {est_epoch_s * 40 / 60:.1f} min")

    print("\nSmoke test PASSED: forward+backward pass works for both architectures on "
          f"device={device}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
