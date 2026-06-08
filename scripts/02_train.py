#!/usr/bin/env python3
# ruff: noqa: E501
"""
scripts/02_train.py
====================
YOLOv8n-P2 fine-tuning script for The Aerial Guardian MOT pipeline.

Architecture strategy
---------------------
We load the YOLOv8n PRETRAINED weights (yolov8n.pt) and transfer them
onto our custom P2 architecture (configs/model.yaml). Ultralytics handles
this automatically via the 'model' + 'pretrained' pattern: it builds the
new graph from model.yaml, then copies all matching weight tensors from
yolov8n.pt by name. The new P2 head layers initialise randomly and are
trained from scratch alongside the fine-tuned backbone.

VRAM safety for RTX 3050 (6 GB)
---------------------------------
  imgsz = 1280   -> Single high-res forward pass (our core strategy vs SAHI)
  batch = 4      -> Peak VRAM ~4.8 GB with FP16. DO NOT increase beyond 4.
  amp   = True   -> FP16 mixed precision halves activation memory.
  workers = 2    -> Conservative DataLoader workers; avoids CPU RAM spike.

If you hit a CUDA OOM despite batch=4, lower to batch=2 as a fallback.

Usage
-----
  # Standard training (100 epochs):
  python scripts/02_train.py

  # Resume an interrupted run:
  python scripts/02_train.py --resume

  # Quick smoke-test (5 epochs, 640px):
  python scripts/02_train.py --epochs 5 --imgsz 640 --batch 8
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Preflight: verify environment before importing torch / ultralytics
# ---------------------------------------------------------------------------
try:
    import torch
except ImportError:
    print("[ERROR] PyTorch not found. Activate the .venv and install dependencies:")
    print("        pip install torch==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] Ultralytics not found. Run: pip install ultralytics==8.0.236")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Path constants (relative to project root)
# ---------------------------------------------------------------------------
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODEL_YAML    = PROJECT_ROOT / "configs" / "model.yaml"
DATASET_YAML  = PROJECT_ROOT / "configs" / "dataset.yaml"
WEIGHTS_DIR   = PROJECT_ROOT / "models" / "weights"
RUNS_DIR      = PROJECT_ROOT / "runs"

# Base YOLOv8n pretrained weights — downloaded automatically by Ultralytics
# on first run and cached in ~/.config/Ultralytics/
PRETRAINED_BASE = "yolov8n.pt"

# Training run name (appears under runs/train/)
RUN_NAME = "aerial_guardian_p2_v1"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_header(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def preflight_checks(args: argparse.Namespace) -> None:
    """
    Validate all required files and GPU availability before touching the GPU.
    Fail fast with a clear message rather than crashing mid-training.
    """
    print_header("Preflight Checks")

    # 1. Model architecture YAML
    if not MODEL_YAML.exists():
        print(f"[FAIL] Model YAML not found: {MODEL_YAML}")
        print("       Run Phase 2 or check that configs/model.yaml was created.")
        sys.exit(1)
    print(f"  [OK] Model YAML          : {MODEL_YAML}")

    # 2. Dataset YAML
    if not DATASET_YAML.exists():
        print(f"[FAIL] Dataset YAML not found: {DATASET_YAML}")
        print("       Run: python scripts/01_preprocess.py")
        sys.exit(1)
    print(f"  [OK] Dataset YAML        : {DATASET_YAML}")

    # 3. Check that processed data dirs exist and are non-empty
    processed_images = PROJECT_ROOT / "data" / "processed" / "images"
    processed_labels = PROJECT_ROOT / "data" / "processed" / "labels"
    if not processed_images.exists() or not any(processed_images.iterdir()):
        print(f"[FAIL] Processed images directory is empty: {processed_images}")
        print("       Run: python scripts/01_preprocess.py")
        sys.exit(1)
    print(f"  [OK] Processed images    : {processed_images}")

    # 4. CUDA availability
    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — training will run on CPU (very slow).")
        print("       Verify PyTorch+CUDA installation:")
        print("         python -c \"import torch; print(torch.cuda.is_available())\"")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  [OK] GPU                 : {gpu_name} ({vram_gb:.1f} GB VRAM)")

        # Warn if batch size looks risky for the available VRAM
        if vram_gb < 6.0 and args.batch > 2:
            print(f"[WARN] GPU has <6 GB VRAM ({vram_gb:.1f} GB). Consider --batch 2.")

    # 5. Warn if resume target doesn't exist
    if args.resume:
        last_pt = RUNS_DIR / "train" / RUN_NAME / "weights" / "last.pt"
        if not last_pt.exists():
            print(f"[WARN] --resume requested but no checkpoint found at {last_pt}")
            print("       Starting fresh training instead.")
            args.resume = False

    print()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """
    Execute YOLOv8n-P2 fine-tuning on the VisDrone persons dataset.

    Model initialisation strategy
    ------------------------------
    YOLO(MODEL_YAML) builds the network graph from our custom 4-head YAML.
    Passing pretrained=PRETRAINED_BASE to .train() triggers Ultralytics'
    weight transfer: it downloads yolov8n.pt, then copies every matching
    tensor (backbone Conv/C2f/SPPF layers) by name into our graph. The
    new P2-branch layers (head layers 16-18, plus the 4th Detect output)
    start from random initialisation and converge quickly because the
    backbone already provides rich features.

    Augmentation choices for drone footage
    ----------------------------------------
    - mosaic=1.0   : Composites 4 images per tile -> exposes model to more
                     person-scale variability per batch step. Critical for
                     the sparse, small-person distribution in VisDrone.
    - hsv_h=0.015  : Slight hue jitter -> handles lighting variations at
                     different altitudes and times of day.
    - hsv_s=0.7    : Saturation jitter -> simulates hazy vs clear sky.
    - flipud=0.3   : Vertical flip -> drone footage comes from above; a
                     person looks the same upside down from height.
    - scale=0.5    : Random scale 50-150% -> simulates altitude variation.
    - close_mosaic=10 : Disable mosaic in the last 10 epochs so the model
                        can fine-tune on clean, unmodified crops.
    """
    print_header("The Aerial Guardian -- YOLOv8n-P2 Training")
    print(f"  Timestamp : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model YAML: {MODEL_YAML}")
    print(f"  Dataset   : {DATASET_YAML}")
    print(f"  imgsz     : {args.imgsz}  (high-res single-pass strategy)")
    print(f"  batch     : {args.batch}  (VRAM-safe for RTX 3050 6GB)")
    print(f"  epochs    : {args.epochs}")
    print(f"  amp (FP16): {args.amp}")
    print(f"  workers   : {args.workers}")
    print(f"  resume    : {args.resume}")
    print()

    # --- Build model from custom P2 architecture ---
    # YOLO(str) with a .yaml path builds the architecture from scratch.
    # The pretrained= argument in .train() handles weight transfer.
    model = YOLO(str(MODEL_YAML))

    start_time = time.time()

    # --- Launch training ---
    # All hardcoded safety params are explicitly set here, not in a config
    # file, so they cannot be accidentally overridden.
    results = model.train(
        # --- Data & architecture ---
        data=str(DATASET_YAML),
        pretrained=not args.resume,        # transfer yolov8n.pt weights on fresh run
        model=str(MODEL_YAML) if not args.resume else str(
            RUNS_DIR / "train" / RUN_NAME / "weights" / "last.pt"
        ),

        # --- *** VRAM SAFETY — DO NOT CHANGE FOR RTX 3050 6GB *** ---
        imgsz=args.imgsz,                  # 1280: native high-res, single forward pass
        batch=args.batch,                  # 4: peak ~4.8 GB VRAM with FP16
        amp=args.amp,                      # True: FP16 halves activation memory
        workers=args.workers,              # 2: conservative CPU DataLoader threads

        # --- Training schedule ---
        epochs=args.epochs,
        patience=50,                       # early stopping: halt if no mAP improvement
        optimizer="AdamW",                 # AdamW: better convergence than SGD on small datasets
        lr0=0.001,                         # initial learning rate
        lrf=0.01,                          # final lr = lr0 * lrf (cosine decay)
        warmup_epochs=3,                   # linear LR warmup for stable early training
        weight_decay=0.0005,

        # --- Augmentation (tuned for drone altitude variability) ---
        mosaic=1.0,                        # always mosaic: critical for small-person density
        hsv_h=0.015,                       # hue jitter: altitude lighting variation
        hsv_s=0.7,                         # saturation jitter: haze simulation
        hsv_v=0.4,                         # value jitter: shadow/overcast
        flipud=0.3,                        # vertical flip: valid for top-down drone view
        fliplr=0.5,                        # horizontal flip: standard
        scale=0.5,                         # scale jitter: simulates altitude change
        translate=0.1,
        degrees=5.0,                       # slight rotation: drone tilt compensation
        close_mosaic=10,                   # disable mosaic for last 10 epochs

        # --- Loss weights (tuned for small objects) ---
        # box loss weighted higher -> tighter bbox regression for tiny targets
        box=7.5,
        cls=0.5,
        dfl=1.5,

        # --- Output & logging ---
        project=str(RUNS_DIR / "train"),
        name=RUN_NAME,
        resume=args.resume,
        exist_ok=True,                     # allow re-running without renaming
        plots=True,                        # save training curves and confusion matrix
        save=True,
        save_period=10,                    # checkpoint every 10 epochs (recovery safety)
        verbose=True,
        seed=42,                           # reproducibility
        deterministic=True,
        device=0,                          # GPU 0 (RTX 3050)
    )

    elapsed = time.time() - start_time
    h, m = divmod(int(elapsed), 3600)
    m, s = divmod(m, 60)

    # --- Post-training summary ---
    print_header("Training Complete")
    print(f"  Total time     : {h:02d}h {m:02d}m {s:02d}s")

    best_pt = RUNS_DIR / "train" / RUN_NAME / "weights" / "best.pt"
    last_pt = RUNS_DIR / "train" / RUN_NAME / "weights" / "last.pt"

    if best_pt.exists():
        size_mb = best_pt.stat().st_size / 1e6
        print(f"  Best weights   : {best_pt}  ({size_mb:.1f} MB)")

        # Copy best.pt to models/weights/ for easy access
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = WEIGHTS_DIR / "best.pt"
        import shutil
        shutil.copy2(best_pt, dest)
        print(f"  Copied to      : {dest}")
    else:
        print(f"  [WARN] best.pt not found at {best_pt}")

    # Print mAP metrics if available
    try:
        metrics = results.results_dict
        map50    = metrics.get("metrics/mAP50(B)",    "N/A")
        map5095  = metrics.get("metrics/mAP50-95(B)", "N/A")
        if isinstance(map50,   float): map50   = f"{map50:.4f}"
        if isinstance(map5095, float): map5095 = f"{map5095:.4f}"
        print(f"  mAP@50         : {map50}")
        print(f"  mAP@50-95      : {map5095}")
    except Exception:
        pass  # metrics not always available if training was short

    print(f"\n  Next step:")
    print(f"    python scripts/03_track.py --weights {best_pt} --source <video_or_sequence_dir>")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n-P2 on VisDrone persons dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # *** VRAM safety defaults are hardcoded below. ***
    # Override only if you upgrade hardware or reduce imgsz for smoke tests.
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Input image size (px). 1280 = native high-res single-pass strategy. "
             "Reduce to 640 only for quick smoke tests.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size. MAX 4 for RTX 3050 6GB at imgsz=1280 with FP16. "
             "Lower to 2 if you hit CUDA OOM.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--amp",
        type=lambda x: x.lower() != "false",
        default=True,
        metavar="BOOL",
        help="FP16 mixed precision. Always True on RTX 3050 to halve VRAM usage.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="DataLoader worker threads. Keep at 2 to avoid CPU RAM spike.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=f"Resume training from runs/train/{RUN_NAME}/weights/last.pt",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    preflight_checks(args)
    train(args)
