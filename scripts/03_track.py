#!/usr/bin/env python3
"""
scripts/03_track.py  (v3 — threaded I/O for >15 FPS)
======================================================
The Aerial Guardian — Main MOT Inference Pipeline.

Pipeline per frame
------------------
1. Load frame (threaded prefetch — overlaps disk I/O with GPU inference).
2. Run YOLOv8n-P2 detector at imgsz=1280 (single forward pass, NO SAHI).
3. Run OpenCV ECC on 320x180 downscaled grayscale → affine warp matrix.
4. Apply warp to ByteTrack Kalman-predicted positions (CMC).
5. ByteTrack 3-stage association → confirmed track list.
6. Render: bounding boxes, ID labels, trajectory tails, FPS HUD.
7. Write annotated frame (threaded async writer).

Performance design
------------------
Profiling showed the GPU pipeline takes ~38ms/frame (theoretical 26 FPS),
but actual was 11 FPS because cv2.imread (disk read) and VideoWriter.write
(JPEG compression) were blocking the GPU — adding ~50ms of dead time.

Fix: a prefetch thread reads frames ahead of the GPU loop, and a writer
thread compresses+saves frames in the background. The GPU stays busy.

Usage
-----
  python scripts/03_track.py --weights models/weights/best.pt \\
      --source data/raw/sequences/uav0000086_00000_v

  python scripts/03_track.py --weights models/weights/best.pt \\
      --source path/to/video.mp4 --no-cmc
"""

import argparse
import sys
import time
import signal
import threading
import queue
from collections import deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from ultralytics import YOLO
except ImportError:
    print("[ERROR] ultralytics not found. Activate .venv.")
    sys.exit(1)

from src.tracker.byte_tracker import ByteTracker, STrack
from src.tracker.ecc_compensator import ECCCompensator
from src.utils.visualizer import TrackVisualizer

OUT_DIR = PROJECT_ROOT / "outputs" / "videos"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Threaded Frame Prefetcher
# Reads frames from disk in a background thread so the GPU loop never waits
# for cv2.imread (which blocks on JPEG decompression for ~8-15ms per frame).
# ---------------------------------------------------------------------------

class FramePrefetcher:
    """
    Reads frames ahead of the inference loop in a daemon thread.
    Maintains a fixed-size buffer so memory stays bounded.
    Yields (frame_id, bgr_ndarray) tuples.
    """

    def __init__(self, source: Path, buffer_size: int = 8):
        self._q: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.width = self.height = 0
        self.fps_native = 30.0
        self.total_frames = 0
        self._img_paths: list = []
        self._cap = None
        self._done = False

        if source.is_dir():
            self._img_paths = sorted(
                list(source.glob("*.jpg")) + list(source.glob("*.png"))
            )
            if not self._img_paths:
                raise FileNotFoundError(f"No images in {source}")
            sample = cv2.imread(str(self._img_paths[0]))
            self.height, self.width = sample.shape[:2]
            self.total_frames = len(self._img_paths)
        elif source.is_file():
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise FileNotFoundError(f"Cannot open: {source}")
            self.width      = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height     = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps_native = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        if self._cap is not None:
            idx = 0
            while True:
                ok, frame = self._cap.read()
                if not ok:
                    break
                idx += 1
                self._q.put((idx, frame))
            self._cap.release()
        else:
            for idx, p in enumerate(self._img_paths, 1):
                frame = cv2.imread(str(p))
                self._q.put((idx, frame))
        self._done = True
        self._q.put(None)  # sentinel

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is None:
            raise StopIteration
        return item


# ---------------------------------------------------------------------------
# Threaded Video Writer
# Compresses and writes frames in a background thread so the GPU loop
# never blocks waiting for VideoWriter (JPEG encoding takes ~8-12ms).
# ---------------------------------------------------------------------------

class AsyncWriter:
    """Writes annotated frames to MP4 in a background thread."""

    def __init__(self, path: Path, fps: float, size: tuple):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        self._q: queue.Queue = queue.Queue(maxsize=16)
        self._thread = threading.Thread(target=self._write_loop, daemon=True)
        self._thread.start()

    def _write_loop(self):
        while True:
            frame = self._q.get()
            if frame is None:
                break
            self._writer.write(frame)

    def write(self, frame: np.ndarray):
        self._q.put(frame.copy())  # copy to avoid mutation while writing

    def release(self):
        self._q.put(None)   # sentinel
        self._thread.join(timeout=30)
        self._writer.release()


# ---------------------------------------------------------------------------
# FPS meter
# ---------------------------------------------------------------------------

class FPSMeter:
    def __init__(self, window: int = 60):
        self._times: deque = deque(maxlen=window)
        self._t0 = None

    def start(self):
        self._t0 = time.perf_counter()

    def stop(self) -> float:
        if self._t0 is None:
            return 0.0
        self._times.append(time.perf_counter() - self._t0)
        self._t0 = None
        return self.current

    @property
    def current(self) -> float:
        if len(self._times) < 2:
            return 0.0
        return len(self._times) / sum(self._times)


# ---------------------------------------------------------------------------
# Detection parsing
# ---------------------------------------------------------------------------

def parse_detections(results, conf_thresh: float = 0.05) -> np.ndarray:
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    xywh  = boxes.xywh.cpu().numpy().astype(np.float32)
    confs = boxes.conf.cpu().numpy().astype(np.float32)
    mask  = confs >= conf_thresh
    if not np.any(mask):
        return np.zeros((0, 5), dtype=np.float32)
    return np.concatenate([xywh[mask], confs[mask, None]], axis=1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> None:
    source  = Path(args.source)
    weights = Path(args.weights)

    if not weights.exists():
        print(f"[ERROR] Weights not found: {weights}")
        sys.exit(1)

    print(f"\n{'='*62}")
    print("  The Aerial Guardian — MOT Inference Pipeline")
    print(f"{'='*62}")
    print(f"  Weights  : {weights}  ({weights.stat().st_size/1e6:.1f} MB)")
    print(f"  Source   : {source}")
    print(f"  imgsz    : {args.imgsz}  (native high-res, single forward pass)")
    print(f"  CMC/ECC  : {'enabled (320x180)' if args.cmc else 'DISABLED'}")
    print(f"  I/O      : threaded prefetch + async writer")
    print(f"  track_thresh : {args.track_thresh}")
    print(f"  max_lost     : {args.max_lost} frames")
    print(f"{'='*62}\n")

    # --- Load & fuse model ---
    model = YOLO(str(weights))
    model.fuse()

    # Warmup GPU (avoids cold-start penalty on first real frame)
    dummy = np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
    for _ in range(2):
        model.predict(dummy, imgsz=args.imgsz, conf=0.5,
                      half=True, verbose=False, device=0)

    # --- Init tracker components ---
    tracker  = ByteTracker(
        track_thresh=args.track_thresh,
        low_thresh=args.low_thresh,
        match_thresh=args.match_thresh,
        max_time_lost=args.max_lost,
    )
    cmc      = ECCCompensator(downscale_wh=(320, 180)) if args.cmc else None
    renderer = TrackVisualizer(trail_len=args.trail_len)

    # --- Open threaded frame source ---
    try:
        src = FramePrefetcher(source, buffer_size=8)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # --- Open threaded writer ---
    seq_name = source.stem
    out_path = OUT_DIR / f"{seq_name}_tracked.mp4"
    writer   = AsyncWriter(out_path, src.fps_native, (src.width, src.height))

    # --- Graceful Ctrl+C ---
    interrupted = False
    def _sig(sig, frm):
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] Ctrl+C — flushing writer and exiting...")
    signal.signal(signal.SIGINT, _sig)

    fps_meter   = FPSMeter(window=60)
    all_fps: list = []
    frame_count = 0
    max_tracks  = 0
    STrack._id_counter = 0   # reset for clean ID count

    print(f"  Processing {src.total_frames} frames...\n")

    try:
        for frame_id, frame in src:
            if interrupted:
                break

            fps_meter.start()

            # 1. YOLOv8n-P2 inference — single pass at imgsz=1280, FP16
            results = model.predict(
                frame,
                imgsz=args.imgsz,
                conf=0.05,
                iou=0.45,
                classes=[0],
                half=True,        # FP16: ~30% faster on Ampere
                verbose=False,
                device=0,
            )
            dets = parse_detections(results, conf_thresh=0.05)

            # 2. ECC Camera Motion Compensation
            warp = cmc.update(frame) if cmc is not None else None

            # 3. ByteTrack association
            tracks = tracker.update(dets, warp=warp)

            # 4. Render
            annotated = renderer.draw(frame, tracks, fps_meter.current, frame_id)

            # 5. Async write (non-blocking)
            writer.write(annotated)

            current_fps = fps_meter.stop()
            all_fps.append(current_fps)
            frame_count += 1
            max_tracks = max(max_tracks, len(tracks))

            if frame_id % 30 == 0 or frame_id == 1:
                print(
                    f"  Frame {frame_id:5d}/{src.total_frames} | "
                    f"FPS: {current_fps:5.1f} | "
                    f"Dets: {len(dets):3d} | "
                    f"Tracks: {len(tracks):3d} | "
                    f"IDs so far: {STrack._id_counter}"
                )

    finally:
        writer.release()   # flush remaining frames before exit

    avg_fps = float(np.mean(all_fps[10:])) if len(all_fps) > 10 else float(np.mean(all_fps) if all_fps else [0])
    total_ids = STrack._id_counter

    print(f"\n{'='*62}")
    print("  Tracking Complete")
    print(f"{'='*62}")
    print(f"  Frames processed : {frame_count}")
    print(f"  Average FPS      : {avg_fps:.1f}  "
          f"({'✓ target met' if avg_fps >= 15 else '✗ below 15 FPS'})")
    print(f"  Max simultaneous : {max_tracks} tracks")
    print(f"  Total unique IDs : {total_ids}")
    print(f"  Output video     : {out_path}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="The Aerial Guardian — MOT inference pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--weights",     type=str, default="models/weights/best.pt")
    p.add_argument("--source",      type=str, required=True,
                   help="VisDrone image folder or video file.")
    p.add_argument("--imgsz",       type=int,   default=1280)
    p.add_argument("--track-thresh",type=float, default=0.25)
    p.add_argument("--low-thresh",  type=float, default=0.10)
    p.add_argument("--match-thresh",type=float, default=0.80,
                   help="Cost threshold (1-IoU). Default 0.80 = IoU >= 0.20.")
    p.add_argument("--max-lost",    type=int,   default=60,
                   help="Frames before lost track removed (~2s at 30fps).")
    p.add_argument("--trail-len",   type=int,   default=40)
    p.add_argument("--cmc",    dest="cmc", action="store_true",  default=True)
    p.add_argument("--no-cmc", dest="cmc", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
