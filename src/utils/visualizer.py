"""src/utils/visualizer.py — Fast bounding box, ID label, and trajectory tail renderer.

PERFORMANCE FIX (v2):
The v1 renderer called cv2.addWeighted once per trail segment per track.
With 28 tracks × 40 trail points = 1,120 full-frame blends per frame at
~12ms each → ~13 seconds per frame (0.07 FPS). Completely unusable.

v2 draws ALL trail segments onto ONE pre-allocated black overlay image,
then blends it onto the output ONCE. This reduces blending cost from
O(tracks × trail_length) → O(1), recovering ~13+ seconds per frame.

Trail fade effect: achieved by scaling BGR channel values by (i/n)
so older segments are darker, newer are full brightness. No per-segment
alpha blend needed — color scaling achieves the same visual result.
"""
import cv2
import numpy as np
from collections import deque
from typing import Dict, List, Tuple


def id_to_bgr(track_id: int) -> Tuple[int, int, int]:
    """
    Unique high-contrast BGR color per track ID.
    Golden-angle HSV spacing, OpenCV hue range [0, 179].
    """
    hue = int((track_id * 137.508) % 180)
    hsv = np.array([[[hue, 220, 220]]], dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


class TrackVisualizer:
    """
    Renders active tracks with bounding boxes, ID labels,
    fading trajectory tails, and a HUD overlay.

    Performance guarantee: exactly ONE cv2.addWeighted call per frame
    regardless of track count or trail length.
    """

    def __init__(self, trail_len: int = 40):
        self.trails: Dict[int, deque] = {}
        self.trail_len = trail_len
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}

    def _get_color(self, track_id: int) -> Tuple[int, int, int]:
        if track_id not in self._color_cache:
            self._color_cache[track_id] = id_to_bgr(track_id)
        return self._color_cache[track_id]

    def draw(
        self,
        frame: np.ndarray,
        tracks: list,
        fps: float,
        frame_id: int,
    ) -> np.ndarray:
        """
        Draw all visual elements and return annotated frame.

        Trail rendering strategy (O(1) blends):
          1. Allocate one black overlay (same shape as frame).
          2. Draw ALL trail segments for ALL tracks onto the overlay.
             Fade = scale BGR values by (i/n): old segments are dark,
             new segments are full-brightness.
          3. cv2.add() the overlay onto the frame — single pass,
             black pixels (0,0,0) add nothing to the background.
        """
        h, w = frame.shape[:2]

        # --- Step 1: Draw ALL trails on a single black overlay ---
        trail_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        active_ids = []

        for track in tracks:
            tid  = track.track_id
            cx, cy, tw, th = track.kalman.bbox
            active_ids.append(tid)

            # Update trail history
            if tid not in self.trails:
                self.trails[tid] = deque(maxlen=self.trail_len)
            self.trails[tid].append((int(cx), int(cy)))

            pts = list(self.trails[tid])
            n   = len(pts)
            if n < 2:
                continue

            color = self._get_color(tid)
            b, g, r = color

            # Draw each segment with brightness scaled by age ratio
            for i in range(1, n):
                alpha_i = i / n          # 0.0 = oldest, 1.0 = newest
                thickness = max(1, int(3 * alpha_i))
                # Scale color intensity for fade: old→dark, new→bright
                faded = (
                    int(b * alpha_i),
                    int(g * alpha_i),
                    int(r * alpha_i),
                )
                cv2.line(trail_overlay, pts[i - 1], pts[i],
                         faded, thickness, cv2.LINE_AA)

        # --- Step 2: Single additive blend — the ONLY blend per frame ---
        # cv2.add: result = saturate(frame + trail_overlay)
        # Black pixels (0,0,0) add nothing; trail pixels light up.
        out = cv2.add(frame, trail_overlay)

        # --- Step 3: Remove stale trails for gone tracks ---
        for dead_id in set(self.trails) - set(active_ids):
            del self.trails[dead_id]

        # --- Step 4: Bounding boxes + ID labels (no blending) ---
        for track in tracks:
            tid      = track.track_id
            cx, cy, tw, th = track.kalman.bbox
            x1 = int(cx - tw / 2)
            y1 = int(cy - th / 2)
            x2 = int(cx + tw / 2)
            y2 = int(cy + th / 2)
            color = self._get_color(tid)

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # ID label with filled background
            label = f"ID:{tid}"
            (tw_l, th_l), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
            lx, ly = x1, max(0, y1 - th_l - 5)
            cv2.rectangle(out, (lx, ly), (lx + tw_l + 4, ly + th_l + 5),
                          color, -1)
            cv2.putText(out, label, (lx + 2, ly + th_l + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (255, 255, 255), 1, cv2.LINE_AA)

        # --- Step 5: HUD bar ---
        self._draw_hud(out, fps, frame_id, len(tracks))
        return out

    def _draw_hud(
        self, frame: np.ndarray, fps: float, frame_id: int, n_tracks: int
    ) -> None:
        """Semi-transparent top bar. Single addWeighted call (small ROI)."""
        bar_h = 28
        roi = frame[:bar_h, :]
        dark = np.zeros_like(roi)
        cv2.addWeighted(dark, 0.55, roi, 0.45, 0, roi)
        frame[:bar_h, :] = roi

        fps_color = (
            (0, 210, 0)   if fps >= 15.0 else
            (0, 190, 210) if fps >= 8.0  else
            (0, 0, 210)
        )
        cv2.putText(frame, f"FPS: {fps:.1f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, fps_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Frame: {frame_id:05d}", (125, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
        h, w = frame.shape[:2]
        cv2.putText(frame, f"Tracks: {n_tracks}", (w - 130, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1, cv2.LINE_AA)
