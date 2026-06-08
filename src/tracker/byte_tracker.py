"""src/tracker/byte_tracker.py — ByteTrack MOT (v2: fixed cost threshold semantics).

BUG FIX — ID explosion root cause:
  v1 used match_thresh as an IoU threshold:
      if iou[r, c] >= match_thresh  →  requires 80% overlap
  This is wrong. ByteTrack uses match_thresh as a COST threshold:
      cost = 1 - IoU
      accept match if cost <= match_thresh  →  IoU >= 1 - 0.8 = 0.2

  At drone altitude, a person occupies 12-30px. Even a 1-2px position
  shift between frames drops IoU below 0.80, so v1 rejected almost every
  match and spawned a new track each frame — causing the ID explosion.

  With cost threshold 0.8 (minimum IoU = 0.20), legitimate matches are
  accepted and ID count stays stable.
"""
import numpy as np
from enum import IntEnum
from collections import deque
from scipy.optimize import linear_sum_assignment
from .kalman_filter import KalmanBoxFilter


class TrackState(IntEnum):
    NEW     = 0
    TRACKED = 1
    LOST    = 2
    REMOVED = 3


def _iou_matrix(tlbr_a: np.ndarray, tlbr_b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between [x1,y1,x2,y2] box sets. Returns N×M matrix."""
    if len(tlbr_a) == 0 or len(tlbr_b) == 0:
        return np.zeros((len(tlbr_a), len(tlbr_b)), dtype=np.float32)
    ix1 = np.maximum(tlbr_a[:, None, 0], tlbr_b[None, :, 0])
    iy1 = np.maximum(tlbr_a[:, None, 1], tlbr_b[None, :, 1])
    ix2 = np.minimum(tlbr_a[:, None, 2], tlbr_b[None, :, 2])
    iy2 = np.minimum(tlbr_a[:, None, 3], tlbr_b[None, :, 3])
    inter = np.maximum(ix2 - ix1, 0) * np.maximum(iy2 - iy1, 0)
    area_a = (tlbr_a[:, 2] - tlbr_a[:, 0]) * (tlbr_a[:, 3] - tlbr_a[:, 1])
    area_b = (tlbr_b[:, 2] - tlbr_b[:, 0]) * (tlbr_b[:, 3] - tlbr_b[:, 1])
    union  = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def _cxywh_to_tlbr(bbox) -> np.ndarray:
    cx, cy, w, h = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)


def _associate(tracks: list, detections: list, cost_thresh: float):
    """
    Hungarian assignment between tracks and detections.

    Cost matrix = 1 - IoU.
    A match is accepted if cost <= cost_thresh  (i.e., IoU >= 1 - cost_thresh).

    With cost_thresh=0.8 → minimum IoU = 0.20  (correct ByteTrack default)
    With cost_thresh=0.5 → minimum IoU = 0.50  (stricter, for stage-2)

    Returns (matched pairs, unmatched track indices, unmatched det indices).
    """
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    t_boxes = np.array([t.kalman.tlbr for t in tracks],  dtype=np.float32)
    d_boxes = np.array([_cxywh_to_tlbr(d[:4]) for d in detections], dtype=np.float32)

    iou  = _iou_matrix(t_boxes, d_boxes)
    cost = 1.0 - iou  # cost matrix: low = good match

    row_ind, col_ind = linear_sum_assignment(cost)

    matched, matched_t, matched_d = [], set(), set()
    for r, c in zip(row_ind, col_ind):
        # FIX: use cost threshold (not IoU threshold)
        if cost[r, c] <= cost_thresh:
            matched.append((r, c))
            matched_t.add(r)
            matched_d.add(c)

    u_tracks = [i for i in range(len(tracks))     if i not in matched_t]
    u_dets   = [i for i in range(len(detections)) if i not in matched_d]
    return matched, u_tracks, u_dets


class STrack:
    """Single object track with Kalman state and trajectory history."""
    _id_counter: int = 0

    def __init__(self, det: np.ndarray):
        """det: [cx, cy, w, h, score]"""
        self.kalman = KalmanBoxFilter()
        self.kalman.initiate(det[:4].astype(np.float32))
        self.score           = float(det[4])
        self.state           = TrackState.NEW
        self.track_id: int   = -1
        self.age: int        = 1
        self.hits: int       = 1
        self.time_since_update: int = 0
        self.history: deque  = deque(maxlen=50)
        self.history.append(tuple(det[:2].astype(int)))

    @classmethod
    def _next_id(cls) -> int:
        cls._id_counter += 1
        return cls._id_counter

    def predict(self) -> None:
        self.kalman.predict()
        self.age += 1
        self.time_since_update += 1

    def update(self, det: np.ndarray) -> None:
        self.kalman.update(det[:4].astype(np.float32))
        self.score = float(det[4])
        self.hits += 1
        self.time_since_update = 0
        cx, cy = self.kalman.bbox[:2]
        self.history.append((int(cx), int(cy)))
        if self.state == TrackState.NEW:
            self.state    = TrackState.TRACKED
            self.track_id = STrack._next_id()

    def re_activate(self, det: np.ndarray) -> None:
        """Re-activate a lost track — KEEP the original track_id."""
        self.kalman.update(det[:4].astype(np.float32))
        self.score = float(det[4])
        self.hits += 1
        self.time_since_update = 0
        self.state = TrackState.TRACKED   # restore without new ID
        cx, cy = self.kalman.bbox[:2]
        self.history.append((int(cx), int(cy)))

    def mark_lost(self) -> None:
        self.state = TrackState.LOST

    def apply_cmc(self, warp: np.ndarray) -> None:
        """Apply ECC affine warp to correct Kalman position+velocity for ego-motion."""
        m = self.kalman.mean
        cx, cy   = float(m[0]), float(m[1])
        m[0] = warp[0, 0] * cx + warp[0, 1] * cy + warp[0, 2]
        m[1] = warp[1, 0] * cx + warp[1, 1] * cy + warp[1, 2]
        vx, vy   = float(m[4]), float(m[5])
        m[4] = warp[0, 0] * vx + warp[0, 1] * vy   # rotation only, no translation
        m[5] = warp[1, 0] * vx + warp[1, 1] * vy

    @property
    def is_confirmed(self) -> bool:
        return self.state == TrackState.TRACKED and self.track_id >= 0


class ByteTracker:
    """
    ByteTrack Multi-Object Tracker.

    Key parameters for drone footage
    ----------------------------------
    track_thresh  = 0.25  : High-conf bucket. Lowered from default 0.5 to
                            catch faint altitude detections (conf ~0.15–0.30).
    low_thresh    = 0.10  : Low-conf bucket floor.
    match_thresh  = 0.80  : COST threshold (= 1 - IoU). Accepts a match when
                            IoU >= 1 - 0.80 = 0.20. This is the standard
                            ByteTrack default. DO NOT interpret as IoU threshold.
    max_time_lost = 60    : 2s at 30fps. Longer window allows re-activation
                            after occlusion by trees/buildings in drone footage.

    3-Stage association per frame
    ------------------------------
    Stage 1: high-conf dets ↔ tracked tracks  (cost_thresh=match_thresh)
    Stage 2: low-conf  dets ↔ unmatched tracked tracks  (cost_thresh=0.5)
    Stage 3: unmatched high-conf dets ↔ lost tracks  (cost_thresh=match_thresh)
    """

    def __init__(
        self,
        track_thresh:  float = 0.25,
        low_thresh:    float = 0.10,
        match_thresh:  float = 0.80,   # COST threshold: IoU >= 1 - 0.80 = 0.20
        max_time_lost: int   = 60,
    ):
        self.track_thresh  = track_thresh
        self.low_thresh    = low_thresh
        self.match_thresh  = match_thresh
        self.max_time_lost = max_time_lost

        self.tracked_stracks: list = []
        self.lost_stracks:    list = []
        self.frame_id: int = 0

    def update(self, detections: np.ndarray, warp: np.ndarray | None = None) -> list:
        """
        Update tracker for one frame.

        Parameters
        ----------
        detections : np.ndarray  shape (N, 5)  [cx, cy, w, h, score]
        warp       : 2×3 affine warp from ECC CMC (None = skip)

        Returns list of confirmed STrack objects.
        """
        self.frame_id += 1

        if len(detections) == 0:
            detections = np.zeros((0, 5), dtype=np.float32)
        else:
            detections = np.asarray(detections, dtype=np.float32)

        # Split by confidence
        high_mask = detections[:, 4] >= self.track_thresh
        low_mask  = (detections[:, 4] >= self.low_thresh) & ~high_mask
        high_dets = detections[high_mask]
        low_dets  = detections[low_mask]

        # --- Predict + CMC ---
        for t in self.tracked_stracks + self.lost_stracks:
            t.predict()
            if warp is not None:
                t.apply_cmc(warp)

        # --- Stage 1: high-conf ↔ tracked ---
        m1, u_t1, u_d_high = _associate(
            self.tracked_stracks, list(high_dets), self.match_thresh)

        matched_t1_set = {ti for ti, _ in m1}
        for ti, di in m1:
            self.tracked_stracks[ti].update(high_dets[di])

        # --- Stage 2: low-conf ↔ unmatched tracked ---
        # NOTE: _associate(tracks, detections, thresh) — tracks first!
        remaining_tracked = [self.tracked_stracks[i] for i in u_t1]
        m2, u_t2, _ = _associate(remaining_tracked, list(low_dets), 0.5)
        for ti, di in m2:
            remaining_tracked[ti].update(low_dets[di])

        # Mark unmatched active tracks as lost
        newly_lost = []
        for i in u_t2:
            remaining_tracked[i].mark_lost()
            newly_lost.append(remaining_tracked[i])

        # --- Stage 3: unmatched high-conf ↔ lost tracks (re-activation) ---
        unmatched_high = high_dets[u_d_high]
        m3, u_lost3, u_d_new = _associate(
            self.lost_stracks, list(unmatched_high), self.match_thresh)

        recovered_set = {ti for ti, _ in m3}
        for ti, di in m3:
            self.lost_stracks[ti].re_activate(unmatched_high[di])

        # --- Init new tracks ---
        new_stracks = []
        for di in u_d_new:
            det = unmatched_high[di]
            if det[4] >= self.track_thresh:
                t = STrack(det)
                t.state    = TrackState.TRACKED
                t.track_id = STrack._next_id()
                new_stracks.append(t)

        # --- Rebuild pools ---
        # Tracked: matched-in-stage1 + matched-in-stage2 + recovered-lost + new
        stage1_matched = [self.tracked_stracks[ti] for ti, _ in m1]
        stage2_matched = [remaining_tracked[ti]    for ti, _ in m2]
        recovered      = [self.lost_stracks[ti]    for ti, _ in m3]

        self.tracked_stracks = (
            stage1_matched + stage2_matched + recovered + new_stracks
        )

        # Lost: newly lost + still-within-time-limit lost (not recovered)
        surviving_lost = [
            t for i, t in enumerate(self.lost_stracks)
            if i not in recovered_set
            and t.time_since_update <= self.max_time_lost
        ]
        self.lost_stracks = newly_lost + surviving_lost

        return [t for t in self.tracked_stracks if t.is_confirmed]
