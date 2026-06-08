"""src/tracker/kalman_filter.py — Constant-velocity Kalman filter for ByteTrack."""
import numpy as np


class KalmanBoxFilter:
    """
    Kalman filter tracking bounding boxes in [cx, cy, w, h] space.

    State vector (8D): [cx, cy, w, h, vcx, vcy, vw, vh]
    Measurement (4D):  [cx, cy, w, h]

    Constant-velocity model: position += velocity each step.
    """

    def __init__(self):
        dt = 1.0  # one time step per frame

        # Transition matrix F: position += velocity
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Measurement matrix H: observe [cx, cy, w, h] only
        self.H = np.eye(4, 8, dtype=np.float32)

        # Process noise Q (higher uncertainty on velocities)
        self.Q = np.diag([
            1.0, 1.0, 1.0, 1.0,   # position noise
            0.01, 0.01, 0.0001, 0.0001  # velocity noise (small)
        ]).astype(np.float32)

        # Measurement noise R (observation uncertainty)
        self.R = np.diag([1.0, 1.0, 10.0, 10.0]).astype(np.float32)

        # State covariance P
        self.P = np.diag([10.0, 10.0, 10.0, 10.0,
                          10000.0, 10000.0, 10000.0, 10000.0]).astype(np.float32)

        self.mean = np.zeros(8, dtype=np.float32)

    def initiate(self, bbox: np.ndarray) -> None:
        """Initialise state from first detection [cx, cy, w, h]."""
        self.mean[:4] = bbox.astype(np.float32)
        self.mean[4:] = 0.0
        # Scale initial covariance by bbox size
        std = max(bbox[2], bbox[3])
        self.P = np.diag([2*std, 2*std, 2*std, 2*std,
                          10*std, 10*std, 10*std, 10*std]).astype(np.float32)

    def predict(self) -> np.ndarray:
        """Kalman predict step. Returns predicted [cx, cy, w, h]."""
        self.mean = self.F @ self.mean
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Clamp width/height to positive
        self.mean[2] = max(self.mean[2], 1.0)
        self.mean[3] = max(self.mean[3], 1.0)
        return self.mean[:4].copy()

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """Kalman update step with new detection [cx, cy, w, h]."""
        z = measurement.astype(np.float32)
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)         # Kalman gain
        y = z - self.H @ self.mean                        # innovation
        self.mean = self.mean + K @ y
        self.P = (np.eye(8, dtype=np.float32) - K @ self.H) @ self.P
        return self.mean[:4].copy()

    @property
    def bbox(self) -> np.ndarray:
        """Return current estimated [cx, cy, w, h]."""
        return self.mean[:4].copy()

    @property
    def tlbr(self) -> np.ndarray:
        """Return [x1, y1, x2, y2] for IoU computation."""
        cx, cy, w, h = self.mean[:4]
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2], dtype=np.float32)
