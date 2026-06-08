"""src/tracker/ecc_compensator.py — OpenCV ECC camera motion compensation."""
import cv2
import numpy as np


class ECCCompensator:
    """
    Estimates inter-frame drone ego-motion using OpenCV ECC and applies
    the resulting affine warp to ByteTrack's Kalman-predicted positions.

    WHY ECC ON A DOWNSCALED FRAME?
    Running findTransformECC on a 1344x756 frame takes ~80ms — a pipeline
    killer. Downscaling to 320x180 reduces it to ~3ms while retaining
    sufficient texture for a reliable warp estimate. Translation components
    are then rescaled back to full-frame coordinates.

    MOTION MODEL: MOTION_EUCLIDEAN (rotation + translation, no shear/scale).
    Appropriate for a stabilised gimbal camera on a drone.
    """

    # ECC termination: stop after 10 iterations OR if improvement < 0.01
    ECC_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01)

    def __init__(self, downscale_wh: tuple = (320, 180)):
        self.dw, self.dh = downscale_wh
        self.prev_gray: np.ndarray | None = None
        # Identity warp (no motion) used as warm-start and fallback
        self._identity = np.eye(2, 3, dtype=np.float32)

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a new frame. Returns a 2x3 affine warp matrix M in
        FULL-FRAME pixel coordinates.

        If ECC fails (insufficient texture / first frame), returns identity.
        """
        # Downscale and convert to grayscale for fast ECC
        small = cv2.resize(frame, (self.dw, self.dh))
        curr_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        warp = self._identity.copy()

        if self.prev_gray is not None:
            try:
                # Estimate motion at downscale resolution
                _, warp_small = cv2.findTransformECC(
                    self.prev_gray, curr_gray,
                    self._identity.copy(),
                    cv2.MOTION_EUCLIDEAN,
                    self.ECC_CRITERIA,
                )
                # Scale translation from downscale → full-frame coords
                # Rotation components (top-left 2x2) stay unchanged
                fw = frame.shape[1]
                fh = frame.shape[0]
                warp = warp_small.copy()
                warp[0, 2] *= fw / self.dw
                warp[1, 2] *= fh / self.dh

            except cv2.error:
                # ECC diverged (e.g., flat sky, rapid motion blur) — use identity
                warp = self._identity.copy()

        self.prev_gray = curr_gray
        return warp
