from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


EYES = ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380))
POSE_POINTS = (1, 152, 33, 263, 61, 291)
POSE_MODEL = np.array(
    [(0, 0, 0), (0, -330, -65), (-225, 170, -135), (225, 170, -135),
     (-150, -150, -125), (150, -150, -125)],
    dtype=np.float64,
)
PRIOR_NAMES = ("ear", "mar", "pitch", "yaw", "roll")


@dataclass
class FrameResult:
    crop: np.ndarray
    prior: np.ndarray
    detected: bool


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def _eye_ratio(points: np.ndarray, indices: tuple[int, ...]) -> float:
    a, b, c, d, e, f = points[list(indices)]
    width = 2 * _distance(a, d)
    return (_distance(b, f) + _distance(c, e)) / width if width > 0 else float("nan")


def _pose(points: np.ndarray, width: int, height: int) -> np.ndarray:
    focal = float(width)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    try:
        ok, rotation, _ = cv2.solvePnP(
            POSE_MODEL, points[list(POSE_POINTS)].astype(np.float64), camera,
            np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            matrix, _ = cv2.Rodrigues(rotation)
            return np.asarray(cv2.RQDecomp3x3(matrix)[0], dtype=np.float32)
    except cv2.error:
        pass
    return np.full(3, np.nan, dtype=np.float32)


def _center_crop(frame: np.ndarray, center: tuple[float, float], side: float, size: int) -> np.ndarray:
    side_pixels = max(2, int(round(side)))
    crop = cv2.getRectSubPix(frame, (side_pixels, side_pixels), center)
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


class FaceExtractor:
    def __init__(self, settings: dict):
        import mediapipe as mp

        self.settings = settings
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "Installed MediaPipe lacks the Face Mesh solutions API; "
                "install the version pinned in requirements-data.txt."
            )
        self.mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.detector = None
        if settings.get("face_detection_fallback", False):
            self.detector = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=0.3,
            )
        self.last_box: tuple[tuple[float, float], float] | None = None

    def close(self) -> None:
        self.mesh.close()
        if self.detector is not None:
            self.detector.close()

    def __enter__(self) -> "FaceExtractor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def extract(self, frame: np.ndarray) -> FrameResult:
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detected = self.mesh.process(rgb).multi_face_landmarks
        origin_x, origin_y = 0, 0
        region_width, region_height = width, height
        if not detected and self.detector is not None:
            faces = self.detector.process(rgb).detections
            if faces:
                box = faces[0].location_data.relative_bounding_box
                center_x = (box.xmin + box.width / 2) * width
                center_y = (box.ymin + box.height / 2) * height
                side = max(box.width * width, box.height * height) * 2.0
                x0 = max(0, int(center_x - side / 2))
                y0 = max(0, int(center_y - side / 2))
                x1 = min(width, int(center_x + side / 2))
                y1 = min(height, int(center_y + side / 2))
                if x1 > x0 and y1 > y0:
                    region = rgb[y0:y1, x0:x1]
                    detected = self.mesh.process(region).multi_face_landmarks
                    origin_x, origin_y = x0, y0
                    region_width, region_height = x1 - x0, y1 - y0
        if not detected:
            center, side = self.last_box or ((width / 2, height / 2), min(width, height))
            return FrameResult(
                _center_crop(frame, center, side, self.settings["face_size"]),
                np.full(len(PRIOR_NAMES), np.nan, dtype=np.float32), False,
            )

        points = np.array([(origin_x + p.x * region_width,
                            origin_y + p.y * region_height)
                           for p in detected[0].landmark], dtype=np.float32)
        eyes = [points[list(indices)].mean(axis=0) for indices in EYES]
        eye_center = (eyes[0] + eyes[1]) / 2
        angle = float(np.degrees(np.arctan2(eyes[1][1] - eyes[0][1], eyes[1][0] - eyes[0][0])))
        rotation = cv2.getRotationMatrix2D(tuple(eye_center), angle, 1.0)
        aligned = cv2.warpAffine(frame, rotation, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        rotated_points = cv2.transform(points[None, :, :], rotation)[0]
        lo = rotated_points.min(axis=0)
        hi = rotated_points.max(axis=0)
        center = (lo + hi) / 2
        side = float(max(hi - lo) * (1 + 2 * self.settings["face_margin"]))
        self.last_box = ((float(center[0]), float(center[1])), side)
        crop = _center_crop(aligned, self.last_box[0], side, self.settings["face_size"])
        ears = [_eye_ratio(points, indices) for indices in EYES]
        mouth_width = _distance(points[61], points[291])
        mar = _distance(points[13], points[14]) / mouth_width if mouth_width > 0 else float("nan")
        prior = np.array([float(np.nanmean(ears)), mar, *_pose(points, width, height)], dtype=np.float32)
        return FrameResult(crop, prior, True)


def _event_count(values: np.ndarray, threshold: float, above: bool) -> int:
    active = values > threshold if above else values < threshold
    active &= np.isfinite(values)
    return int(np.sum(active[1:] & ~active[:-1]))


def summarize_priors(priors: np.ndarray, valid: np.ndarray, settings: dict) -> dict[str, float]:
    result: dict[str, float] = {"detection_failure_rate": float(1 - valid.mean())}
    times = np.linspace(0, settings["window_seconds"], len(priors), endpoint=False)
    for column, name in enumerate(PRIOR_NAMES):
        values = priors[:, column].astype(np.float64)
        usable = np.isfinite(values) & valid
        result[f"{name}_valid_ratio"] = float(usable.mean())
        for statistic in ("mean", "std", "min", "max", "trend_per_s"):
            result[f"{name}_{statistic}"] = float("nan")
        if usable.any():
            selected = values[usable]
            result.update({
                f"{name}_mean": float(selected.mean()),
                f"{name}_std": float(selected.std()),
                f"{name}_min": float(selected.min()),
                f"{name}_max": float(selected.max()),
            })
            if usable.sum() > 1:
                result[f"{name}_trend_per_s"] = float(np.polyfit(times[usable], selected, 1)[0])
    ear = priors[:, 0]
    mar = priors[:, 1]
    closed = np.isfinite(ear) & valid & (ear < settings["ear_closed_threshold"])
    result["ecr_sampled"] = float(closed.sum() / valid.sum()) if valid.any() else float("nan")
    result["blink_events_sampled"] = float(_event_count(ear, settings["ear_closed_threshold"], False))
    result["yawn_events_sampled"] = float(_event_count(mar, settings["mar_yawn_threshold"], True))
    return result
