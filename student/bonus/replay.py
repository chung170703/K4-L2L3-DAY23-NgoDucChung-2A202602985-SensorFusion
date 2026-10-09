"""Replay the student tracker on cached detections (bonus experiments).

Mirrors the per-frame order of ``fusion_lab.scripts.run_lab.run``: predict once,
LiDAR update, then the optional camera update. The only addition is an optional
extrinsic perturbation of the *tracker's* camera model, which emulates calibration
drift while the simulated pixels still come from the true GT boxes.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from fusion_lab import tracking_params
from fusion_lab.evaluation import tracking_counts
from fusion_lab.scripts import run_lab
from fusion_lab.tracking.filter import Filter
from fusion_lab.tracking.manager import TrackManager
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace

CACHE = Path(__file__).parent / ".cache" / "frames.pkl"


def load_cache(path: Path = CACHE) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def rotation(axis: str, degrees: float) -> np.ndarray:
    """4x4 rotation about a camera-frame axis (x forward, y left, z up)."""
    a = np.deg2rad(degrees)
    c, s = np.cos(a), np.sin(a)
    r = {"x": [[1, 0, 0], [0, c, -s], [0, s, c]],
         "y": [[c, 0, s], [0, 1, 0], [-s, 0, c]],
         "z": [[c, -s, 0], [s, c, 0], [0, 0, 1]]}[axis]
    out = np.eye(4)
    out[:3, :3] = r
    return out


def perturb_camera(sensor: Sensor, axis: str, degrees: float) -> None:
    """Rotate the tracker's camera extrinsic about ``axis`` by ``degrees``."""
    if degrees == 0:
        return
    transform = np.asarray(sensor.sens_to_veh) @ rotation(axis, degrees)
    sensor.sens_to_veh = np.asmatrix(transform)
    sensor.veh_to_sens = np.asmatrix(np.linalg.inv(transform))


class CountingFilter(Filter):
    """Filter that counts camera updates and keeps their pre-update innovation."""

    def __init__(self, kalman_mod):
        super().__init__(kalman_mod)
        self.camera_updates = 0
        self.camera_gammas: list[np.ndarray] = []

    def update(self, track, meas):
        if meas.sensor.name == "camera":
            self.camera_updates += 1
            self.camera_gammas.append(np.asarray(self._k.innovation(track.x, meas)).ravel())
        super().update(track, meas)


def _snapshot(manager: TrackManager) -> list[dict[str, Any]]:
    return [{"id": t.id, "state": t.state, "score": t.score,
             "x": np.asarray(t.x).ravel().copy(), "P": np.asarray(t.P).copy(),
             "height": t.height, "width": t.width, "length": t.length, "yaw": t.yaw}
            for t in manager.track_list]


def replay(cache: dict[str, Any], mode: str, axis: str = "z", degrees: float = 0.0,
           trace: bool = False, probe_distances: bool = False) -> dict[str, Any]:
    """Run the tracker over cached frames.

    Returns tracking totals, the frame records, camera filter statistics and, when
    requested, a per-frame trace / Mahalanobis probe of every camera pair.
    """
    ws = load_workspace()
    assoc, cam = ws["association"], ws["camera_fusion"]
    det_cfg = ws["detection_pipeline"].load_fpn_resnet_config(None)
    kalman = ws["kalman"]

    lidar = Sensor("lidar", SimpleNamespace(), cam)
    camera = None
    if mode == "fused":
        camera = Sensor("camera", cache["camera_calib"], cam)
        perturb_camera(camera, axis, degrees)

    probe: list[dict[str, Any]] = []
    current: dict[str, Any] = {}
    original = assoc.mahalanobis_distance
    if probe_distances:
        def wrapped(track, meas):
            d2 = original(track, meas)
            if meas.sensor.name == "camera":
                probe.append({"frame": current["frame"], "meas": id(meas), "track": track.id,
                              "state": track.state, "d2": d2,
                              "z": tuple(np.asarray(meas.z).ravel().round(6)),
                              "gamma": np.asarray(kalman.innovation(track.x, meas)).ravel()})
            return d2

        assoc.mahalanobis_distance = wrapped

    kf = CountingFilter(kalman)
    manager = TrackManager(ws["track_management"])
    records, traces = [], []
    try:
        for item in cache["frames"]:
            n = item["frame"]
            current["frame"] = n
            observations = run_lab._lidar_observations(n, item["detections"], lidar, det_cfg)
            for track in manager.track_list:
                kf.predict(track)
                track.set_t(n * tracking_params.dt)
            after_predict = _snapshot(manager) if trace else None
            assoc.associate_and_update(manager, observations, kf, lidar)
            after_lidar = _snapshot(manager) if trace else None
            camera_meas = []
            if camera is not None and item["camera_pixels"] is not None:
                for pixel in item["camera_pixels"]:
                    camera.generate_measurement(n, pixel, camera_meas)
                assoc.associate_and_update(manager, camera_meas, kf, camera)
            labels = [SimpleNamespace(box=SimpleNamespace(center_x=x, center_y=y, center_z=z))
                      for _, x, y, z in item["gt"]]
            records.append({"mode": mode, "frame": n, "valid_gt": len(labels),
                            **tracking_counts(manager.track_list, labels)})
            if trace:
                traces.append({"frame": n, "after_predict": after_predict,
                               "after_lidar": after_lidar, "after_camera": _snapshot(manager),
                               "camera_pixels": item["camera_pixels"], "gt": item["gt"],
                               "detections": item["detections"]})
    finally:
        assoc.mahalanobis_distance = original

    total = {k: sum(r[k] for r in records)
             for k in ("matches", "sum_sq_err", "ghosts", "misses", "confirmed")}
    rmse = float(np.sqrt(total["sum_sq_err"] / total["matches"])) if total["matches"] else None
    gammas = np.array(kf.camera_gammas).reshape(-1, 2)
    return {"mode": mode, "axis": axis, "degrees": degrees, "rmse": rmse, **total,
            "mean_confirmed_tracks": total["confirmed"] / len(records),
            "camera_updates": kf.camera_updates, "camera_gammas": gammas,
            "records": records, "trace": traces, "probe": probe}
