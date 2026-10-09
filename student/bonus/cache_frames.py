"""Cache per-frame detector output, ground truth and FRONT camera data (bonus helper).

The LiDAR detector is the slow part of ``fusion-run-lab`` and does not depend on
tracking or camera settings, so the bonus experiments run it once and replay the
tracker on the cached detections. The platform and Part A-D code are untouched;
the observation building mirrors ``fusion_lab.scripts.run_lab`` exactly.

Usage (from the repo root)::

    python student/bonus/cache_frames.py --config student/config/paths.yaml
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fusion_lab.scripts import run_lab

DEFAULT_OUT = Path(__file__).parent / ".cache" / "frames.pkl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    run_lab._setup_import_paths()
    from simple_waymo_open_dataset_reader import WaymoDataFileReader, dataset_pb2, label_pb2
    from simple_waymo_open_dataset_reader import utils as waymo_utils

    from fusion_lab.lidar_pcl import pcl_from_range_image
    from fusion_lab.evaluation import valid_ground_truth
    from fusion_lab.workspace_loader import load_workspace

    cfg = run_lab._load_paths_config(args.config)
    ws = load_workspace()
    weights = run_lab._resolve_weights(cfg)
    det_cfg = ws["detection_pipeline"].load_fpn_resnet_config(str(weights))
    model = ws["detection_pipeline"].create_fpn_model(det_cfg, str(weights))
    tfrecord = Path(cfg["waymo_dir"]) / cfg["segment"]
    start, end = int(cfg["frame_start"]), int(cfg["frame_end"])
    rng = np.random.default_rng(args.seed)

    frames, calib = [], None
    for cnt, frame in enumerate(WaymoDataFileReader(str(tfrecord))):
        if cnt < start:
            continue
        if cnt > end:
            break
        if calib is None:
            cam = waymo_utils.get(frame.context.camera_calibrations, dataset_pb2.CameraName.FRONT)
            calib = SimpleNamespace(
                intrinsic=list(cam.intrinsic), width=cam.width, height=cam.height,
                extrinsic=SimpleNamespace(transform=list(cam.extrinsic.transform)),
            )
        points = pcl_from_range_image(frame, dataset_pb2.LaserName.TOP)
        tensor = torch.from_numpy(ws["bev_mapping"].bev_maps_from_pcl(points, det_cfg)).unsqueeze(0).float()
        detections = ws["detection_pipeline"].detect_objects_from_bev(tensor, model, det_cfg)
        labels = valid_ground_truth(frame.laser_labels, det_cfg, label_pb2.Label.Type.TYPE_VEHICLE)
        group = next((g for g in frame.camera_labels if g.name == dataset_pb2.CameraName.FRONT), None)
        pixels = None
        if group is not None:  # same draw order as run_lab._front_observations
            pixels = []
            for label in group.labels:
                if label.type == label_pb2.Label.Type.TYPE_VEHICLE:
                    centre = np.array([label.box.center_x, label.box.center_y])
                    pixels.append(centre + rng.normal(0, 0.5, 2))
        image = next((i.image for i in frame.images if i.name == dataset_pb2.CameraName.FRONT), None)
        frames.append({
            "frame": cnt,
            "detections": [np.asarray(d, dtype=float) for d in detections],
            "gt": [(l.id, l.box.center_x, l.box.center_y, l.box.center_z) for l in labels],
            "camera_pixels": pixels,
            "front_jpeg": image,
        })
        print("cached frame", cnt, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("wb") as f:
        pickle.dump({"frames": frames, "camera_calib": calib, "seed": args.seed,
                     "segment": tfrecord.name}, f)
    print("saved", args.out, len(frames), "frames")


if __name__ == "__main__":
    main()
