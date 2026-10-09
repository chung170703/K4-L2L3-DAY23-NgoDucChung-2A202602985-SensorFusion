"""Bonus: export the student tracker's tracks for CVAT and inspect them visually.

Writes, for a short frame window of the graded segment (fused mode, seed 0):

* ``cvat_tracks.json``  - written with ``fusion_lab.export_cvat.export_tracks_json``
  (per-frame track states in the vehicle frame plus the projected FRONT-image box);
* ``cvat_annotations.xml`` - the same tracks as "CVAT for video 1.1" rectangle
  tracks (one CVAT track per tracker id, attribute ``track_id`` / ``state``);
* ``.cache/cvat_images/`` - the matching FRONT frames (not committed: Waymo data).
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from fusion_lab.export_cvat import export_tracks_json
from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace
from replay import load_cache, replay

OUT = Path(__file__).parent
WIDTH, HEIGHT = 1920, 1280


def box_corners(track) -> np.ndarray:
    x, y, z = track["x"][:3]
    half_l, half_w, half_h = track["length"] / 2, track["width"] / 2, track["height"] / 2
    c, s = np.cos(track["yaw"]), np.sin(track["yaw"])
    rot = np.array([[c, -s], [s, c]])
    corners = []
    for dl in (-half_l, half_l):
        for dw in (-half_w, half_w):
            xy = rot @ np.array([dl, dw])
            for dz in (-half_h, half_h):
                corners.append([x + xy[0], y + xy[1], z + dz])
    return np.array(corners)


def project_box(camera: Sensor, track):
    """Axis-aligned image rectangle of the projected 3D box, or None if not visible."""
    pixels = []
    for corner in box_corners(track):
        state = np.zeros((6, 1))
        state[:3, 0] = corner
        if not camera.in_fov(np.asmatrix(state)):
            return None
        pixels.append(np.asarray(camera.get_hx(np.asmatrix(state))).ravel())
    pixels = np.array(pixels)
    xtl, ytl = np.clip(pixels.min(axis=0), 0, [WIDTH, HEIGHT])
    xbr, ybr = np.clip(pixels.max(axis=0), 0, [WIDTH, HEIGHT])
    if xbr - xtl < 4 or ybr - ytl < 4:
        return None
    return [float(xtl), float(ytl), float(xbr), float(ybr)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=40)
    parser.add_argument("--end", type=int, default=70)
    args = parser.parse_args()

    cache = load_cache()
    camera = Sensor("camera", cache["camera_calib"], load_workspace()["camera_fusion"])
    fused = replay(cache, "fused", trace=True)
    frames_json, per_track = [], {}
    image_dir = OUT / ".cache" / "cvat_images"
    image_dir.mkdir(parents=True, exist_ok=True)

    for item in fused["trace"]:
        n = item["frame"]
        if not args.start <= n <= args.end:
            continue
        local = n - args.start
        cached = next(f for f in cache["frames"] if f["frame"] == n)
        (image_dir / f"frame_{local:04d}.jpg").write_bytes(cached["front_jpeg"])
        tracks = []
        for t in item["after_camera"]:
            box = project_box(camera, t)
            tracks.append({
                "track_id": t["id"], "state": t["state"], "score": round(float(t["score"]), 4),
                "position_vehicle_m": [round(float(v), 3) for v in t["x"][:3]],
                "velocity_vehicle_mps": [round(float(v), 3) for v in t["x"][3:]],
                "size_hwl_m": [round(float(v), 3) for v in (t["height"], t["width"], t["length"])],
                "yaw_rad": round(float(t["yaw"]), 4), "front_box_xyxy_px": box,
            })
            if box is not None:
                per_track.setdefault(t["id"], []).append((local, t["state"], box))
        frames_json.append({
            "frame": n, "cvat_frame_index": local, "tracks": tracks,
            "gt_vehicle_ids": [g[0] for g in item["gt"]],
        })
    export_tracks_json(frames_json, OUT / "cvat_tracks.json")

    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"
    meta = ET.SubElement(ET.SubElement(root, "meta"), "task")
    ET.SubElement(meta, "mode").text = "interpolation"
    ET.SubElement(meta, "size").text = str(args.end - args.start + 1)
    labels = ET.SubElement(meta, "labels")
    label = ET.SubElement(labels, "label")
    ET.SubElement(label, "name").text = "vehicle_track"
    ET.SubElement(label, "type").text = "rectangle"
    attrs = ET.SubElement(label, "attributes")
    for name in ("track_id", "state"):
        attr = ET.SubElement(attrs, "attribute")
        ET.SubElement(attr, "name").text = name
        ET.SubElement(attr, "mutable").text = "True"
        ET.SubElement(attr, "input_type").text = "text"
        ET.SubElement(attr, "default_value").text = ""
        ET.SubElement(attr, "values").text = ""
    size = ET.SubElement(meta, "original_size")
    ET.SubElement(size, "width").text = str(WIDTH)
    ET.SubElement(size, "height").text = str(HEIGHT)
    for index, (tid, boxes) in enumerate(sorted(per_track.items())):
        track = ET.SubElement(root, "track", id=str(index), label="vehicle_track", source="manual")
        for local, state, (xtl, ytl, xbr, ybr) in boxes:
            box = ET.SubElement(track, "box", frame=str(local), outside="0", occluded="0",
                                keyframe="1", xtl=f"{xtl:.2f}", ytl=f"{ytl:.2f}",
                                xbr=f"{xbr:.2f}", ybr=f"{ybr:.2f}", z_order="0")
            ET.SubElement(box, "attribute", name="track_id").text = str(tid)
            ET.SubElement(box, "attribute", name="state").text = state
        last = boxes[-1][0]
        if last < args.end - args.start:
            xtl, ytl, xbr, ybr = boxes[-1][2]
            ET.SubElement(track, "box", frame=str(last + 1), outside="1", occluded="0", keyframe="1",
                          xtl=f"{xtl:.2f}", ytl=f"{ytl:.2f}", xbr=f"{xbr:.2f}", ybr=f"{ybr:.2f}", z_order="0")
    ET.indent(root)
    ET.ElementTree(root).write(OUT / "cvat_annotations.xml", encoding="utf-8", xml_declaration=True)
    for tid, boxes in sorted(per_track.items()):
        print("track", tid, "frames", boxes[0][0] + args.start, "-", boxes[-1][0] + args.start,
              sorted({s for _, s, _ in boxes}))


if __name__ == "__main__":
    main()
