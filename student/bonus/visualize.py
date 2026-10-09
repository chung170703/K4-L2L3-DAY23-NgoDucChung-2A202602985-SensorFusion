"""Bonus: annotated BEV / camera figures showing what the camera update does.

Replays the student tracker (fused mode, seed 0) on cached detections, finds the
confirmed track whose position error shrinks most thanks to one camera update, and
draws it in BEV and on the FRONT image, plus a per-frame RMSE comparison.
"""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from fusion_lab.tracking.sensors import Sensor
from fusion_lab.workspace_loader import load_workspace
from replay import load_cache, replay

OUT = Path(__file__).parent


def nearest_gt(position, gt, gate=2.0):
    best = min(gt, key=lambda g: np.hypot(position[0] - g[1], position[1] - g[2]), default=None)
    if best is None or np.hypot(position[0] - best[1], position[1] - best[2]) > gate:
        return None
    return best


def best_camera_correction(trace):
    """Largest 3D-error reduction caused by a camera update on a confirmed track."""
    best = None
    for item in trace:
        lidar_state = {t["id"]: t for t in item["after_lidar"]}
        for track in item["after_camera"]:
            before = lidar_state.get(track["id"])
            if before is None or track["state"] != "confirmed":
                continue
            gt = nearest_gt(before["x"][:3], item["gt"])
            if gt is None:
                continue
            truth = np.array(gt[1:])
            e0 = np.linalg.norm(before["x"][:3] - truth)
            e1 = np.linalg.norm(track["x"][:3] - truth)
            if best is None or e0 - e1 > best["gain"]:
                best = {"frame": item["frame"], "id": track["id"], "gt": gt, "gain": e0 - e1,
                        "err_lidar": e0, "err_cam": e1, "item": item}
    return best


def main() -> None:
    cache = load_cache()
    fused = replay(cache, "fused", trace=True)
    lidar = replay(cache, "lidar")
    pick = best_camera_correction(fused["trace"])
    item, tid, gt = pick["item"], pick["id"], pick["gt"]
    frame = pick["frame"]
    before = next(t for t in item["after_lidar"] if t["id"] == tid)
    after = next(t for t in item["after_camera"] if t["id"] == tid)
    print(f"frame {frame} track {tid}: lidar-only error {pick['err_lidar']:.3f} m -> "
          f"after camera {pick['err_cam']:.3f} m")

    # 1) BEV
    fig, ax = plt.subplots(figsize=(7, 7))
    for k, d in enumerate(item["detections"]):
        ax.scatter(d[2], d[1], marker="x", c="gray", s=40, label="LiDAR detection" if k == 0 else None)
    for k, g in enumerate(item["gt"]):
        ax.scatter(g[2], g[1], marker="*", c="gold", edgecolors="k", s=200, label="GT vehicle centre" if k == 0 else None)
    ax.scatter(before["x"][1], before["x"][0], c="tab:blue", s=90, label="track after LiDAR update")
    ax.scatter(after["x"][1], after["x"][0], c="tab:red", s=90, label="track after camera update")
    ax.annotate("", xy=(after["x"][1], after["x"][0]), xytext=(before["x"][1], before["x"][0]),
                arrowprops=dict(arrowstyle="->", color="tab:red", lw=2))
    ax.set_title(f"BEV, frame {frame}, track {tid}\nposition error vs GT: "
                 f"{pick['err_lidar']:.3f} m -> {pick['err_cam']:.3f} m after camera update")
    ax.set_xlabel("y, left (m)")
    ax.set_ylabel("x, forward (m)")
    ax.invert_xaxis()
    ax.set_xlim(before["x"][1] + 6, before["x"][1] - 6)
    ax.set_ylim(before["x"][0] - 8, before["x"][0] + 8)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "viz_bev_camera_update.png", dpi=130)
    plt.close(fig)

    # 2) FRONT image
    ws = load_workspace()
    camera = Sensor("camera", cache["camera_calib"], ws["camera_fusion"])
    jpeg = next(f for f in cache["frames"] if f["frame"] == frame)["front_jpeg"]
    image = Image.open(io.BytesIO(jpeg))
    p_before = np.asarray(camera.get_hx(np.asmatrix(before["x"]).T)).ravel()
    p_after = np.asarray(camera.get_hx(np.asmatrix(after["x"]).T)).ravel()
    p_meas = min(item["camera_pixels"], key=lambda p: np.linalg.norm(p - p_before))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image)
    ax.scatter(*p_meas, marker="+", c="lime", s=300, linewidths=3, label="camera measurement (GT 2D box centre + noise)")
    ax.scatter(*p_before, marker="o", facecolors="none", edgecolors="tab:blue", s=160, linewidths=2,
               label=f"h(x) before camera update (residual {np.linalg.norm(p_meas - p_before):.1f} px)")
    ax.scatter(*p_after, marker="o", facecolors="none", edgecolors="tab:red", s=160, linewidths=2,
               label=f"h(x) after camera update (residual {np.linalg.norm(p_meas - p_after):.1f} px)")
    ax.set_xlim(p_meas[0] - 250, p_meas[0] + 250)
    ax.set_ylim(p_meas[1] + 150, p_meas[1] - 150)
    ax.set_title(f"FRONT camera, frame {frame}, track {tid}")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "viz_front_camera_update.png", dpi=130)
    plt.close(fig)

    # 3) per-frame RMSE, lidar-only vs fused
    def per_frame(records):
        return [np.sqrt(r["sum_sq_err"] / r["matches"]) if r["matches"] else np.nan for r in records]

    fig, ax = plt.subplots(figsize=(9, 4))
    frames = [r["frame"] for r in fused["records"]]
    ax.plot(frames, per_frame(lidar["records"]), label=f"lidar-only (RMSE {lidar['rmse']:.3f} m)", alpha=0.8)
    ax.plot(frames, per_frame(fused["records"]), label=f"lidar + camera (RMSE {fused['rmse']:.3f} m)", alpha=0.8)
    ax.axvline(frame, color="k", ls=":", label=f"frame {frame} (figures 1-2)")
    ax.set(xlabel="frame", ylabel="per-frame RMSE of matched confirmed tracks (m)",
           title="Camera update reduces per-frame tracking error (seed 0)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "viz_per_frame_rmse.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
