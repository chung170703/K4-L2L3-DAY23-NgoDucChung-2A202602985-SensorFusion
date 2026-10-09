"""Bonus: camera extrinsic drift vs. innovation, gating and RMSE.

Rotates the tracker's FRONT-camera extrinsic about the camera up (yaw) or left
(pitch) axis by several angles while keeping the simulated pixels (true GT box
centres + seeded noise) unchanged, then replays the student tracker on the cached
detections. Writes ``calibration_sweep.json`` and ``calibration_sweep.png``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from fusion_lab import tracking_params as params
from replay import load_cache, replay

OUT = Path(__file__).parent
LEVELS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
GATE = float(chi2.ppf(params.gating_threshold, df=2))


def pair_table(probe):
    """Map (frame, track id, measurement pixels) -> (d^2, innovation) for every in-FOV pair."""
    return {(p["frame"], p["track"], p["z"]): (p["d2"], p["gamma"]) for p in probe}


def drift_vs_baseline(baseline, probe):
    """Follow the pairs that the undrifted camera accepts through a drifted run."""
    table = pair_table(probe)
    kept = [(k, v) for k, v in baseline.items() if v[0] < GATE]
    d2_d, g_d, g_0, still_in = [], [], [], 0
    for key, (d2_0, gamma_0) in kept:
        if key in table:
            d2, gamma = table[key]
            d2_d.append(d2)
            g_d.append(gamma)
            g_0.append(gamma_0)
            still_in += d2 < GATE
    g_d, g_0 = np.array(g_d).reshape(-1, 2), np.array(g_0).reshape(-1, 2)
    return {
        "baseline_pairs_in_gate": len(kept),
        "pairs_still_in_fov": len(d2_d),
        "pairs_still_in_gate": int(still_in),
        "frac_pairs_retained": still_in / len(kept),
        "median_d2_of_those_pairs": float(np.median(d2_d)) if d2_d else None,
        "mean_gamma_u_shift_px": float((g_d[:, 0] - g_0[:, 0]).mean()) if len(g_d) else None,
        "mean_gamma_v_shift_px": float((g_d[:, 1] - g_0[:, 1]).mean()) if len(g_d) else None,
        "mean_abs_gamma_px": float(np.abs(g_d).mean()) if len(g_d) else None,
    }


def main() -> None:
    cache = load_cache()
    lidar = replay(cache, "lidar")
    rows = [{"axis": "-", "degrees": None, "mode": "lidar-only", "rmse": lidar["rmse"],
             "matches": lidar["matches"], "ghosts": lidar["ghosts"], "misses": lidar["misses"],
             "camera_updates": 0}]
    baseline = pair_table(replay(cache, "fused", probe_distances=True)["probe"])
    for axis, name in (("z", "yaw"), ("y", "pitch")):
        for deg in LEVELS:
            r = replay(cache, "fused", axis, deg, probe_distances=True)
            g = r["camera_gammas"]
            row = {
                "axis": name, "degrees": deg, "mode": "fused", "rmse": r["rmse"],
                "d_rmse_vs_lidar": r["rmse"] - lidar["rmse"], "matches": r["matches"],
                "ghosts": r["ghosts"], "misses": r["misses"], "camera_updates": r["camera_updates"],
                "mean_abs_gamma_updates_px": float(np.abs(g).mean()) if len(g) else None,
                **drift_vs_baseline(baseline, r["probe"]),
            }
            rows.append(row)
            print(name, deg, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k not in ("axis", "mode", "degrees")}, flush=True)

    (OUT / "calibration_sweep.json").write_text(json.dumps(
        {"gate_chi2_2dof": GATE, "levels_deg": LEVELS, "rows": rows}, indent=2))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for name, style in (("yaw", "o-"), ("pitch", "s--")):
        sel = [r for r in rows if r["axis"] == name]
        x = [r["degrees"] for r in sel]
        axes[0].plot(x, [r["rmse"] for r in sel], style, label=f"fused, {name} drift")
        axes[1].plot(x, [100 * r["frac_pairs_retained"] for r in sel], style, label=name)
        axes[2].plot(x, [r["mean_gamma_u_shift_px"] if name == "yaw" else r["mean_gamma_v_shift_px"] for r in sel], style, label=name)
    axes[0].axhline(lidar["rmse"], color="k", ls=":", label="lidar-only")
    axes[0].axhline(0.45, color="r", ls=":", label="0.45 m threshold")
    axes[0].set(xlabel="extrinsic offset (deg)", ylabel="RMSE (m)", title="RMSE vs drift")
    axes[1].set(xlabel="extrinsic offset (deg)", ylabel="% of baseline camera pairs still inside gate",
                title=f"Gating (chi2 99.5%, 2 dof = {GATE:.1f})")
    axes[2].set(xlabel="extrinsic offset (deg)", ylabel="mean innovation shift of retained pairs (px)",
                title="yaw -> shift in u, pitch -> shift in v")
    for a in axes:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT / "calibration_sweep.png", dpi=130)


if __name__ == "__main__":
    main()
