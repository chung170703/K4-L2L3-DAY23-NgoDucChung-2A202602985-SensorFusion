# Bonus evidence

Run from the repo root with the lab environment active:

```bash
python student/bonus/cache_frames.py --config student/config/paths.yaml   # ~ detector once, writes .cache/ (not committed)
cd student/bonus && python calibration.py && python visualize.py
```

| File | What |
|---|---|
| `cache_frames.py`, `replay.py` | Cache detector output; replay the student tracker, optionally with a rotated camera extrinsic |
| `calibration.py`, `calibration_sweep.json`, `calibration_sweep.png` | Extrinsic drift sweep (yaw/pitch, 0.25-5 deg) |
| `visualize.py`, `viz_*.png` | Annotated BEV / FRONT-image / per-frame RMSE figures |

The platform and `student/artifacts/` are untouched; the 0 deg replay reproduces the graded metrics.
