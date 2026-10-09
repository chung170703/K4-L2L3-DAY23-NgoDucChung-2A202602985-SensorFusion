# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Ngô Đức Chung
- MSSV: 2A202602985
- Email: (không ghi công khai trên repo public)
- Link repo (fork): https://github.com/chung170703/K4-L2L3-DAY23-NgoDucChung-2A202602985-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`): ghi trên LMS (hash của commit cuối, không thể tự nhúng vào chính commit đó)

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: `compare`, frame 0–198 (199 frame), `training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord`, seed 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701, 0.7004, tp=519 / fp=16 / fn=222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1503 m, 502, 11.3436, 0, 239, 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: 0.1359 m, 502, 9.2668, 0, 239, 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:

  Cả hai mode ghép đúng 502 cặp, 0 ghost và 239 miss, nên `precision_track = 502/(502+0) = 1.00`
  và `coverage = 502/519 = 0.967` (đều vượt ngưỡng 0.75 / 0.70). RMSE 0.150 m (lidar) và
  0.136 m (fused) cùng dưới 0.45 m, và `rmse_fused − rmse_lidar = −0.015 m ≤ 0.05 m`.
  Fused thấp hơn lidar khoảng 9.6 % RMSE (sum_sq_err 11.34 → 9.27): ở 137/195 frame có
  sai số khác nhau, sai số fused nhỏ hơn. Số cặp, ghost, miss và `mean_confirmed_tracks`
  giống hệt nhau vì camera chỉ cập nhật trạng thái EKF, không cộng/trừ score, không tạo
  hay xóa track; vòng đời track do lidar quyết định hoàn toàn. 239 miss gồm 8 miss ở
  frame 0–3 (2 xe × 4 frame, track chưa được xác nhận: frame đầu có confirmed là frame 4 vì
  score phải vượt 0.8 với bước 1/6), 222 `det_fn` của detector LiDAR (recall 0.70) và 9 miss
  còn lại ở các frame sau, nơi track bị mất hoặc cần xác nhận lại sau khoảng trống detection.
  Camera trong lab là tâm hộp 2D GT FRONT cộng nhiễu seed 0, không phải detector ảnh, nên
  mức giảm RMSE chỉ cho thấy phép đo 2D có nhiễu làm tinh chỉnh vị trí, không chứng minh
  chất lượng một camera detector thật. Vì có một segment và một seed nên chênh 0.015 m
  chưa đủ để kết luận về mặt thống kê.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. **Đo lidar 3D và camera 2D trong EKF.** Lidar có `z = (x, y, z)` (mét, 3×1), `R = diag(0.1²)` và
   `h(x)` tuyến tính, `H` 3×6 (`Sensor.get_H`, phép quay xe→cảm biến, cột vận tốc bằng 0). Camera có
   `z = (u, v)` pixel (2×1), `R = diag(5², 5²)`, `h(x)` phi tuyến: `u = c_i − f_i·y_s/x_s`,
   `v = c_j − f_j·z_s/x_s` với `p_s = R p + t` (`camera_measurement_prediction`, `camera_fusion.py`),
   nên `H` là Jacobian 2×6 theo quy tắc dây chuyền. Camera không quan sát độ sâu: một pixel ứng với
   cả một tia 3D, nên nó chỉ khống chế hướng ngang/dọc, còn khoảng cách vẫn phải do lidar giữ.
2. **Gating Mahalanobis.** Khoảng cách Euclid bỏ qua độ bất định: cùng một khoảng cách, track có `P`
   lớn thì hợp lý còn track có `P` nhỏ thì không. `d² = γᵀS⁻¹γ` (`mahalanobis_distance`) chuẩn hóa
   innovation theo `S = HPHᵀ + R`, và `chi2_gate` loại cặp có `d²` vượt `chi2.ppf(0.995, dim_meas)`
   (dim_meas = 3 hoặc 2). Gating trước khi gán giúp greedy không nhận cặp vô nghĩa và tránh làm
   hỏng `x`, `P` của track bằng một đo ngoại lai. Cặp ngoài FOV bị loại từ `association_cost_matrix`
   trước khi chiếu hay tính Mahalanobis.
3. **Track-then-fuse.** Detector lidar tạo hộp 3D từng frame, tracker tạo/giữ track từ lidar, rồi camera
   chỉ cập nhật track có sẵn. Trong `grade_run.log`, `confirmed`, `matches`, `ghosts`, `misses` ở hai
   mode lidar và fused hoàn toàn trùng nhau (cùng 502 matches, 239 miss, `mean_confirmed_tracks`
   2.5226) trong khi `sum_sq_err` khác (11.34 so với 9.27). Nếu là fuse-then-track thì camera sẽ
   làm thay đổi số track hoặc số detection.
4. **Camera lệch calibration.** Pixel dự báo `h(x)` bị dịch hệ thống nên innovation `γ = z − h(x)` có
   trung bình khác 0 và cùng dấu qua nhiều frame (không phải nhiễu trắng), `d²` tăng. Hệ quả: nhiều
   cặp bị chi² gate loại, hoặc nếu lọt qua thì `K·γ` kéo `x` lệch về một phía, làm RMSE fused tăng
   so với lidar, có thể vượt ngưỡng `rmse_fused − rmse_lidar ≤ 0.05 m`. Lệch xa theo hướng ngang còn
   đẩy track ra ngoài FOV ảnh.
5. **Sensor tường minh ở frame rỗng.** Khi `meas_list` rỗng, không có `meas.sensor` nào để suy ra lượt
   hiện tại là lidar hay camera, mà `manager.manage_tracks` cần biết điều đó: lidar rỗng phải trừ score
   các track nằm trong FOV rồi xét xóa, còn camera rỗng thì không được đụng score/xóa/tạo track. Vì
   vậy `associate_and_update` luôn nhận `sensor` và luôn gọi `manage_tracks(unassigned_tracks,
   unassigned_meas, sensor)` (test `test_empty_pass_runs_management`). Lidar quyết định init/score/delete
   vì nó có đo 3D đủ tin cậy để sinh track và phát hiện track biến mất; camera 2D thiếu độ sâu và ở lab
   này là đo giả lập từ GT có nhiễu, nên chỉ được dùng để sửa trạng thái EKF
   (`handle_updated_track` chỉ ghi hit khi `sensor.name == "lidar"`).
6. **Vòng đời track** (`track_management.py`). Track mới có `score = 1/window = 1/6`, state `initialized`.
   Hit lidar cộng 1/6 (tối đa 1), miss lidar trong FOV trừ 1/6. Xác nhận khi `score > 0.8`
   (cần 4 hit, nên frame 4 là frame đầu có track confirmed). Track chưa confirmed mà có hit thì là
   `tentative`. Đã `confirmed` thì không hạ trạng thái khi miss, một miss đơn lẻ chỉ giảm score
   (1 → 5/6). Xóa khi `P[0,0]` hoặc `P[1,1] > max_P = 9` (bất kể score), hoặc confirmed có
   `score < 0.6`, hoặc chưa confirmed có `score ≤ 0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không


## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude Sonnet 5.5)
- Dùng cho phần nào (hàm, câu hỏi, debug): đọc đề lab, cài Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`), sửa lỗi kiểu trả về của `mahalanobis_distance`, tải dữ liệu, chạy `fusion-run-lab` và soạn nháp báo cáo này.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` (128 passed, 0 failed/xfailed); chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198; kiểm tra `matches+ghosts==confirmed` và `matches+misses==valid_gt` trên 199 record mỗi mode; `python tools/check_submission.py`.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
