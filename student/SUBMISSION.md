# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Ngô Đức Chung
- MSSV: 2A202602985
- Link repo (fork): https://github.com/chung170703/K4-L2L3-DAY23-NgoDucChung-2A202602985-SensorFusion

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

Đã làm cả 3 mục bonus (export CVAT, trực quan hoá, phân tích calibration). Bằng chứng nằm trong `student/bonus/`;
các kịch bản chỉ chạy lại tracker của chính lab trên kết quả detector đã cache, **không** sửa platform hay Part A–D,
và `student/artifacts/` vẫn là lần chạy chấm điểm gốc (`compare --seed 0`, frame 0–198). Replay ở mức 0° tái tạo đúng
metrics gốc (lidar 0.1503 m / 502 matches, fused 0.1359 m / 502 matches), nên các kịch bản so sánh được với bài nộp.

### Bonus 1 — Phân tích calibration (mục 2, +4)

- Code: `student/bonus/cache_frames.py` (cache detector, GT, pixel camera seed 0), `replay.py` (chạy lại tracker, xoay extrinsic
  camera của *tracker*), `calibration.py` (quét và vẽ). Số liệu: `calibration_sweep.json`; hình: `calibration_sweep.png`.
- Cách làm: pixel đo vẫn là tâm hộp 2D GT FRONT cộng nhiễu seed 0 (như lab), còn extrinsic camera mà EKF dùng
  bị xoay quanh trục lên (yaw) hoặc trục trái (pitch) 0.25°, 0.5°, 1°, 2°, 5° (6 mức kể cả 0°, hai trục).
  Với mỗi mức, lấy 585 cặp (track, đo camera) mà camera chuẩn (0°) chấp nhận qua cổng χ² (ngưỡng 10.60, 2 bậc tự do,
  p = 0.995) rồi theo dõi chính các cặp đó khi lệch.

| Trục | Lệch | RMSE fused (m) | − RMSE lidar (m) | Số update camera | Cặp còn trong cổng | Dịch innovation γ (u / v, px) |
|---|---|---|---|---|---|---|
| yaw | 0° | 0.1359 | −0.0145 | 509 | 100.0 % | +0.0 / +0.0 |
| yaw | 0.25° | 0.1732 | +0.0229 | 486 | 95.6 % | −7.7 / +0.0 |
| yaw | 0.5° | 0.2112 | +0.0608 | 322 | 65.8 % | −33.0 / −9.2 |
| yaw | 1° | 0.1726 | +0.0222 | 20 | 12.8 % | −68.1 / −14.6 |
| yaw | 2° | 0.1760 | +0.0257 | 19 | 10.4 % | −103.8 / −14.3 |
| yaw | 5° | 0.1503 | 0.0000 | 28 | 5.6 % | −210.2 / −14.8 |
| pitch | 0.25° | 0.1750 | +0.0247 | 503 | 99.0 % | −0.0 / +7.5 |
| pitch | 0.5° | 0.2233 | +0.0730 | 370 | 76.1 % | −0.2 / +15.4 |
| pitch | 1° | 0.1876 | +0.0372 | 87 | 15.4 % | −0.3 / +31.5 |
| pitch | 2° | 0.2319 | +0.0815 | 241 | 14.7 % | −0.8 / +63.0 |
| pitch | 5° | 0.1503 | +0.0000 | 8 | 10.9 % | −0.3 / +158.5 |

(RMSE lidar-only = 0.1503 m; ở mọi mức, matches = 502, ghost = 0, miss = 239 như bài nộp.)

- **Triệu chứng trên innovation:** lệch extrinsic làm dịch có hệ thống trung bình của γ = z − h(x), không phải nhiễu trắng.
  Yaw chủ yếu dịch thành phần u (≈ −31 px/độ ở 0.25°, ≈ −42 px/độ ở 5°), pitch chủ yếu dịch thành phần v (≈ +31 px/độ ở mọi mức);
  do chỉ xét các cặp còn lại trong vùng nhìn thấy nên giá trị lớn chỉ mang tính chỉ báo. Hệ quả là d² = γᵀS⁻¹γ tăng (trung vị d² của các cặp đó từ 1.7 ở 0° lên 8.7 ở yaw 0.5°, 22.8 ở 1°, 391.6 ở 5°).
- **Vì sao gating chặn hoặc không chặn:** σ = 5 px nên cổng χ² rộng chỉ cỡ 20–30 px. Lệch ≥ 1° đẩy hầu hết cặp ra ngoài cổng
  (còn 5–15 % cặp), nên update camera giảm mạnh từ 509 (yaw ≥ 1° chỉ còn 19–28, pitch 5° còn 8) và ở 5° fused quay về đúng lidar-only (0.1503 m).
  Lệch 0.25–0.5° nhỏ hơn bề rộng cổng nên phần lớn cặp vẫn lọt (95.6 % / 65.8 % ở yaw) trong khi đo đã bị lệch vài chục pixel:
  các update này kéo state lệch và đẩy RMSE lên (0.173 → 0.211 m ở yaw, 0.175 → 0.223 m ở pitch). Đây là vùng nguy hiểm nhất:
  gating không chặn được lỗi nhỏ nhưng hệ thống, và RMSE không đơn điệu theo độ lệch (yaw 1° tốt hơn 0.5° khi số update lọt cổng giảm từ 322 xuống 20).
- **Hệ quả với rubric:** RMSE vẫn dưới 0.45 m ở mọi mức, nhưng điều kiện nhất quán `rmse_fused − rmse_lidar ≤ 0.05 m` bị vi phạm
  ở yaw 0.5° (+0.061 m), pitch 0.5° (+0.073 m) và pitch 2° (+0.082 m). Hạn chế: một segment, một seed; pixel camera là GT có nhiễu
  nên không phản ánh lỗi của detector ảnh; "cặp còn trong cổng" được tính trên các cặp của lần chạy 0°, còn bản thân quỹ đạo track
  thay đổi theo từng mức lệch (phản hồi từ các update).

### Bonus 3 — Export track sang CVAT và kiểm tra trực quan (mục 2, +3)

- Code `student/bonus/cvat_export.py` (dùng `fusion_lab.export_cvat.export_tracks_json`), chạy lại tracker `fused`, seed 0, trên
  frame 40–70 (31 frame). Đầu ra: `cvat_tracks.json` (trạng thái từng track mỗi frame: id, `state`, vị trí/vận tốc trong hệ xe,
  kích thước, yaw, hộp 2D chiếu lên ảnh FRONT) và `cvat_annotations.xml` ("CVAT 1.1", rectangle track, thuộc tính `track_id`, `state`).
  Hộp 2D là bao ngoài của 8 đỉnh hộp 3D của track được chiếu bằng mô hình camera của lab (Part G).
- Đã tạo task CVAT trên app.cvat.ai (task #2657002, 31 ảnh FRONT, nhãn `vehicle_track`) và import `cvat_annotations.xml` bằng định dạng
  CVAT 1.1; import thành công, mở job thấy các track đúng vị trí xe. Ảnh chụp: `cvat_ghost_track6_frame46.jpg`.
  Ảnh Waymo không được commit (chỉ nằm trong `.cache/` local).
- **Ghost trên ảnh:** ở frame 46 (CVAT frame 6) hộp `track_id: 6`, `state: tentative` nằm trên dải bụi cây ven đường, không có xe
  (Waymo GT ở frame này chỉ có 3 xe, không xe nào ở đó). Track 6 sinh ra ở frame 44 từ một detection LiDAR không ứng với xe nào, tồn tại 6 frame
  (44–49) và bị xóa trước khi đủ điểm xác nhận. Vì chưa bao giờ `confirmed` nên không được tính vào `ghost_track_frames`
  (= 0 trong lần chạy chấm điểm), đúng thiết kế lifecycle (xác nhận khi score > 0.8): đây là ghost bị bộ lọc vòng đời chặn,
  khác với ghost lọt qua và bị tính. Cùng đoạn này còn có track 7 (frame 60–64) cũng không được xác nhận; hai track 0 và 1 liên tục
  `confirmed` suốt 31 frame.
- **Đổi ID:** kiểm tra gán track–GT trên toàn 199 frame (gate 2 m, theo track `confirmed`) cho thấy không xe GT nào đổi track ID;
  xe `VJ3F-…` được giữ bởi track 5 trong frame 48–58 (confirmed) rồi mất track trong khi GT còn tới frame 69, và xe `8EFR…` chỉ được
  track 10 nhận từ frame 98 (confirmed) dù GT có từ frame 67. Đây là miss (mất track), không phải đổi ID.

### Bonus 2 — Trực quan hoá BEV và ảnh camera (mục 2, +3)

Code `student/bonus/visualize.py`. Ba ảnh có chú thích (frame và track được chọn tự động là cặp có camera update
cải thiện sai số vị trí nhiều nhất trong lần chạy `fused`, nên là ví dụ tốt nhất chứ không phải ví dụ điển hình):

1. `viz_bev_camera_update.png` — BEV frame 150, track 10: sai số 3D so với GT giảm từ 0.221 m (sau update LiDAR) xuống 0.014 m
   (sau update camera); thấy cả đo LiDAR, tâm GT và hai vị trí track.
2. `viz_front_camera_update.png` — cùng frame trên ảnh FRONT: residual pixel giữa đo camera và h(x) giảm từ 23.7 px xuống 6.5 px
   sau update camera.
3. `viz_per_frame_rmse.png` — RMSE theo frame của lidar-only (0.150 m) và fused (0.136 m). Camera không tốt hơn ở mọi frame
   (ví dụ frame khoảng 5–25 fused cao hơn lidar-only), cải thiện tập trung ở khoảng frame 70–150.


## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude Code (Claude Sonnet 5.5)
- Dùng cho phần nào (hàm, câu hỏi, debug): Claude Code viết code bốn hàm Part E–H (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`), sửa lỗi kiểu trả về của `mahalanobis_distance`, tải dữ liệu, chạy `fusion-run-lab`, soạn nội dung báo cáo này (kể cả sáu câu giải thích) từ số liệu của lần chạy, và viết toàn bộ code bonus trong `student/bonus/` (replay tracker, quét calibration, hình trực quan, export/import CVAT). Phần việc trên trình duyệt (tạo và xóa task CVAT) cũng do Claude thực hiện theo yêu cầu của tôi.
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
