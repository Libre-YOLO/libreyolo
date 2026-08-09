# RKNN export integration

- **LibreYOLO module:** `libreyolo/export/rknn.py` and `RknnExporter` in
  `libreyolo/export/exporter.py`
- **Clean API reference:** https://github.com/airockchip/rknn_model_zoo at
  `bad6c7334531becaf90a561988519b7bec34d0ab` - Apache-2.0
- **Reference files:** `examples/LPRNet/python/convert.py` and
  `py_utils/rknn_executor.py`
- **NOT derived from:** Ultralytics code or any AGPL implementation
- **Port method:** native integration with LibreYOLO's exporter, calibration,
  metadata, and test contracts; Rockchip's model-zoo examples supplied only the
  public vendor call sequence
- **Runtime/compiler:** `rknn-toolkit2`, separately installed and governed by
  Rockchip's custom RKNN SDK License; not bundled by LibreYOLO
- **Evidence:** mocked unit tests for API/error/cleanup behavior plus real
  Toolkit2 2.3.2 compilation and x86 simulator runs in WSL2
- **Verification status (2026-08-04):** YOLO9-t, YOLO9-E2E-t, PicoDet-s, and
  YOLO-NAS-s pass RK3588 Toolkit2 2.3.2 host simulation plus task-aware
  real-image comparison. RF-DETR and the tested G1 transformer candidates fail
  decoded-output parity; YOLOX-n also fails the raw acceptance gate. YOLO9-P2
  lacks a usable permissive trained task fixture and triggers RK3588
  register-width diagnostics. See `docs/rknn.md` for measurements and the exact
  enabled scope.
