# triton

- **LibreYOLO surface:** `libreyolo/backends/triton.py`, shared exported-runtime
  metadata parsing, factory and CLI routing, tests, and `docs/triton.md`.
- **Protocol and client references:** NVIDIA Triton Inference Server HTTP V2
  client and model-configuration documentation, plus
  `https://github.com/triton-inference-server/client` at commit
  `7d9f59050731bdcd8f5902a6836b3ebcbf2a57b5` (BSD-3-Clause).
- **Tested server pin:** Triton Server `2.68.0` / container `26.04-py3`;
  server tag object `dd16c0cad0399599240e8eb37178865eca994303` (BSD-3-Clause).
- **Derivation:** the implementation is original code derived from the public
  Triton V2 protocol/API contract and LibreYOLO's existing `BaseBackend` and
  export-metadata contracts. No NVIDIA source code is copied or adapted.
- **Distribution:** `tritonclient[http]` is an optional runtime dependency and
  is not vendored. No third-party source or model weights are added, so the
  repository source and weight notice files do not change.
