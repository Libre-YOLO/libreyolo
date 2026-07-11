# ADR 0011: Export support tiers

## Status

Accepted.

## Decision

Export support is described by `(family, task, format)` entries in
`libreyolo/export/support.py`. Family and task keys use their canonical names
from the model registry and `libreyolo.tasks.TASKS`.

Each combination has one tier:

- `validated`: numeric parity is covered in CI or a documented nightly run.
- `experimental`: conversion is available, but numeric parity is not guaranteed.
- `blocked`: preflight raises `NotImplementedError` with a reason before tracing.

CoreML conversion without a macOS prediction run can be experimental, but it
cannot be validated. Documentation is generated from the matrix and checked for
drift.

## Consequences

Exporters warn for experimental combinations. Blocked combinations fail before
dependency checks, calibration loading, tracing, or artifact creation. Adding a
validated entry requires a parity test and updating the `since` field.
