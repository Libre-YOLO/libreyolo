# ADR 0011: Export support tiers

## Status

Accepted.

## Decision

Export support is described by `(family, task, format)` entries in
`libreyolo/export/support.py`. Family and task keys use their canonical names
from the model registry and `libreyolo.tasks.TASKS`.

Each combination has one tier:

- `validated`: numeric parity is covered in CI or a documented nightly run.
- `available`: conversion is implemented, but numeric runtime parity evidence is incomplete or has not been recorded.
- `blocked`: preflight raises `NotImplementedError` with a reason before tracing.

A CoreML conversion without a macOS prediction run is available but not
validated. Documentation is generated from the matrix and checked for drift.

## Consequences

Validated and available combinations proceed without an acknowledgement or
blanket warning. Their recorded evidence and constraints remain visible in the
generated documentation. Blocked combinations fail before dependency checks,
calibration loading, tracing, or artifact creation. Adding a validated entry
requires a parity test and updating the `since` field.
