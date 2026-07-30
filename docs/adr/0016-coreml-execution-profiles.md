# ADR 0016: Core ML execution profiles

## Status

Accepted.

## Context

ADR 0011 records export support by family, task, and format. Core ML behavior
also depends on model size, input canvas, precision, embedded NMS, graph-shape
specializations, converter passes, and runtime compute units. A family-level
`validated` row must not make an untested variant look hardware-validated.

## Decision

`libreyolo/export/coreml_profiles.py` is the machine-readable source of truth
for exact Core ML campaign candidates and for the smaller set of execution
profiles that passed saved-package and public-API gates on Apple hardware.

Each profile identifies:

- family, task, size, and fixed canvas;
- raw versus embedded-NMS output;
- graph-shape specializations such as a SAM package's maximum prompt count,
  external class cardinality, raw graph class/logit width, pose keypoint
  geometry and per-class schema, or classifier activation;
- checkpoint topology distinctions that are not encoded by family and size,
  including restoration checkpoint variant, architecture signature, output
  scale, and embedding width;
- conversion precision and compute units;
- admitted runtime compute units and the safe default;
- any required converter-pass exclusions; and
- the checkpoint or deterministic fixture used for validation.

Version 2 promotion also binds three identities:

- the exact live export source, using either a hash of the admitted source
  artifact or a deterministic hash of the prepared PyTorch tensor state and
  canonicalized traced graph;
- the final converter-produced Core ML protobuf, after conversion, optional
  NMS wrapping, and multifunction assembly, together with the declared host
  contract; and
- the independently generated Apple-hardware evidence record.

All three identities are required. A recipe row without them is a campaign
candidate, not a validated execution profile, and cannot be looked up by
profile id.

`compute_units="validated"` is the public default. Export first uses any
candidate row only to plan graph capture, then hashes the exact prepared source
and resolves again. Only that second, source-verified result may select the
profile's tested conversion planner. After conversion, the final deployment
ABI must also match before the package can be marked validated. Runtime
resolves the profile and verifies the package metadata, deployment ABI, and
model interface before Core ML compiles a proxy. If there is no exact profile,
`compute_units="validated"` fails closed; experimental conversion or loading
requires an explicit native planner. A promoted profile defaults to CPU-only;
explicit `all` is admitted only where a separate public-runtime gate measured
it.

Exact profiles are FP32 and raw-output unless the registry states otherwise.
Another size, canvas, precision, NMS setting, prompt bound, frozen class
cardinality, or planner remains experimental even when its family/task support
row is validated. It may convert, but the exporter warns and does not write a
validated profile marker.

Validated packages record the profile id/version, source and deployment-ABI
identities, evidence digest, conversion and runtime compute-unit policy,
precision, graph specializations, validation reference, M4 toolchain scope,
and required converter passes. Export saves to a staging directory, reloads
the persisted spec without native compilation, and validates the metadata,
deployment ABI, and interface before replacing a destination. The loader
repeats those checks before creating an `MLModel` proxy. Missing or modified
fields fail closed. Legacy version-1 packages or packages without the marker
may load only through an explicit CPU-only opt-in with a warning; the validated
default and accelerator planners never silently accept them.

The broad support matrix preserves historical conversion/parity evidence, but
it is not the runtime promotion registry. At adoption of version 2, every
legacy row was deliberately demoted until it can be re-exported and replayed
on Apple hardware with the new source and final-ABI evidence. Therefore the
version-2 registry initially contains zero promoted profiles and the public
validated route intentionally rejects every candidate.

## Validation meaning

An execution profile proves conversion/runtime fidelity for the recorded graph
and host contract. It does not prove model accuracy, arbitrary custom
checkpoints, every Apple device or OS, Neural Engine placement, or device
performance. Those limitations remain explicit in the generated support
ledger.

## Consequences

- New hardware promotions require an exact registry entry and a support-ledger
  row.
- Historical parity evidence is not relabeled as version-2 evidence; every
  promotion requires a fresh artifact and fresh hardware replay.
- Variant campaigns may add profiles without changing family-level converter
  code.
- Embedded NMS, FP16, flexible shapes, and alternate compute planners need
  independent hardware gates.
- Converter-pass workarounds are narrow, versioned, recorded in metadata, and
  must improve the fixed parity gate rather than weaken it.
