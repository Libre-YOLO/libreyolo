# Depth Anything 3 provenance

- Upstream code: `ByteDance-Seed/Depth-Anything-3`
- Pinned code commit: `41736238f5bced4debf3f2a12375d2466874866d`
- Code license: Apache-2.0
- Upstream weights: `depth-anything/DA3MONO-LARGE`
- Pinned weight revision: `f465978e618db8cc79c83b8bbf24964857db1875`
- Weight license: Apache-2.0
- LibreYOLO artifact: `LibreYOLO/LibreDepthAnything3l-depth`

LibreYOLO vendors the minimal DA3MONO-LARGE ViT-L and DPT dependency graph.
The converter strips only the official high-level API's `model.` prefix and
wraps all 406 unchanged tensors in checkpoint schema v1.0. Strict key and
shape parity is checked before conversion verification.

The official head emits positive relative depth. LibreYOLO reproduces the
official sky handling and then applies `1 / clamp(depth, 1e-6)` inside the
network wrapper so the public result follows the depth-task contract: relative
inverse depth, higher means closer, with no metric unit.

DA3METRIC-LARGE is excluded because metric depth needs a different public
contract. Small/Base are excluded because their any-view camera/ray outputs
need a multi-image geometry API. Large/Giant/Nested weights use
CC-BY-NC-4.0 and are not present in any LibreYOLO download path.
