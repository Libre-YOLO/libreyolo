---
name: libreyolo-upload-hf-dataset
description: Prepare and upload a LibreYOLO dataset repo to the HuggingFace LibreYOLO org. Use when publishing a training/eval/smoke dataset (detection, segmentation, pose, semantic, depth, or classification). Covers the redistribution license gate, rebuilding from clean upstreams, the LibreYOLO format for each task, the dataset card, and how to wire auto-download.
---

# Upload a LibreYOLO dataset repo to HuggingFace

Use this skill when publishing a dataset to `https://huggingface.co/datasets/LibreYOLO/<repo>`.

Scope: **data repos** consumed by `model.train(...)` / `model.val(...)` — a single
`.zip` (or a `download:` script) in a LibreYOLO-supported layout. The companion
skill `libreyolo-upload-hf-model` covers weights.

## The golden rule

**Never redistribute a third party's *packaged* artifact** — their `.zip`,
their tarball, their weights. Even when the underlying data is permissively
licensed, *their repackaging* may carry a license they attached, and mirroring
their file is redistributing *their* artifact.

Instead: **rebuild from the canonical upstream** and publish your own artifact
with pinned provenance. `scripts/build_imagenette.py` is the reference example —
it pulls fast.ai's Apache-2.0 tarball, pins its `sha256`, repackages to a
LibreYOLO-layout `.zip`, and uploads. Copy that shape.

## The redistribution license gate (do this FIRST)

Before downloading or hosting anything, establish the **source** license and
decide. This gate is the whole point of the skill — the bug it prevents is
someone wiring `data="foo"` to a URL nobody checked.

| Source license | Host under LibreYOLO? |
|---|---|
| Apache-2.0, MIT, BSD, CC0, CC-BY, PDDL | **Yes** — with attribution in the card |
| CC-BY-SA, ODbL (share-alike) | Yes, but the derivative inherits share-alike — note it, ask if unsure |
| CC-BY-**NC** and any `-NC` variant | **No** — non-commercial conflicts with LibreYOLO's use; substitute or gate, ask the user |
| ImageNet, "research only", custom EULA | **No** — not cleanly redistributable; find a clean substitute |
| GPL/AGPL data, unknown, or no license | **Stop, ask the user** — treat "no license" as all-rights-reserved |

Worked precedent: `imagenet10` (ImageNet-derived → not redistributable) was
**replaced** by `smoke10`, a subset of Apache-2.0 Imagenette. When the license
fails the gate, substitute a clean-sourced equivalent; do not host it anyway.

## LibreYOLO format by task

The dataset must be in the layout the loader expects for its task. Full spec:
`docs/dataset_schema.md` and the loaders under `libreyolo/data/`. Summary:

| Task | Layout | Notes |
|---|---|---|
| Detection | `images/**` + `labels/**.txt` (YOLO TXT), **or** COCO JSON | `<class> <cx> <cy> <w> <h>`, normalized |
| Instance segmentation | polygon `.txt`, **or** COCO JSON | COCO keeps holes/multi-part + RLE; YOLO TXT is one ring per instance |
| OBB | rotated-box `.txt` (`<class> x1 y1 … x4 y4`), **or** COCO JSON | 8 normalized corner coords |
| Pose | YOLO TXT + `kpt_shape` / `flip_idx` in the yaml, **or** COCO JSON | box then K keypoints |
| Semantic segmentation | `images/**` + `masks_dir/**.png` | single-channel class-ID PNG, `255`=ignore; optional `label_mapping` |
| Depth | `images/**` + `depths_dir/**` | 16-bit PNG (`depth_scale`, default 256) or `.npy`; `0`=invalid |
| Classification | ImageFolder `train/<class>/*`, `val/<class>/*` | class = sorted folder name |

**Converting into these** (target on the right):
- Already COCO JSON → keep it; wire with an `annotations:` block (no conversion).
- A competitor's YOLO-TXT export → already compatible (`data.yaml` + `labels/`).
- Roboflow → export as YOLO **or** COCO; both load.
- Raw class masks → single-channel PNGs under `masks_dir`; remap source IDs with `label_mapping`.
- Folder of labelled images → ImageFolder `train/val/<class>`.

## The dataset-repo contract

A dataset repo contains the artifact plus a card. No bare zips.

```
<repo>/
├── README.md                # dataset card: frontmatter + Provenance / Contents / Use / License
└── <name>.zip               # the data in LibreYOLO layout (or split .zips)
```

Large sets may ship split archives; keep the card + a manifest. Do not upload
the raw upstream tarball alongside the rebuilt zip.

## Dataset card template

Frontmatter keys are HF dataset-card standard. `license` is the **source**
license (see the gate). Body sections below the `---`:

```
---
license: apache-2.0
task_categories:
  - image-classification        # or object-detection / image-segmentation / depth-estimation
tags:
  - libreyolo
  - <dataset-tag>
pretty_name: <Pretty Name> (LibreYOLO)
size_categories:
  - 10K<n<100K
---
```

- **# `<Pretty Name>` (LibreYOLO)** — one sentence: what it is, which task, why hosted.
- **## Provenance** — source URL + org + license; pinned source `sha256`; the
  transform applied (e.g. "extracted `.tgz` → `.zip`", "converted COCO → YOLO TXT");
  the line "No third-party repackaging used."
- **## Contents** — the layout and per-split counts; class list or class count.
- **## Use with LibreYOLO** — a short snippet, e.g.:

      from libreyolo import LibreYOLO
      model = LibreYOLO("<checkpoint>")
      model.val(data="<name>")      # auto-downloads this dataset

- **## License** — the source license, inherited, with attribution.

## Wiring auto-download

Two mechanisms — pick by task:

1. **YAML datasets** (detection / seg / pose / semantic / depth) —
   `libreyolo/config/datasets/<name>.yaml` with a `download:`:
   - a **single URL** to the HF zip → auto-downloads on first use, OR
   - a `download: |` Python script (multi-line) for multi-step fetch/convert —
     runs only under `allow_download_scripts=True`. Scripts must
     `from libreyolo.data.utils import download` and pull from
     `huggingface.co/datasets/LibreYOLO/...` or official sources
     (`cocodataset.org`) — **never** a competitor host.
2. **Classification known-names** — add to `_KNOWN_DATASETS` in
   `libreyolo/data/classify_dataset.py`, value = the HF `resolve/main/<name>.zip`
   URL.

Both are guarded:
- `tests/unit/test_data_utils.py::test_builtin_dataset_yamls_do_not_use_ultralytics_assets`
- `tests/unit/test_data_utils.py::test_known_classify_datasets_are_libre_hosted`

Run them after wiring; they must stay green.

## Upload workflow

1. **License gate** (section above). If it fails: substitute a clean source or stop and ask.
2. Fetch the canonical upstream and **pin its `sha256`** (reproducibility + tamper check).
3. Convert to the LibreYOLO layout for the task; verify counts and a few samples.
4. Zip with `ZIP_STORED` (images are already compressed — deflate wastes CPU for ~0 gain).
5. Write the dataset card (provenance + license).
6. Create + upload with `huggingface_hub` (`repo_type="dataset"`), token from env
   only (`HF_TOKEN`), never committed:

   ```python
   from huggingface_hub import HfApi
   api = HfApi(token=os.environ["HF_TOKEN"])
   api.create_repo("LibreYOLO/<name>", repo_type="dataset", private=False, exist_ok=True)
   api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                   repo_id="LibreYOLO/<name>", repo_type="dataset")
   api.upload_file(path_or_fileobj="<name>.zip", path_in_repo="<name>.zip",
                   repo_id="LibreYOLO/<name>", repo_type="dataset")
   ```
7. **Verify the resolve URL serves real bytes**, not an LFS pointer:
   `https://huggingface.co/datasets/LibreYOLO/<name>/resolve/main/<name>.zip`
8. Wire auto-download (section above) and run the guard tests.
9. End-to-end check: `resolve_classify_data("<name>")` (classify) or
   `model.val(data="<name>.yaml")` (yaml) on a cleared cache.

Prefer a committed build script (like `scripts/build_imagenette.py`) over
one-off commands, so the artifact is reproducible from source.

## Ask the user when

- The source license is CC-BY-NC, research-only, unknown, or absent.
- There is no pinnable upstream version/commit (reproducibility needs it).
- The data contains faces / PII / sensitive content (privacy layers on top of license).
- The dataset is large (multi-GB) — hosting + egress cost is a decision.
- A dataset repo with that name already exists (overwrite is destructive).
- No supported layout fits the task cleanly.

## Common traps

- **Mirroring a competitor's packaged zip** — the exact bug this skill exists to prevent. Rebuild from upstream.
- Re-hosting ImageNet-derived data (redistribution restricted) — substitute Imagenette (Apache-2.0).
- CC-BY-NC data in a commercially-usable project — the `-NC` blocks it.
- Not pinning the source `sha256` — a silent upstream re-cut changes the data under you.
- Zipping already-JPEG images with deflate — use `ZIP_STORED`.
- Renaming class folders — class index is the **sorted folder name**; reordering changes label IDs. Keep upstream naming or document the mapping.
- Uploading without a card / license — HF flags "no license" and users can't tell if it is safe to use.
- Pointing the classify resolver at a `.tgz` — it extracts `.zip` only; convert first (yaml `download:` scripts can handle either via the `download()` helper).
