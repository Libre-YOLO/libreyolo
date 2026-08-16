# Hugging Face Hub integration

LibreYOLO has two separate Hub transports:

- `hf://` and bare `owner/repo` references select one detector checkpoint file.
- `hf+vlm://owner/repo@<commit>` selects one immutable, multi-file VLM
  publication artifact.

Detector checkpoints use the standard LibreYOLO metadata schema described in
[`checkpoint_schema.md`](checkpoint_schema.md). VLM artifacts use the stricter
directory contract in [`vlm_hub_artifact.md`](vlm_hub_artifact.md). The two URI
forms are not interchangeable.

Detector Hub transport is optional:

```bash
pip install libreyolo[hf]
```

Loading an immutable VLM artifact also needs the VLM runtime:

```bash
pip install "libreyolo[vlm,hf]"
```

`import libreyolo` never imports `huggingface_hub`; it is only loaded when a
Hub reference or push is actually used.

## Loading detector checkpoints from the Hub

Any repository that contains a LibreYOLO checkpoint can be loaded directly:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("someuser/their-finetune")          # bare owner/repo
model = LibreYOLO("hf://someuser/their-finetune")     # explicit form
model = LibreYOLO("hf://someuser/repo@main/best.pt")  # pin revision and file
```

Resolution rules:

- A local file or directory with the same name always wins over the bare
  `owner/repo` form. Use the `hf://` form to bypass that precedence.
- The bare form is only treated as a Hub reference when the last segment has
  no file extension, so paths like `weights/model.pt` keep their meaning.
- Without an explicit filename, the single `*.pt` (or `*.safetensors`) file
  in the repo is used. Repos with several checkpoints raise an error listing
  them; select one with `hf://owner/repo/<filename>`.

- A revision passed as `@revision` must not contain `/`; use a commit SHA or a
  slash-free branch name.

Downloads go to the shared `huggingface_hub` cache and are loaded through the
same safe path as local files: `torch.load(weights_only=True)` plus metadata
validation.

Hub checkpoints are identified by their LibreYOLO metadata. Recognized
upstream checkpoints are still auto-converted, but a file with no metadata
that cannot be converted is rejected rather than guessed at, because raw
tensor keys from an arbitrary Hub repo can match an unrelated family and fail
much later with a confusing error. To load such a file anyway, download it and
pass a local path, which re-enables legacy architecture detection.

Private and gated repos work once you are authenticated (see below).

## Pushing detector checkpoints to the Hub

Any loaded model can be published, together with an auto-generated model card
derived from its checkpoint metadata (family, size, task, classes, metrics):

```python
model = LibreYOLO("runs/detect/train/weights/best.pt")
model.push_to_hub("someuser/my-finetune")                  # public
model.push_to_hub("someuser/my-finetune", private=True)    # private
model.push_to_hub("someuser/my-finetune", license="mit",
                  metrics={"mAP50": 0.62})
```

The repo is created if missing. The checkpoint is uploaded as `model.pt`, so
repeated pushes update the same file. Anyone can then load it back with
`LibreYOLO("someuser/my-finetune")`.

To upload automatically when a training run finishes, use the Hub logger:

```python
model.train(data="data.yaml", epochs=100, loggers="hf:someuser/my-finetune")
```

or, with options:

```python
from libreyolo.training import HuggingFaceHubLogger

model.train(
    data="data.yaml",
    loggers=HuggingFaceHubLogger("someuser/my-finetune", private=False),
)
```

The logger verifies write access at construction time, which also creates the
target repository up front, so a credential or repository-name problem fails
before training starts instead of discarding hours of work at the end. On
train end it pushes `weights/best.pt` (falling back to `last.pt`) with the
final metrics rendered into the model card.

Note the differing defaults: `push_to_hub` creates a public repo (you are
explicitly publishing), while the logger creates a private one, because it
uploads unattended and a model trained on proprietary data must not become
public by surprise. Pass `private=False` to publish from training. Repos that
already exist keep their current visibility either way.

This detector push path, including `HuggingFaceHubLogger`, does not accept
LibreVLM directory checkpoints. VLM publication is an explicit, reviewed
operation described below.

## Loading and publishing VLM artifacts

A published VLM adapter is addressed only by a canonical URI with a lowercase,
40-character commit SHA:

```python
from libreyolo import LibreVLM
from libreyolo.models.vlm import inspect_vlm_hub_artifact

uri = "hf+vlm://someuser/strawberry-vlm@0123456789abcdef0123456789abcdef01234567"
manifest = inspect_vlm_hub_artifact(uri)  # manifest only; no tensor payload
model = LibreVLM(uri)
```

Branches, tags, abbreviated hashes, bare repository IDs, query strings, and
file suffixes are rejected. `LibreVLM(uri)` validates the artifact, acquires
and validates the exact Qwen base snapshot recorded by it, and revalidates both
before use. The base weights are not stored in the adapter repository.

To materialize an artifact without loading a model:

```python
from libreyolo.models.vlm import download_vlm_artifact, validate_vlm_artifact

info = download_vlm_artifact(uri, "artifacts/strawberry-vlm")
validate_vlm_artifact(info.root)
```

The destination must not already exist. Online inspection and download also
require the repository tree at that commit to equal the manifest inventory.
`local_files_only=True` validates the cached allowlisted files, but cannot
prove that the remote commit has no additional files.

Upload is separate from training and from the detector Hub logger:

```python
from libreyolo.models.vlm import push_vlm_artifact

uri = push_vlm_artifact(
    "artifacts/strawberry-vlm",
    "someuser/strawberry-vlm",
)  # private=True by default; returns an immutable hf+vlm:// URI
```

`push_vlm_artifact` accepts only a fully validated artifact and refuses any
pre-existing repository, including an empty one. It creates a private repo,
uploads the exact artifact in one commit, and verifies that commit through a
fresh download. With `private=False`, visibility changes only after that
verification succeeds. See [`vlm_hub_artifact.md`](vlm_hub_artifact.md) for the
required human evidence gates and build workflow.

## Authentication

Reading public repos needs no login. Private repos, gated repos, and all
pushes need a Hugging Face token, resolved the standard way:

1. Run `hf auth login` once (stores a token on the machine), or
2. set the `HF_TOKEN` environment variable, or
3. pass `token="hf_..."` to `push_to_hub`, `HuggingFaceHubLogger`, or the VLM
   Hub functions.

Pushing requires a token with write scope; create one at
<https://huggingface.co/settings/tokens>. Error messages repeat these steps
whenever authentication is the likely cause.
