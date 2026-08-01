---
name: launch-serverless-gpu-job
description: >-
  Launch a one-off / batch GPU job (training, evaluation, export, benchmarking)
  on a rented or serverless GPU when you don't have a local one — or need a
  bigger/faster card than you have. Covers three providers: Vast.ai (cheap
  marketplace GPUs you rent + tear down; fully automated here via a helper CLI),
  Modal (managed serverless, per-second, auto scale-to-zero — what LibreYOLO's
  own nightly e2e runs on), and Beam (managed serverless, simple decorator). Use
  whenever someone wants to "run this on a cloud GPU", "train on a rented GPU",
  "spin up a GPU for a job", "launch on Vast/Modal/Beam", or check what's still
  billing. NOT for standing inference endpoints / autoscaling serving — this is
  for batch jobs that start, do work, and stop.
---

# Launch a serverless / rented GPU job

For running a **batch job** — a training run, a `val` sweep, an export, a
benchmark — on someone else's GPU and only paying for the run. Three providers,
one decision up front, then follow that provider's section.

## Pick the provider first

| | **Vast.ai** | **Modal** | **Beam** |
|---|---|---|---|
| Model | Marketplace: you rent a raw box, drive it over SSH, **destroy it yourself** | Managed serverless: job is a decorated Python fn, runs on demand | Managed serverless: job is a decorated Python fn |
| Teardown | **Manual** — a forgotten box bills until destroyed (the main risk) | **Automatic** — scales to zero when the fn returns | **Automatic** — scales to zero |
| Cost | Cheapest raw $/hr (e.g. RTX 4090 ~$0.35/hr) | Higher $/hr, but per-second and no idle waste | Per-second, autoscale |
| Best for | Long training on a tight budget; any GPU you can name | Reproducible batch jobs & CI; caching across runs | Quick jobs with a tiny bit of setup |
| In this repo | helper `scripts/vast_job.py` (full lifecycle) | **already used**: `tools/ci/modal_nightly.py` runs nightly e2e | — |
| Setup cost | Vast account + API key + 2FA + SSH key | `pip install modal` + `modal token new` | `pip install beam-client` + `beam configure` |

Rule of thumb: **Modal** if you want a job that provisions, runs, and cleans
itself up with no teardown discipline (and LibreYOLO already has a working
example). **Vast** when raw $/hr matters and you'll babysit it. **Beam** for a
quick managed job with a minimal API.

## Universal money safety

Every path here spends real money the moment a GPU spins up.

- **Vast** bills wall-clock until you `destroy`. Idle GPUs bill exactly like busy
  ones, so killing a training job saves nothing on its own; only `stop` or
  `destroy` moves the meter. A *stopped* box still bills its disk: forgotten,
  that is the big trap; chosen, it is a 50x saving (see "Pause instead of
  destroy"). Use `audit` / `guard` / `destroy` (below).
- **Modal / Beam** scale to zero when the function returns, so a forgotten box is
  much less likely — but a hung or infinite-timeout job **still bills**. Always
  set a `timeout=`, and don't leave an interactive session holding a GPU.
- **Pull/persist checkpoints at milestones** (to HF or a volume) so a crash or
  credit-out never loses the model.

## Where to put the data

Decide this before launching — a slow staging path can cost more than the GPU
time it delays.

- **Dataset → one tar in a Hugging Face dataset repo** (the default). HF's CDN
  serves ~100 MB/s to every provider, the same recipe works on Vast, Modal and
  Beam, and a private repo only needs `HF_TOKEN` in the job env. Pack it once
  (`tar cf mydata.tar mydata/`), upload
  (`huggingface-cli upload <you>/<name> mydata.tar --repo-type dataset`), and
  stage it on-box with `hf_hub_download` — `onstart/train.sh.tmpl` does exactly
  this.
- **Don't** scp a dataset up from your home connection, and don't pull from slow
  origins on the box (cocodataset.org throttles one connection to ~2 MB/s; if you
  must, `aria2c -x16 -s16 -k1M <url>` recovers it).
- **Checkpoints / results → push back to HF (or a provider Volume) at
  milestones**, not just at the end — the box is ephemeral, so a crash,
  preemption, or running out of credit otherwise loses the run. On Vast, also
  `pull` a local copy before `destroy`.

---

## Provider 1 — Vast.ai (rent + drive + destroy)

Everything goes through one stdlib-only helper that bakes in the sharp edges
(2FA-gated ops, transient offers, `-y`-on-destroy, ssh-detach, banner-tolerant
ssh). Run any Python:

```bash
export PYTHONUTF8=1                       # avoids Windows console crashes
PY=python                                 # any python; the helper is stdlib-only
SK=skills/launch-serverless-gpu-job/scripts/vast_job.py
$PY $SK <command> ...
# commands: preflight tfa-login launch wait list cost audit ssh exec tail pull logs guard destroy destroy-all
```

### One-time setup

```bash
python -m venv ~/.vast-cli
~/.vast-cli/Scripts/python -m pip install vastai   # Windows; Linux/mac: ~/.vast-cli/bin/pip
vastai set api-key <YOUR_KEY>                       # key needs FULL instance permissions
vastai tfa login --method-type totp -c <6-digit>    # 2FA session, lasts ~7 days
$PY $SK preflight                                    # expect: authenticated + 2FA active + SSH key
```

The helper finds the `vastai` binary via `~/.vast-cli` or PATH; override with
`VAST_BIN=/path/to/vastai`. SSH pubkey defaults to `~/.ssh/id_ed25519.pub`
(override with `VAST_PUBKEY`).

- `FAIL: no API key` → the **user** runs `vastai set api-key` (never paste a key
  into chat; rotate any that leaks).
- `FAIL: instance ops BLOCKED (no 2FA session)` → Vast requires a 2FA session for
  **all** instance ops; ask the user for a TOTP code and run
  `$PY $SK tfa-login --code NNNNNN`.

### Launch → drive → tear down

```bash
$PY $SK launch --gpu RTX_4090 --max-price 0.6 --disk 40 --onstart <file.sh> --label myjob
$PY $SK wait  <id>                                  # poll to running; prints COLD_START (~30s cached)
$PY $SK exec  <id> --cmd "cd /root/repo && python train.py" --log /root/train.log
$PY $SK tail  <id> /root/train.log -n 60            # watch a remote log
$PY $SK pull  <id> /root/out/best.pt ~/ckpts/       # scp a checkpoint DOWN
$PY $SK cost                                        # credit + running $/hr + runway
$PY $SK destroy <id>                                # ALWAYS destroy when done
$PY $SK audit                                       # end-of-session: is anything still billing?
```

- `--gpu` is Vast's `gpu_name` with underscores: `RTX_4090`, `RTX_5090`,
  `RTX_3090`, `A100_SXM4`, `H100_SXM`. Omit for cheapest-of-any; `--gpu-ram 24000`
  filters VRAM (MB). The helper searches many offers and retries across churn
  (`no_such_ask ... not available` is normal, not an error).
- Use **`exec`** (not `ssh`) for anything long-running: it detaches with
  `</dev/null` so the job survives disconnect; then `tail` the log.
- **Right-size by $/job, not $/hr.** A tiny model half-idles a 4090; a cheaper
  card can be cheaper *per epoch* even if slower (watch VRAM).
- **CUDA graphs cost VRAM, so budget for them when packing several trainings on
  one card.** Capture holds a private memory pool: measured on 8x RTX 5090 with
  3 yolov9t trainings per card, 18.6 GB eager against 20.3 GB graphed, i.e.
  about +0.6 GB per lane. The pool shrinks under pressure rather than OOMing
  (a lane capped at 4.1 GB reserved 3.9 GB and lost no throughput), but a big
  model is a different story: rfdetr added 5.42 GB of private pool and OOMed a
  16 GB card outright. Measure per family before choosing packing depth.
- For unattended runs, arm the watchdog in the background:
  `$PY $SK guard <id> --max-hours 24 --done-file /root/JOB_DONE` (destroys on the
  done-marker or timeout; best-effort — needs this process alive).

### Pause instead of destroy: a 50x lever, and when it is wrong

`destroy` is not the only way to stop spending. `stop` halts GPU billing while
keeping the container disk, so the box comes back with your environment,
dataset and checkpoints already in place.

```bash
vastai stop instance  <ID>     # GPU billing stops, disk keeps billing
vastai start instance <ID>     # resume, same disk, IF the GPUs are still free
vastai reboot instance <ID>    # stop+start, for a wedged container
vastai recycle instance <ID>   # destroy+recreate the container, keeps the contract
```

Measured on an 8x RTX 4090 with a 250 GB disk (2026-07-31):

| state | rate | note |
|---|---|---|
| running | $3.4828/hr | GPUs plus disk |
| **stopped** | **$0.0694/hr** | disk only, **50x cheaper, 98% saving** |
| destroyed | $0 | disk and data gone |

The stopped rate is arithmetic you can predict before renting:

```
stopped $/hr = disk_GB * storage_cost_per_GB_per_month / 730
             = 250 * 0.20 / 730 = $0.0694/hr   ($50/month; cheap is not free)
```

**Stop or destroy? Compare against what a rebuild costs.** Rebuilding means
renting again, pulling the image, installing, and re-staging the data. Measured
on a real campaign box: about 15 minutes of setup ($0.87 at this rate) plus
43 GB of inbound transfer ($0.11), so roughly $1.00. Against $0.0694/hr:

- Coming back within **~14 hours**: STOP. The disk bill is less than a rebuild.
- Gone longer than that: DESTROY, and rebuild from the HF-hosted dataset.
- Add the host lottery to the destroy side of the ledger. Finding a host whose
  egress works cost about $1 of duds in one session, so in practice stopping
  wins out to roughly a day.

**The risk that makes stop unsafe for scarce hardware:** stopping releases the
GPUs. Nothing reserves them for you, so `start` succeeds only if the host still
has them free, and a popular multi-GPU config can be gone when you return. Your
DISK is safe; your GPUs are not. For a config you must have back, either keep
it running or destroy it and plan to hunt again.

### Two more levers worth knowing before you rent

**Right-size the disk. It bills on ALLOCATED GB, not used GB.** The box above
allocated 250 GB and used 45 GB, so its stopped rate was 5.5x higher than
needed; 120 GB would have cost $0.033/hr stopped. You cannot shrink a disk
after creation, so decide at rental: staged data plus checkpoints plus roughly
30% headroom.

**Interruptible (bid) instances trade reliability for price, and the discount
varies.** `min_bid` on the offer is the floor. On the box above it was $3.20
against $3.4828 on demand, a mere 8%, which is not worth preemption risk; other
hosts discount far more, so read `min_bid` rather than assuming. Only consider
it when the work is genuinely resumable, and remember a preempted instance
keeps billing its disk exactly like a stopped one.

### `onstart` scripts (provided)

- `onstart/probe.sh` — first-launch health/throughput probe (GPU + download speed).
- `onstart/train.sh.tmpl` — training template: installs the tools the bare pytorch
  image lacks (git/aria2/unzip + cv2 libs), stages a dataset from HF, runs your
  `TRAIN_CMD`, writes `/root/JOB_DONE`. Fill the `__ALL_CAPS__` placeholders.

### Vast gotchas (baked into the helper)

1. 2FA session mandatory for instance ops (~7 days); no bypass.
2. API keys can be permission-scoped (create/list yes; `show instance`/`logs` no)
   → the helper falls back to SSH for logs.
3. `destroy` needs `-y` or the raw CLI hangs *while billing*.
4. Poll `instances-v1`, not `show instance` (often perm-gated).
5. A **stopped** instance is not free: its disk bills until `destroy`. That is a
   trap when you forgot about it and a deliberate 50x saving when you meant it,
   so see "Pause instead of destroy" above. Run `audit` after any crash or
   disconnect to catch the forgotten kind.

---

## Provider 2 — Modal (managed serverless)

LibreYOLO already runs its **nightly e2e suite on Modal** — read
`tools/ci/modal_nightly.py` as the canonical, working, in-repo example (it clones
a ref, installs with `uv`, runs `make test_nightly` on an L4, and caches weights
in a Modal Volume across runs). Copy that pattern for real jobs.

Setup: `pip install modal` then `modal token new` (opens a browser).

Minimal training job — a self-contained `modal_train.py`:

```python
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")     # cv2 system libs
    .pip_install("libreyolo[rfdetr]")                  # or a git+https / wheel URL
)
app = modal.App("libreyolo-train")
cache = modal.Volume.from_name("libreyolo-cache", create_if_missing=True)

@app.function(gpu="A100", timeout=6 * 60 * 60, volumes={"/cache": cache})
def train():
    import os, torch
    os.environ["HF_HOME"] = "/cache/hf"                # cache weights/datasets across runs
    assert torch.cuda.is_available()
    from libreyolo import LibreYOLO9
    m = LibreYOLO9(None, "t")
    m.train(data="coco.yaml", epochs=100, project="/cache/runs")
    cache.commit()                                     # persist the volume

@app.local_entrypoint()
def main():
    train.remote()                                     # runs on Modal's GPU
```

Run it: `modal run modal_train.py`.

- GPU strings: `T4 L4 A10 L40S A100 A100-40GB A100-80GB H100 H200 B200`; multi-GPU
  `gpu="H100:8"`; fallback preference `gpu=["H100", "A100-80GB"]`.
- Billing is **per-second** and the container **scales to zero** when `train()`
  returns — no box to forget. Still set `timeout=` so a hung run can't bill forever.
- Persist anything you want to keep in the Volume (or push to HF) — the container
  filesystem is ephemeral.

---

## Provider 3 — Beam (managed serverless)

Setup: `pip install beam-client` then `beam configure` (paste the token from the
Beam dashboard).

Minimal job — `beam_train.py`:

```python
from beam import function, Image, Volume

image = (
    Image(python_version="python3.11")
    .add_commands(["apt-get update && apt-get install -y libgl1 libglib2.0-0"])
    .add_python_packages(["libreyolo[rfdetr]"])
)

@function(
    gpu="RTX4090",                                     # or "A10G" / "H100"
    image=image,
    volumes=[Volume(name="libreyolo-cache", mount_path="./cache")],
    timeout=86400,                                     # seconds; -1 disables (avoid)
)
def train():
    import os, torch
    os.environ["HF_HOME"] = "./cache/hf"
    assert torch.cuda.is_available()
    from libreyolo import LibreYOLO9
    LibreYOLO9(None, "t").train(data="coco.yaml", epochs=100, project="./cache/runs")

if __name__ == "__main__":
    train.remote()                                     # runs on Beam's GPU
```

Run it: `python beam_train.py` (the `.remote()` call dispatches to Beam).

- GPU menu is smaller than Modal/Vast: `A10G` (24Gi), `RTX4090` (24Gi),
  `H100` (80Gi); `gpu=["H100","A10G"]` sets preference; multi-GPU via
  `gpu_count=N` (by request).
- Managed + autoscale-to-zero like Modal; per-second billing. Persist state to a
  `Volume` (mounted into the container) or push to HF, since the container is
  otherwise ephemeral. Keep a real `timeout` (never leave `-1` on a batch job).

> Beam coverage here is intentionally minimal and less battle-tested than the Vast
> and Modal paths. Verify the current decorator/GPU names against
> https://docs.beam.cloud if a call is rejected.
