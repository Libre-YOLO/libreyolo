#!/bin/bash
# Probe onstart: report GPU + measure download throughput. Vast captures this
# script's stdout to /var/log/onstart.log (read it with `vast_job.py logs <id>`).
# Use this as a first launch to confirm a host is healthy and fast before you
# commit a long training run to it.
echo "PROBE_START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || echo "nvidia-smi FAILED"
# NOTE: speed.cloudflare.com/__down returns 403 from datacenter IPs. Use a large
# public file (HuggingFace CDN) as the throughput probe instead.
if command -v curl >/dev/null 2>&1; then
  curl -o /dev/null -sL -w 'DL_RESULT bytes=%{size_download} sec=%{time_total} Bps=%{speed_download}\n' \
    https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin
else
  python - <<'PY'
import urllib.request, time
u = "https://huggingface.co/bert-base-uncased/resolve/main/pytorch_model.bin"
t = time.time(); n = 0; r = urllib.request.urlopen(u, timeout=120)
while True:
    b = r.read(1 << 20)
    if not b: break
    n += len(b)
dt = time.time() - t
print(f"DL_RESULT bytes={n} sec={dt:.2f} MBps={n/1e6/dt:.1f} Mbps={n*8/1e6/dt:.0f}")
PY
fi
echo "PROBE_DONE $(date -u +%FT%TZ)"
