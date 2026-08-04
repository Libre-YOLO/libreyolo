# Triton Inference Server

LibreYOLO can run an exported model through NVIDIA Triton's HTTP V2 endpoint.
The first integration supports HTTP/HTTPS inference only. gRPC, authentication,
shared memory, and model loading or unloading are outside this contract.

Install the optional clients used for export and serving:

```bash
pip install "libreyolo[onnx,triton]"
```

## Build a CPU model repository

Export with a dynamic batch axis. This example uses YOLO9; the same layout and
commands apply to an RF-DETR detection checkpoint.

```python
from pathlib import Path

from libreyolo import LibreYOLO

model_dir = Path("triton_repo/yolo9/1")
model_dir.mkdir(parents=True, exist_ok=True)
LibreYOLO("LibreYOLO9t.pt").export(
    format="onnx",
    output_path=str(model_dir / "model.onnx"),
    dynamic=True,
    simplify=False,
)
```

Triton does not preserve ONNX custom metadata in its model-config response.
LibreYOLO therefore requires the complete exported metadata as one JSON value
named `libreyolo_metadata` in `config.pbtxt`. The deployment helper validates
the metadata, preserves ONNX input/output order, handles JSON escaping, and
pins the model to CPU:

```python
from libreyolo import create_triton_config

create_triton_config(
    "triton_repo/yolo9/1/model.onnx",
    "triton_repo/yolo9/config.pbtxt",
    model_name="yolo9",
    max_batch_size=8,
)
```

`max_batch_size: 8` matches the dynamic export and enables server batching up
to eight images per request. For a fixed batch-one ONNX model, use
`max_batch_size: 0`; LibreYOLO will then send images sequentially.

The resulting repository is:

```text
triton_repo/
  yolo9/
    config.pbtxt
    1/
      model.onnx
```

## Start and check the server

The commands below pin Triton Server 26.04. They deliberately omit Docker GPU
flags, and `KIND_CPU` in the model config prevents GPU placement.

```bash
docker run --rm --name libreyolo-triton \
  -p 8000:8000 -p 8002:8002 \
  -v "$(pwd)/triton_repo:/models:ro" \
  nvcr.io/nvidia/tritonserver:26.04-py3 \
  tritonserver --model-repository=/models --exit-on-error=true
```

In another terminal, wait for readiness before constructing the client:

```bash
until curl --fail --silent http://127.0.0.1:8000/v2/health/ready; do sleep 1; done
```

## Run and compare

```python
from libreyolo import LibreYOLO, SAMPLE_IMAGE

remote = LibreYOLO("http://127.0.0.1:8000/yolo9")
remote_result = remote.predict(SAMPLE_IMAGE)

native = LibreYOLO("LibreYOLO9t.pt")
native_result = native.predict(SAMPLE_IMAGE)

print(len(remote_result.boxes), len(native_result.boxes))
print(remote_result.boxes.xyxy[:3])
print(native_result.boxes.xyxy[:3])
```

An explicit version uses a second URL path segment, for example
`http://127.0.0.1:8000/yolo9/1`. Without it, Triton's configured version policy
selects the version. Stop and remove the test server with `docker stop
libreyolo-triton` if it is running in the background, or press Ctrl+C for the
foreground command.

The backend rejects missing LibreYOLO metadata, multiple model inputs,
configuration/metadata output mismatches, unsupported input datatypes, and
unready servers or models with a direct error. Network and request timeouts
default to 30 seconds; use `TritonBackend(url, timeout=...)` for a different
limit.
