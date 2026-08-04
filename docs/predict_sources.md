# Prediction sources

LibreYOLO classifies prediction inputs before loading them. Webcam indices and
network stream URLs therefore enter the video path directly instead of being
interpreted as image filenames.

| Source | Python value | Behavior |
| --- | --- | --- |
| Image or image URL | `"image.jpg"`, `PIL.Image`, NumPy, tensor | One result |
| Image directory or list | `"images/"`, `[image1, image2]` | List, or a lazy generator with `stream=True` |
| Video file or URL | `"clip.mp4"` | List by default; generator with `stream=True` |
| Screen | `"screen"`, `"screen 1"` | One capture; continuous with `stream=True` |
| Webcam | `0`, `1`, or a numeric CLI source | Continuous; requires `stream=True` in Python |
| Network stream | `rtsp://`, `rtmp://`, `tcp://`, `udp://`, or an HLS `.m3u8` URL | Continuous; requires `stream=True` in Python |
| YouTube | A `youtube.com` or `youtu.be` page URL | Resolved through `yt-dlp`, then read as a live stream |
| Multiple cameras | A list of stream sources or a `.streams` text file | One capture thread per source; results are yielded with the source in `result.path` |

## Python

Live sources are unbounded, so the Python API requires lazy consumption:

```python
from libreyolo import LibreYOLO

model = LibreYOLO("LibreYOLO9t.pt")

for result in model(0, stream=True):
    print(result.path, result.frame_idx, result.boxes)
```

RTSP and YouTube use the same call:

```python
for result in model("rtsp://127.0.0.1:8554/camera", stream=True):
    ...

for result in model("https://youtu.be/VIDEO_ID", stream=True):
    ...
```

Install the YouTube resolver only when needed:

```bash
pip install "libreyolo[stream]"
```

By default each capture thread keeps only its newest frame. This bounds memory
and latency when inference is slower than capture. Set `stream_buffer=True` to
preserve every captured frame at the cost of a growing queue and increasing
latency.

## Multiple cameras

Pass a Python list:

```python
sources = [0, "rtsp://127.0.0.1:8554/loading-dock"]
for result in model(sources, stream=True):
    print(result.path, result.frame_idx)
```

Or put one source per line in `cameras.streams`; blank lines and lines beginning
with `#` are ignored:

```text
# Local webcam
0

# Docker-published camera
rtsp://127.0.0.1:8554/loading-dock
```

```python
for result in model("cameras.streams", stream=True):
    ...
```

Capture is concurrent, while inference and result delivery remain ordered by
availability. Each result carries a per-source `frame_idx`. Credentials in a
stream URL are redacted from `result.path`.

## CLI

The CLI enables lazy streaming automatically for live inputs. It emits one JSON
object per frame in JSON mode.

```bash
libreyolo predict source=0 model=LibreYOLO9t.pt show=true
libreyolo predict source=rtsp://127.0.0.1:8554/camera model=LibreRFDETRn.pt
libreyolo predict source=cameras.streams model=LibreYOLO9t.pt --json
```

Use `q` to stop a displayed stream or `Ctrl+C` otherwise.

## Local RTSP loop

One reproducible local setup uses MediaMTX as the RTSP server and FFmpeg as the
publisher:

```bash
docker run --rm --name libreyolo-mediamtx -p 8554:8554 bluenviron/mediamtx:latest
ffmpeg -re -stream_loop -1 -i clip.mp4 -c copy -rtsp_transport tcp -f rtsp rtsp://127.0.0.1:8554/camera
```

Then consume `rtsp://127.0.0.1:8554/camera`. OpenCV must have the corresponding
video backend enabled; the standard `opencv-python` wheels include FFmpeg.
