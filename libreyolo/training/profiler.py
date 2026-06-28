"""Lightweight built-in training profiler.

Answers one question fast: **where does each training step's time go, and is the
GPU starved?** It profiles a short window of *real* training steps and reports
two complementary views:

1. **Real timing (unsynced).** The first half of the window runs with no extra
   synchronization, so the measured step time and the dataloader stall reflect
   true overlap behavior. This drives the honest verdict (dataloader-bound vs
   compute-bound), the real img/s, and the GPU-idle estimate.
2. **Compute composition (synced).** The second half brackets each phase with a
   CUDA sync to attribute GPU time to forward / backward / optimizer / to_device.

Splitting the window matters: syncing every phase serializes the GPU and hands
the dataloader workers slack, which *hides* starvation. So we never derive the
verdict from synced numbers — only the composition.

It writes a self-contained ``profile_report.html`` (auto-opened in the browser)
and a Chrome/Perfetto trace (``profile_trace.json``), plus ``profile_summary.json``.

Zero overhead when disabled: the trainer holds ``None`` and the hot-loop calls
short-circuit to a shared no-op context.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Optional

import torch
from torch.profiler import record_function

PHASES = ("to_device", "forward", "backward", "optimizer")
_NULL = contextlib.nullcontext()


class TrainStepProfiler:
    """Profiles a short window of training steps and reports the breakdown."""

    def __init__(
        self,
        *,
        device: torch.device,
        warmup: int = 5,
        active: int = 20,
        trace: bool = True,
        open_report: bool = True,
        save_dir: Optional[Path] = None,
        logger=None,
        meta: Optional[dict] = None,
    ) -> None:
        self.device = device
        self.warmup = max(0, int(warmup))
        self.active = max(2, int(active))
        self.trace = trace
        self.open_report = open_report
        self.save_dir = Path(save_dir) if save_dir else None
        self.logger = logger
        self.meta = meta or {}

        self._is_cuda = getattr(device, "type", str(device)) == "cuda"
        # First half of the active window is unsynced (real timing); second half
        # is synced (compute composition).
        self._half = max(1, self.active // 2)
        self._step_i = 0          # total iterations seen (incl. warmup)
        self._m = 0               # measured iterations (0-based within window)

        # Real (unsynced) timing.
        self._real_step_ms = 0.0
        self._real_step_n = 0
        self._real_dl_ms = 0.0
        self._real_dl_n = 0
        self._prev_fetch: Optional[float] = None

        # Synced compute composition.
        self._sums = {p: 0.0 for p in PHASES}
        self._synced_n = 0

        self.finished = False
        self.summary: Optional[dict] = None

        self._torch_prof = None
        if self.trace:
            self._start_torch_profiler()

    # -- internals -----------------------------------------------------------

    def _sync(self) -> None:
        if self._is_cuda:
            torch.cuda.synchronize(self.device)

    def _active_now(self) -> bool:
        return (not self.finished) and self._step_i >= self.warmup

    def _unsynced_phase(self) -> bool:
        return self._active_now() and self._m < self._half

    def _synced_phase(self) -> bool:
        return self._active_now() and self._m >= self._half

    def _start_torch_profiler(self) -> None:
        try:
            from torch.profiler import ProfilerActivity, profile, schedule

            acts = [ProfilerActivity.CPU]
            if self._is_cuda:
                acts.append(ProfilerActivity.CUDA)
            self._torch_prof = profile(
                activities=acts,
                schedule=schedule(
                    wait=0, warmup=self.warmup, active=self.active, repeat=1
                ),
                record_shapes=False,
                with_stack=False,
                profile_memory=False,
            )
            self._torch_prof.start()
        except Exception:
            self._torch_prof = None  # trace is best-effort; never break training

    # -- public hooks (called from the training loop) ------------------------

    def phase(self, name: str):
        """Context for one phase: labels the trace across the whole active
        window, and additionally sync-times it during the synced half."""
        if not self._active_now():
            return _NULL
        return _PhaseTimer(self, name)

    def wrap_loader(self, iterable):
        """Yield batches while measuring real step time + dataload stall."""
        it = iter(iterable)
        while True:
            t0 = time.perf_counter()
            try:
                with record_function("step/dataload"):
                    batch = next(it)
            except StopIteration:
                return
            t1 = time.perf_counter()
            if self._unsynced_phase():
                self._real_dl_ms += (t1 - t0) * 1000.0
                self._real_dl_n += 1
                if self._prev_fetch is not None:
                    self._real_step_ms += (t1 - self._prev_fetch) * 1000.0
                    self._real_step_n += 1
            self._prev_fetch = t1
            yield batch

    def step(self) -> None:
        """Advance one iteration; finalizes + reports when the window closes."""
        if self.finished:
            return
        if self._torch_prof is not None:
            try:
                self._torch_prof.step()
            except Exception:
                pass
        if self._synced_phase():
            self._synced_n += 1
        if self._step_i >= self.warmup:
            self._m += 1
        self._step_i += 1
        if self._m >= self.active:
            self._finish()

    # -- finalize ------------------------------------------------------------

    def _finish(self) -> None:
        self.finished = True
        trace_path = None
        if self._torch_prof is not None:
            try:
                self._torch_prof.stop()
                if self.trace and self.save_dir is not None:
                    self.save_dir.mkdir(parents=True, exist_ok=True)
                    trace_path = self.save_dir / "profile_trace.json"
                    self._torch_prof.export_chrome_trace(str(trace_path))
            except Exception:
                trace_path = None
        self._build_summary(trace_path)
        timeline_path = None
        if trace_path is not None:
            timeline_path = self._write_timeline_html(self._curate_trace(trace_path))
        self._report(trace_path, timeline_path)
        if self.open_report and timeline_path is not None:
            try:
                import webbrowser

                webbrowser.open(timeline_path.as_uri())
            except Exception:
                pass

    def _build_summary(self, trace_path) -> None:
        synced_n = max(self._synced_n, 1)
        comp = {p: self._sums[p] / synced_n for p in PHASES}
        comp_total = sum(comp.values()) or 1.0

        real_step = self._real_step_ms / self._real_step_n if self._real_step_n else 0.0
        real_dl = self._real_dl_ms / self._real_dl_n if self._real_dl_n else 0.0
        real_compute = max(real_step - real_dl, 0.0)
        idle_frac = (real_dl / real_step) if real_step > 0 else 0.0
        batch = self.meta.get("batch", 0) or 0
        img_s = (batch / (real_step / 1000.0)) if real_step > 0 else 0.0

        self.summary = {
            "meta": self.meta,
            "window": {
                "warmup": self.warmup,
                "real_steps": self._real_step_n,
                "synced_steps": self._synced_n,
            },
            "real": {
                "step_ms": round(real_step, 3),
                "dataload_ms": round(real_dl, 3),
                "compute_ms": round(real_compute, 3),
                "img_per_s": round(img_s, 2),
                "gpu_idle_fraction": round(idle_frac, 3),
            },
            "composition_ms": {p: round(comp[p], 3) for p in PHASES},
            "composition_total_ms": round(comp_total, 3),
            "bound": "dataloader" if idle_frac >= 0.2 else "compute",
            "trace": str(trace_path) if trace_path else None,
        }
        if self.save_dir is not None:
            try:
                self.save_dir.mkdir(parents=True, exist_ok=True)
                (self.save_dir / "profile_summary.json").write_text(
                    json.dumps(self.summary, indent=2)
                )
            except Exception:
                pass

    def _emit(self, line: str) -> None:
        if self.logger is not None:
            self.logger.info(line)
        else:
            print(line)

    def _report(self, trace_path, timeline_path=None) -> None:
        s = self.summary
        r = s["real"]
        m = self.meta
        idle = r["gpu_idle_fraction"] * 100.0
        comp_total = s["composition_total_ms"] or 1.0
        bar_w = 20

        self._emit("=" * 64)
        self._emit(f"  LibreYOLO training profile — {m.get('model', '?')}")
        self._emit(
            f"  device={m.get('device','?')}  batch={m.get('batch','?')}  "
            f"imgsz={m.get('imgsz','?')}  amp={m.get('amp','?')}  workers={m.get('workers','?')}"
        )
        self._emit(
            f"  window: {s['window']['real_steps']} real-timed + "
            f"{s['window']['synced_steps']} compute-split (+{self.warmup} warmup)"
        )
        self._emit(
            f"  REAL step {r['step_ms']:.1f} ms = dataload {r['dataload_ms']:.1f} ms "
            f"+ compute {r['compute_ms']:.1f} ms  ->  {r['img_per_s']:.1f} img/s"
        )
        if s["bound"] == "dataloader":
            self._emit(
                f"  >> VERDICT: DATALOADER-BOUND — GPU idle ~{idle:.0f}% "
                "(waiting on data)."
            )
            self._emit(
                "     Levers: workers↑, cache='ram'/'disk', lighter aug (mosaic), "
                "or larger batch."
            )
        else:
            self._emit(
                f"  >> VERDICT: COMPUTE-BOUND — GPU idle only ~{idle:.0f}% (healthy)."
            )
        self._emit("  GPU compute composition (synchronized):")
        for p in PHASES:
            ms = s["composition_ms"][p]
            frac = ms / comp_total
            fill = int(round(frac * bar_w))
            bar = "#" * fill + "." * (bar_w - fill)
            self._emit(f"    {p:<10} {ms:8.2f} ms  |{bar}| {frac*100:5.1f}%")
        if timeline_path:
            self._emit(f"  timeline:  {timeline_path}")
            if self.open_report:
                self._emit("             (opening in your browser…)")
        if trace_path:
            self._emit(f"  raw trace: {trace_path}  (load in Perfetto/Nsight if you want)")
        self._emit("=" * 64)

    def _curate_trace(self, trace_path):
        """Reduce the raw torch trace to the bounded set worth drawing.

        Keeps our ``step/*`` phase spans (CPU + GPU projections), the
        significant cpu ops, the GPU kernels and memcpys, and the CPU->GPU flow
        links — within a steady window of ~3 middle training steps.
        """
        try:
            import json
            data = json.loads(Path(trace_path).read_text(encoding="utf-8"))
        except Exception:
            return None
        events = data.get("traceEvents", []) if isinstance(data, dict) else data

        steps = sorted(
            (e for e in events
             if e.get("ph") == "X" and str(e.get("name", "")).startswith("ProfilerStep")),
            key=lambda e: e.get("ts", 0),
        )
        if steps:
            mid = len(steps) // 2
            sel = steps[max(0, mid - 1): mid + 2] or steps[:3]
            w0 = sel[0]["ts"]
            w1 = sel[-1]["ts"] + sel[-1].get("dur", 0)
        else:
            w0 = w1 = None

        def in_win(ts):
            return w0 is None or (w0 <= ts <= w1)

        LANE = {"pcpu": 0, "cpu": 1, "pgpu": 2, "gpu": 3, "mem": 4}
        kept = []
        for e in events:
            if e.get("ph") != "X":
                continue
            cat = e.get("cat", "")
            name = str(e.get("name", ""))
            ts = e.get("ts")
            dur = e.get("dur", 0) or 0
            if ts is None or not in_win(ts):
                continue
            if cat == "user_annotation" and name.startswith("step/"):
                lane, label = "pcpu", name[5:]
            elif cat == "gpu_user_annotation" and name.startswith("step/"):
                lane, label = "pgpu", name[5:]
            elif cat == "cpu_op" and dur >= 20:
                lane, label = "cpu", name
            elif cat == "kernel" and dur >= 2:
                lane, label = "gpu", name
            elif cat in ("gpu_memcpy", "gpu_memset"):
                lane, label = "mem", name
            else:
                continue
            kept.append({"name": label, "_ts": ts, "_dur": dur,
                         "lane": LANE[lane], "cat": lane})
        if not kept:
            return None

        CAP = 8000
        if len(kept) > CAP:
            keepers = [e for e in kept if e["cat"] in ("pcpu", "pgpu", "mem")]
            rest = sorted((e for e in kept if e["cat"] in ("cpu", "gpu")),
                          key=lambda e: e["_dur"], reverse=True)
            kept = keepers + rest[: max(0, CAP - len(keepers))]

        base = min(e["_ts"] for e in kept)
        for e in kept:
            e["t"] = round((e["_ts"] - base) / 1000.0, 3)
            e["d"] = round(e["_dur"] / 1000.0, 4)
            del e["_ts"]
            del e["_dur"]

        lane_rows = {}
        for li in range(5):
            evs = sorted([e for e in kept if e["lane"] == li], key=lambda x: x["t"])
            ends = []
            for e in evs:
                placed = False
                for i, end in enumerate(ends):
                    if e["t"] >= end:
                        e["row"] = i
                        ends[i] = e["t"] + max(e["d"], 0.001)
                        placed = True
                        break
                if not placed:
                    e["row"] = len(ends)
                    ends.append(e["t"] + max(e["d"], 0.001))
            lane_rows[li] = max(len(ends), 1)

        s_by_id, f_by_id = {}, {}
        for e in events:
            if e.get("cat") != "ac2g":
                continue
            ph, i, ts = e.get("ph"), e.get("id"), e.get("ts")
            if i is None or ts is None or not in_win(ts):
                continue
            if ph == "s":
                s_by_id[i] = ts
            elif ph == "f":
                f_by_id[i] = ts
        flows = []
        for i, sts in s_by_id.items():
            fts = f_by_id.get(i)
            if fts is None:
                continue
            flows.append([round((sts - base) / 1000.0, 3),
                          round((fts - base) / 1000.0, 3)])
            if len(flows) >= 4000:
                break

        total = max((e["t"] + e["d"] for e in kept), default=1.0)
        return {"events": kept, "flows": flows, "lane_rows": lane_rows,
                "total_ms": round(total, 3), "meta": self.meta}

    def _write_timeline_html(self, curated):
        """Render the curated trace into a self-contained timeline.html."""
        if self.save_dir is None or not curated:
            return None
        import json
        try:
            payload = json.dumps(curated).replace("</", "<\\/")
            page = _TIMELINE_HTML.replace("/*__DATA__*/", payload)
            path = self.save_dir / "timeline.html"
            path.write_text(page, encoding="utf-8")
            return path
        except Exception:
            return None

    def _open_in_perfetto(self, trace_path, timeout: float = 30.0) -> None:
        """Open the trace as a GPU/CPU timeline in Perfetto.

        Serves a tiny local page that loads the trace same-origin and hands the
        bytes to ui.perfetto.dev via the official postMessage API — sidestepping
        the mixed-content block that defeats the ``?url=`` deep link. Auto-opens;
        if the browser blocks the popup, the page shows a one-click button.
        """
        import functools
        import http.server
        import socketserver
        import threading
        import webbrowser

        directory = str(trace_path.parent)
        trace_name = trace_path.name
        loaded = threading.Event()
        origin = "https://ui.perfetto.dev"

        open_html = (
            "<!doctype html><meta charset='utf-8'>"
            "<title>LibreYOLO profile</title>"
            "<style>body{font-family:Segoe UI,sans-serif;background:#0f1115;color:#e6e6e6;"
            "padding:40px;text-align:center}button{font-size:16px;padding:12px 22px;border:0;"
            "border-radius:8px;background:#3498db;color:#fff;cursor:pointer;margin-top:18px}"
            "#s{color:#8b95a5;margin-top:10px}</style>"
            "<h2>LibreYOLO training profile</h2><div id='s'>loading trace...</div>"
            "<button id='b' style='display:none' onclick='go()'>&#9654; Open timeline in Perfetto</button>"
            "<script>var O='" + origin + "';var BUF=null;"
            "fetch('./" + trace_name + "').then(function(r){return r.arrayBuffer();})"
            ".then(function(b){BUF=b;fetch('/__loaded').catch(function(){});"
            "document.getElementById('s').textContent='trace ready ('+(b.byteLength/1e6).toFixed(1)+' MB)';go();});"
            "function go(){var w=window.open(O);"
            "if(!w){document.getElementById('b').style.display='inline-block';"
            "document.getElementById('s').textContent='click the button to open the timeline';return;}"
            "var t=setInterval(function(){w.postMessage('PING',O);},64);"
            "function h(e){if(e.data!=='PONG')return;clearInterval(t);"
            "window.removeEventListener('message',h);"
            "w.postMessage({perfetto:{buffer:BUF,title:'LibreYOLO training profile',"
            "fileName:'" + trace_name + "'}},O);}"
            "window.addEventListener('message',h);}</script>"
        )
        try:
            (trace_path.parent / "open.html").write_text(open_html, encoding="utf-8")
        except Exception:
            return

        class Handler(http.server.SimpleHTTPRequestHandler):
            def end_headers(self):
                self.send_header("Cache-Control", "no-store")
                super().end_headers()

            def do_GET(self):
                if "__loaded" in self.path:
                    loaded.set()
                    self.send_response(204)
                    self.end_headers()
                    return
                super().do_GET()

            def log_message(self, *args):
                pass

        handler = functools.partial(Handler, directory=directory)
        try:
            httpd = socketserver.TCPServer(("127.0.0.1", 0), handler)
        except Exception:
            self._emit(f"  (could not start local server — drag {trace_name} "
                       "into https://ui.perfetto.dev)")
            return
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{port}/open.html"
        self._emit(f"  opening the GPU/CPU timeline in Perfetto…  {url}")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        loaded.wait(timeout)
        if loaded.is_set():
            self._emit("  trace handed to Perfetto (timeline should be open).")
            time.sleep(5.0)  # grace for the popup handshake / a manual click
        else:
            self._emit(f"  (page didn't load within {int(timeout)}s — open {url} "
                       f"or drag {trace_name} into https://ui.perfetto.dev)")
        try:
            httpd.shutdown()
        except Exception:
            pass

    def _write_html(self, trace_path):
        """Write a self-contained HTML report; returns its path (or None)."""
        if self.save_dir is None or self.summary is None:
            return None
        s = self.summary
        r = s["real"]
        m = self.meta
        comp_total = s["composition_total_ms"] or 1.0
        idle = r["gpu_idle_fraction"] * 100.0
        if s["bound"] == "dataloader":
            banner_bg = "#c0392b"
            verdict = f"DATALOADER-BOUND — GPU idle ~{idle:.0f}% (waiting on data)"
        else:
            banner_bg = "#1e8449"
            verdict = f"COMPUTE-BOUND — GPU idle only ~{idle:.0f}% (healthy)"

        palette = {
            "to_device": "#f1c40f", "forward": "#3498db",
            "backward": "#9b59b6", "optimizer": "#2ecc71",
        }
        rows = []
        for p in PHASES:
            ms = s["composition_ms"][p]
            pct = ms / comp_total * 100.0
            rows.append(
                f'<tr><td class="ph">{p}</td><td class="ms">{ms:.2f} ms</td>'
                f'<td class="barcell"><div class="bar" style="width:{pct:.1f}%;'
                f'background:{palette[p]}"></div></td>'
                f'<td class="pct">{pct:.1f}%</td></tr>'
            )

        # Real split: dataload vs compute, as one stacked bar.
        step = r["step_ms"] or 1.0
        dl_pct = r["dataload_ms"] / step * 100.0
        cp_pct = r["compute_ms"] / step * 100.0
        trace_html = ""
        if trace_path is not None:
            trace_html = (
                '<p class="meta">Full async timeline: drag '
                f'<code>{Path(trace_path).name}</code> into '
                '<a href="https://ui.perfetto.dev" target="_blank">ui.perfetto.dev</a>.</p>'
            )
        css = (
            "body{font-family:Segoe UI,Roboto,-apple-system,sans-serif;"
            "background:#0f1115;color:#e6e6e6;margin:0;padding:32px}"
            ".card{max-width:780px;margin:auto;background:#171a21;"
            "border:1px solid #262b36;border-radius:12px;padding:24px 28px}"
            "h1{font-size:19px;margin:0 0 4px}h2{font-size:14px;color:#cbd3df;margin:20px 0 8px}"
            ".meta{color:#8b95a5;font-size:13px;margin:6px 0}"
            ".banner{color:#fff;font-weight:600;padding:10px 14px;"
            "border-radius:8px;margin:14px 0 8px;background:%s}"
            ".big{font-size:15px;margin:10px 0}"
            ".stack{display:flex;height:22px;border-radius:5px;overflow:hidden;margin:8px 0}"
            ".seg{height:22px;display:flex;align-items:center;justify-content:center;"
            "font-size:11px;color:#0b0d12;font-weight:600}"
            "table{width:100%%;border-collapse:collapse}"
            "td{padding:6px 8px;font-size:14px;vertical-align:middle}"
            ".ph{width:88px;color:#cbd3df}"
            ".ms{width:88px;text-align:right;font-variant-numeric:tabular-nums}"
            ".pct{width:52px;text-align:right;color:#8b95a5}"
            ".bar{height:16px;border-radius:4px;min-width:2px}"
            "code{color:#f1c40f}a{color:#4aa3ff}"
        ) % banner_bg
        html = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>LibreYOLO profile — {m.get('model','?')}</title>"
            f"<style>{css}</style></head><body><div class='card'>"
            f"<h1>LibreYOLO training profile — {m.get('model','?')}</h1>"
            f"<div class='meta'>device {m.get('device','?')} · batch {m.get('batch','?')}"
            f" · imgsz {m.get('imgsz','?')} · amp {m.get('amp','?')}"
            f" · workers {m.get('workers','?')} · "
            f"{s['window']['real_steps']}+{s['window']['synced_steps']} steps</div>"
            f"<div class='banner'>{verdict}</div>"
            f"<div class='big'>Real step <b>{r['step_ms']:.0f} ms</b> &rarr; "
            f"<b>{r['img_per_s']:.1f} img/s</b></div>"
            "<div class='stack'>"
            f"<div class='seg' style='width:{dl_pct:.1f}%;background:#e67e22'>dataload {r['dataload_ms']:.0f}ms</div>"
            f"<div class='seg' style='width:{cp_pct:.1f}%;background:#3498db'>compute {r['compute_ms']:.0f}ms</div>"
            "</div>"
            "<h2>GPU compute composition (synchronized)</h2>"
            f"<table>{''.join(rows)}</table>"
            f"{trace_html}</div></body></html>"
        )
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path = self.save_dir / "profile_report.html"
            path.write_text(html, encoding="utf-8")
            return path
        except Exception:
            return None


class _PhaseTimer:
    """Labels the trace via record_function; sync-times only in the synced half."""

    __slots__ = ("prof", "name", "_t0", "_rf")

    def __init__(self, prof: TrainStepProfiler, name: str) -> None:
        self.prof = prof
        self.name = name
        self._t0 = None
        self._rf = None

    def __enter__(self):
        self._rf = record_function("step/" + self.name)
        self._rf.__enter__()
        if self.prof._synced_phase():
            self.prof._sync()
            self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self._t0 is not None:
            self.prof._sync()
            self.prof._sums[self.name] += (time.perf_counter() - self._t0) * 1000.0
        if self._rf is not None:
            self._rf.__exit__(*exc)
        return False


_TIMELINE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<title>LibreYOLO profiler - timeline</title>
<style>
 html,body{margin:0;background:#0f1115;color:#e6e6e6;font-family:Segoe UI,Roboto,sans-serif;overflow:hidden}
 #hdr{padding:7px 12px;font-size:13px;color:#cbd3df;border-bottom:1px solid #20242c;white-space:nowrap;overflow:hidden}
 #hdr b{color:#fff}
 .sw{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 9px;vertical-align:middle}
 #tip{position:fixed;pointer-events:none;background:rgba(0,0,0,.88);border:1px solid #333;
      padding:4px 8px;font-size:12px;border-radius:5px;display:none;white-space:nowrap;z-index:9;max-width:60vw;overflow:hidden}
 canvas{display:block;cursor:grab}
</style></head><body>
<div id="hdr"></div><canvas id="c"></canvas><div id="tip"></div>
<script>
var DATA = /*__DATA__*/;
var cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip'),
    hdr=document.getElementById('hdr');
var LEFT=128,TOP=28,ROW=12,GAP=12;
var LANES=['phase · CPU','CPU ops','phase · GPU','GPU kernels','GPU memcpy'];
var PH={dataload:'#e67e22',to_device:'#f1c40f',forward:'#3498db',backward:'#9b59b6',optimizer:'#2ecc71'};
function col(e){if(e.cat==='pcpu'||e.cat==='pgpu')return PH[e.name]||'#7f8c8d';
 if(e.cat==='cpu')return '#5d6d7e';if(e.cat==='gpu')return '#16a085';return '#e74c3c';}
var laneY=[],yy=TOP+8;
for(var i=0;i<5;i++){laneY[i]=yy;yy+=(DATA.lane_rows[i]||1)*ROW+GAP;}
var pxPerMs=1,viewStart=0,W=0,H=0,fitted=false;
function xOf(t){return LEFT+(t-viewStart)*pxPerMs;}
function fit(){pxPerMs=(W-LEFT-20)/(DATA.total_ms||1);viewStart=0;}
function resize(){W=cv.width=window.innerWidth;H=cv.height=window.innerHeight-hdr.offsetHeight;
 if(!fitted){fit();fitted=true;}draw();}
function niceStep(span){var raw=span/10,p=Math.pow(10,Math.floor(Math.log10(raw||1))),n=raw/p;
 n=n<1.5?1:n<3?2:n<7?5:10;return Math.max(n*p,0.001);}
function draw(){
 ctx.clearRect(0,0,W,H);
 ctx.strokeStyle='rgba(130,170,210,.08)';ctx.lineWidth=1;ctx.beginPath();
 var cy=laneY[1]+(DATA.lane_rows[1]||1)*ROW,gy=laneY[3];
 for(var f=0;f<DATA.flows.length;f++){var a=DATA.flows[f],xa=xOf(a[0]),xb=xOf(a[1]);
  if((xa<LEFT&&xb<LEFT)||(xa>W&&xb>W))continue;ctx.moveTo(xa,cy);ctx.lineTo(xb,gy);}
 ctx.stroke();
 for(var k=0;k<DATA.events.length;k++){var e=DATA.events[k],ex=xOf(e.t),ew=Math.max(e.d*pxPerMs,1);
  if(ex+ew<LEFT||ex>W)continue;var ey=laneY[e.lane]+e.row*ROW;
  ctx.fillStyle=col(e);ctx.fillRect(ex<LEFT?LEFT:ex,ey,ew,ROW-1);}
 ctx.fillStyle='#0f1115';ctx.fillRect(0,0,LEFT,H);
 ctx.fillStyle='#cbd3df';ctx.font='11px Segoe UI';ctx.textBaseline='top';
 for(var i=0;i<5;i++)ctx.fillText(LANES[i],8,laneY[i]);
 ctx.fillStyle='#0f1115';ctx.fillRect(0,0,W,TOP);
 ctx.strokeStyle='#222';ctx.beginPath();ctx.moveTo(0,TOP);ctx.lineTo(W,TOP);ctx.stroke();
 var span=(W-LEFT)/pxPerMs,st=niceStep(span),t0=Math.floor(viewStart/st)*st;
 ctx.font='10px Segoe UI';
 for(var t=t0;xOf(t)<W;t+=st){var px=xOf(t);if(px<LEFT)continue;
  ctx.strokeStyle='#171a21';ctx.beginPath();ctx.moveTo(px,TOP);ctx.lineTo(px,H);ctx.stroke();
  ctx.fillStyle='#8b95a5';ctx.fillText((st<1?t.toFixed(2):t.toFixed(0))+' ms',px+3,9);}
}
var drag=false,lx=0;
cv.addEventListener('mousedown',function(e){drag=true;lx=e.clientX;cv.style.cursor='grabbing';});
window.addEventListener('mouseup',function(){drag=false;cv.style.cursor='grab';});
window.addEventListener('mousemove',function(e){
 if(drag){viewStart-=(e.clientX-lx)/pxPerMs;lx=e.clientX;draw();return;}
 var r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top,found=null;
 if(mx>LEFT)for(var k=0;k<DATA.events.length;k++){var ev=DATA.events[k],ex=xOf(ev.t),
  ew=Math.max(ev.d*pxPerMs,1),ey=laneY[ev.lane]+ev.row*ROW;
  if(mx>=ex&&mx<=ex+ew&&my>=ey&&my<=ey+ROW)found=ev;}
 if(found){tip.style.display='block';tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY+14)+'px';
  tip.innerHTML='<b>'+found.name+'</b> &mdash; '+found.d.toFixed(found.d<1?3:2)+' ms';}
 else tip.style.display='none';
});
cv.addEventListener('wheel',function(e){e.preventDefault();var r=cv.getBoundingClientRect(),
 mx=e.clientX-r.left,mt=viewStart+(mx-LEFT)/pxPerMs,fac=e.deltaY<0?1.2:1/1.2;
 pxPerMs*=fac;viewStart=mt-(mx-LEFT)/pxPerMs;draw();},{passive:false});
hdr.innerHTML='LibreYOLO profiler &mdash; <b>'+(DATA.meta.model||'')+'</b> &middot; batch '+
 (DATA.meta.batch||'?')+' &middot; imgsz '+(DATA.meta.imgsz||'?')+' &middot; '+DATA.events.length+' events'+
 '<span style="color:#8b95a5;font-size:11px">'+
 '<i class=sw style="background:#3498db"></i>forward<i class=sw style="background:#9b59b6"></i>backward'+
 '<i class=sw style="background:#e67e22"></i>dataload<i class=sw style="background:#16a085"></i>kernel'+
 '<i class=sw style="background:#e74c3c"></i>memcpy &middot; scroll=zoom drag=pan</span>';
window.addEventListener('resize',resize);resize();
</script></body></html>"""
