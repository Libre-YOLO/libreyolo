"""Label command: launch the local bounding-box annotation tool in the browser."""

import errno

import typer

from ..command_utils import exit_with_error
from ..output import OutputHandler


def label_cmd(
    data: str = typer.Option(
        ..., help="Dataset YAML (or a folder containing one) to label"
    ),
    host: str = typer.Option("127.0.0.1", help="Host/interface to bind"),
    port: int = typer.Option(8000, help="Port to bind (auto-bumps if taken)"),
    device: str = typer.Option("auto", help="Device for AI auto-label: 0, cpu, mps, auto"),
    no_assist: bool = typer.Option(
        False, "--no-assist", help="Disable AI auto-label (pure manual labeller)"
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not auto-open the browser"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose stderr output"),
) -> None:
    """Launch a local web tool to draw bounding boxes and save YOLO labels.

    Opens an existing dataset (``data=path/to/data.yaml``), lets you draw/edit
    boxes in the browser, and writes LibreYOLO-native ``labels/*.txt`` files so
    the dataset trains with no conversion. v1 handles bounding boxes only.
    """
    out = OutputHandler(json_mode=json_output, quiet=quiet)

    from libreyolo.label.server import serve

    httpd = None
    url = ""
    session = None
    for candidate in range(port, port + 20):
        try:
            httpd, url, session = serve(
                data=data,
                host=host,
                port=candidate,
                open_browser=not no_browser,
                device=device,
                assist=not no_assist,
            )
            break
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)):
                continue
            exit_with_error(out, "io_error", f"Failed to start label server: {exc}")
        except Exception as exc:  # noqa: BLE001 - dataset/config problems
            exit_with_error(out, "io_error", f"Failed to open dataset: {exc}")

    if httpd is None or session is None:
        exit_with_error(
            out,
            "io_error",
            f"No free port found in range {port}-{port + 19}.",
            suggestion="Pass a different port=.",
        )

    meta = session.meta()
    warn = "" if meta["writable"] else f"\n  WARNING (read-only): {meta['reason']}"
    out.result(
        {
            "url": url,
            "images": meta["count"],
            "classes": meta["nc"],
            "writable": meta["writable"],
            "_human_text": (
                f"LibreLabel running at {url}\n"
                f"  {meta['count']} images, {meta['nc']} classes. "
                "Drag to draw a box, number keys set the class, A/D move between "
                "images (auto-saves). Press ? for shortcuts, Ctrl+C to stop." + warn
            ),
        }
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out.progress("Shutting down LibreLabel...")
    finally:
        httpd.shutdown()
        httpd.server_close()
