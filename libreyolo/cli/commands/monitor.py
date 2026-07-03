"""Monitor command: watch training runs live in the browser.

A single server watches a whole *runs root* (default ``runs/``) and addresses
each run by URL, so multiple runs on one machine share one port. Open a tab
per run, or use the index at ``/`` to see them all. Reads the on-disk artifacts
every ``libreyolo train`` writes (``status.json`` / ``metrics.jsonl`` /
``train.log`` / images); it never touches the training processes, so it works
on live, finished, or crashed runs.
"""

import errno
from pathlib import Path

import typer

from ..command_utils import exit_with_error
from ..output import OutputHandler


def monitor_cmd(
    run_dir: str = typer.Argument(
        None,
        help=(
            "A runs root to watch (default: runs/), or a single run directory "
            "to open directly. Either way every run under the root is listed."
        ),
    ),
    host: str = typer.Option("127.0.0.1", help="Host/interface to bind"),
    port: int = typer.Option(8420, help="Port to bind (auto-bumps if taken)"),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Do not auto-open the browser"
    ),
    json_output: bool = typer.Option(False, "--json", help="JSON output to stdout"),
    quiet: bool = typer.Option(False, "--quiet", help="Suppress stderr"),
    verbose: bool = typer.Option(False, "--verbose", help="Verbose stderr output"),
) -> None:
    """Serve a live web dashboard for training runs (charts, log, images)."""
    out = OutputHandler(json_mode=json_output, quiet=quiet)

    from libreyolo.ui.train_monitor import discover_runs, is_run_dir

    # Resolve what to watch. Pointing at a single run roots the server at its
    # parent (so sibling runs still show in the index) and deep-links to it.
    open_run = None
    if run_dir:
        target = Path(run_dir)
        if not target.exists():
            exit_with_error(
                out,
                "source_not_found",
                f"Path not found: {target}",
                suggestion="Pass an existing run dir or runs root, or omit it for runs/.",
            )
        if is_run_dir(target):
            open_run = target
            root = target.parent
        else:
            root = target
    else:
        root = Path("runs")
        if not root.exists():
            exit_with_error(
                out,
                "source_not_found",
                "No runs/ directory found.",
                suggestion="Start a training run, or pass a runs root explicitly.",
            )

    runs = discover_runs(root)
    if not runs:
        exit_with_error(
            out,
            "source_not_found",
            f"No training runs found under {root}.",
            suggestion="Start a training run, or point at a directory that contains runs.",
        )

    from libreyolo.ui.train_monitor import serve

    httpd = None
    url = ""
    for candidate in range(port, port + 20):
        try:
            httpd, url = serve(
                root,
                host=host,
                port=candidate,
                open_browser=not no_browser,
                open_run=open_run,
            )
            break
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, getattr(errno, "WSAEADDRINUSE", 10048)):
                continue
            exit_with_error(out, "io_error", f"Failed to start monitor server: {exc}")

    if httpd is None:
        exit_with_error(
            out,
            "io_error",
            f"No free port found in range {port}-{port + 19}.",
            suggestion="Pass a different --port.",
        )

    out.result(
        {
            "url": url,
            "root": str(root),
            "runs": len(runs),
            "_human_text": (
                f"LibreYOLO training monitor running at {url}\n"
                f"Watching {len(runs)} run(s) under: {root}\n"
                "Open the URL for the run index, or a tab per run "
                "(the ?run= in each URL identifies it). Press Ctrl+C to stop."
            ),
        }
    )

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        out.progress("Shutting down monitor...")
    finally:
        httpd.shutdown()
        httpd.server_close()
