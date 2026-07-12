"""Output routing for the LibreYOLO CLI.

stdout is the API (results only). stderr is for humans (progress, logs).
"""

import json
import logging
import sys
import warnings
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from io import StringIO
from typing import Any

from pathlib import Path

from .errors import CLIError

logger = logging.getLogger(__name__)


def _json_default(obj: Any) -> Any:
    """Strict JSON default: only allow Path → str. Everything else is an error."""
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


class OutputHandler:
    """Routes output to stdout (results) and stderr (progress/errors)."""

    def __init__(self, *, json_mode: bool = False, quiet: bool = False) -> None:
        self.json_mode = json_mode
        self.quiet = quiet
        self.is_tty = sys.stdout.isatty()

    def result(self, data: dict[str, Any]) -> None:
        """Write result to stdout. In JSON mode, adds schema_version."""
        if self.json_mode:
            public_data = {
                key: value for key, value in data.items() if not key.startswith("_")
            }
            public_data["schema_version"] = 1
            print(json.dumps(public_data, default=_json_default))
        else:
            self._print_human(data)

    def progress(self, message: str) -> None:
        """Write progress info to stderr via logger. Respects --quiet."""
        if self.quiet:
            return
        logger.info(message)

    def warning(self, message: str) -> None:
        """Write warnings to stderr."""
        if self.quiet:
            return
        logger.warning(message)

    @contextmanager
    def library_output(self):
        """Keep library/third-party prints off the CLI result stream.

        Direct ``print`` calls are routed to stderr for normal commands and
        discarded under ``--quiet``. This preserves stdout as one machine-
        readable JSON document in ``--json --quiet`` mode.
        """
        if not self.quiet:
            with redirect_stdout(sys.stderr):
                yield
            return

        # Quiet mode is a strict stderr boundary. Some libraries write
        # directly, some issue warnings, and logging handlers may retain the
        # original stderr stream instead of consulting redirect_stderr().
        sink = StringIO()
        previous_disable = logging.root.manager.disable
        with (
            redirect_stdout(sink),
            redirect_stderr(sink),
            warnings.catch_warnings(),
        ):
            warnings.simplefilter("ignore")
            logging.disable(logging.CRITICAL)
            try:
                yield
            finally:
                logging.disable(previous_disable)

    def error(self, err: CLIError) -> None:
        """Write error. With --json: JSON to stdout. Without: log to stderr."""
        if self.json_mode:
            payload = {
                "schema_version": 1,
                "error": err.code,
                "message": err.message,
                "suggestion": err.suggestion,
            }
            for key, value in err.context.items():
                if key not in payload:
                    payload[key] = value
            print(json.dumps(payload, default=_json_default))
        else:
            logger.error("Error [%s]: %s", err.code, err.message)
            if err.suggestion:
                logger.info("  Suggestion: %s", err.suggestion)

    def _print_human(self, data: dict[str, Any]) -> None:
        """Format data as human-readable text to stdout."""
        if "_human_text" in data:
            print(data["_human_text"])
        else:
            for key, value in data.items():
                if not key.startswith("_"):
                    print(f"  {key}: {value}")
