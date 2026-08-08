"""Repository-level checks for capability language and status markers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.unit


_REPO_ROOT = Path(__file__).resolve().parents[2]
_FORBIDDEN_WORDS = ("experi" + "mental", "\u5b9e\u9a8c" + "\u6027")
_SHORT_STATUS = "ex" + "p"
_FORBIDDEN_STATUS_MARKERS = (
    f"`{_SHORT_STATUS}` means",
    f"<td>{_SHORT_STATUS}</td>",
    f"| {_SHORT_STATUS} |",
)


def _tracked_paths() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        Path(raw.decode("utf-8"))
        for raw in result.stdout.split(b"\0")
        if raw
    ]


def _tracked_text_matches(patterns: tuple[str, ...]) -> list[str]:
    command = ["git", "grep", "-n", "-I", "-i", "-F"]
    for pattern in patterns:
        command.extend(("-e", pattern))
    command.extend(("--", "."))
    result = subprocess.run(
        command,
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if result.returncode not in (0, 1):
        raise subprocess.CalledProcessError(
            result.returncode,
            command,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout.splitlines()


def test_tracked_tree_has_no_deprecated_capability_vocabulary() -> None:
    """Keep capability access separate from validation depth."""
    forbidden_words = tuple(word.casefold() for word in _FORBIDDEN_WORDS)
    violations = [
        str(path)
        for path in _tracked_paths()
        if any(word in path.as_posix().casefold() for word in forbidden_words)
    ]
    violations.extend(
        _tracked_text_matches(_FORBIDDEN_WORDS + _FORBIDDEN_STATUS_MARKERS)
    )

    assert not violations, "Forbidden capability labels found:\n" + "\n".join(
        violations
    )
