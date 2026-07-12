"""Regression tests for collision-safe inference artifact allocation."""

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from libreyolo.utils import general
from libreyolo.utils.general import (
    log_saved_result,
    release_save_path_reservation,
    resolve_save_path,
)


pytestmark = pytest.mark.unit


def _reserve_publish_worker(output_path, payload, barrier, result_queue):
    """Reserve the shared explicit path, synchronize, then publish a payload."""
    try:
        reserved = resolve_save_path(output_path, "source.jpg")
        barrier.wait(timeout=30)
        Path(reserved).write_text(payload, encoding="utf-8")
        released = release_save_path_reservation(reserved)
        result_queue.put((str(reserved), payload, released, None))
    except BaseException as exc:
        result_queue.put((None, payload, False, repr(exc)))


def _hold_reservation_worker(output_path, ready_queue, release_event):
    """Hold a reservation so another process can attempt to release it."""
    try:
        reserved = resolve_save_path(output_path, "source.jpg")
        ready_queue.put((str(reserved), None))
        if not release_event.wait(timeout=30):
            raise TimeoutError("reservation release event was not set")
        Path(reserved).write_text("owner", encoding="utf-8")
        release_save_path_reservation(reserved)
    except BaseException as exc:
        ready_queue.put((None, repr(exc)))


def test_duplicate_source_paths_receive_distinct_names(tmp_path):
    output_dir = tmp_path / "results"
    source = tmp_path / "input" / "photo.jpg"

    first = resolve_save_path(output_dir, source)
    second = resolve_save_path(output_dir, source)

    assert first == output_dir / "photo.jpg"
    assert second == output_dir / "photo2.jpg"


def test_duplicate_basenames_from_different_directories_do_not_collide(tmp_path):
    output_dir = tmp_path / "results"

    first = resolve_save_path(output_dir, tmp_path / "a" / "photo.jpg")
    second = resolve_save_path(output_dir, tmp_path / "b" / "photo.jpg")

    assert first != second
    assert {first.name, second.name} == {"photo.jpg", "photo2.jpg"}


def test_explicit_file_is_incremented_for_multi_image_output(tmp_path):
    explicit = tmp_path / "prediction.jpg"

    first = resolve_save_path(explicit, "first.jpg")
    second = resolve_save_path(explicit, "second.jpg")

    assert first == explicit
    assert second == tmp_path / "prediction2.jpg"


def test_existing_suffix_bearing_directory_is_treated_as_directory(tmp_path):
    output_dir = tmp_path / "results.v1"
    output_dir.mkdir()

    resolved = resolve_save_path(output_dir, "photo.jpg")

    assert resolved.parent == output_dir
    assert resolved.name == "photo.jpg"


def test_existing_artifact_is_never_reused(tmp_path):
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "photo.jpg").write_bytes(b"original")

    resolved = resolve_save_path(output_dir, "photo.jpg")

    assert resolved == output_dir / "photo2.jpg"
    assert (output_dir / "photo.jpg").read_bytes() == b"original"


def test_force_ext_rewrites_explicit_file_without_reusing_existing_png(tmp_path):
    (tmp_path / "cutout.png").write_bytes(b"original")

    resolved = resolve_save_path(
        tmp_path / "cutout.jpg",
        "photo.jpg",
        ext="png",
        force_ext=True,
    )

    assert resolved == tmp_path / "cutout2.png"
    assert (tmp_path / "cutout.png").read_bytes() == b"original"


def test_concurrent_allocations_are_unique(tmp_path):
    output_dir = tmp_path / "results"

    with ThreadPoolExecutor(max_workers=8) as pool:
        paths = list(
            pool.map(
                lambda _: resolve_save_path(output_dir, "same.jpg"),
                range(16),
            )
        )

    assert len(paths) == len(set(paths)) == 16


def test_multiprocess_explicit_file_allocations_are_unique(tmp_path):
    context = multiprocessing.get_context("spawn")
    explicit = str(tmp_path / "prediction.jpg")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_reserve_publish_worker,
            args=(explicit, payload, barrier, result_queue),
        )
        for payload in ("first", "second")
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)

    assert all(not process.is_alive() for process in processes)
    assert [process.exitcode for process in processes] == [0, 0]
    reports = [result_queue.get(timeout=5) for _ in processes]
    assert [report[3] for report in reports] == [None, None]
    assert all(report[2] for report in reports)

    paths = [Path(report[0]) for report in reports]
    assert len(set(paths)) == 2
    assert {path.name for path in paths} == {"prediction.jpg", "prediction2.jpg"}
    assert {path.read_text(encoding="utf-8") for path in paths} == {
        "first",
        "second",
    }


def test_process_cannot_release_another_process_reservation(tmp_path):
    context = multiprocessing.get_context("spawn")
    explicit = tmp_path / "prediction.jpg"
    ready_queue = context.Queue()
    release_event = context.Event()
    owner = context.Process(
        target=_hold_reservation_worker,
        args=(str(explicit), ready_queue, release_event),
    )
    owner.start()

    try:
        reserved, error = ready_queue.get(timeout=30)
        assert error is None
        assert Path(reserved) == explicit
        assert release_save_path_reservation(reserved) is False

        contender = resolve_save_path(explicit, "source.jpg")
        assert contender == tmp_path / "prediction2.jpg"
        assert release_save_path_reservation(contender) is True
    finally:
        release_event.set()
        owner.join(timeout=60)

    assert not owner.is_alive()
    assert owner.exitcode == 0
    assert explicit.read_text(encoding="utf-8") == "owner"


def test_release_never_deletes_published_artifact(tmp_path):
    reserved = resolve_save_path(tmp_path / "prediction.jpg", "source.jpg")
    reserved.write_bytes(b"published")

    assert release_save_path_reservation(reserved) is True
    assert reserved.read_bytes() == b"published"


def test_logging_success_releases_process_reservation(tmp_path):
    explicit = tmp_path / "prediction.jpg"
    reserved = resolve_save_path(explicit, "source.jpg")
    reserved.write_bytes(b"published")
    result = SimpleNamespace()

    log_saved_result(result, reserved)
    reserved.unlink()
    reused = resolve_save_path(explicit, "source.jpg")

    assert result.saved_path == str(explicit)
    assert reused == explicit
    assert release_save_path_reservation(reused) is True


def test_save_path_allocation_has_no_9999_name_ceiling(tmp_path, monkeypatch):
    original_lexists = os.path.lexists

    def first_9999_names_exist(path):
        candidate = Path(path)
        if candidate.parent == tmp_path and candidate.suffix == ".jpg":
            index_text = candidate.stem.removeprefix("prediction")
            index = int(index_text) if index_text else 1
            if index <= 9999:
                return True
        return original_lexists(path)

    monkeypatch.setattr(general.os.path, "lexists", first_9999_names_exist)

    reserved = resolve_save_path(tmp_path / "prediction.jpg", "source.jpg")

    assert reserved.name == "prediction10000.jpg"
    assert release_save_path_reservation(reserved) is True


def test_directory_aliases_reserve_distinct_physical_artifacts(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias_dir = tmp_path / "alias"
    try:
        alias_dir.symlink_to(real_dir, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    first = resolve_save_path(real_dir, "photo.jpg")
    second = resolve_save_path(alias_dir, "photo.jpg")

    assert first.resolve(strict=False) != second.resolve(strict=False)
    assert first.name == "photo.jpg"
    assert second.name == "photo2.jpg"


@pytest.mark.skipif(os.name != "nt", reason="Windows paths are case-insensitive")
def test_windows_equivalent_case_variants_do_not_share_an_artifact(tmp_path):
    output_dir = tmp_path / "results"

    first = resolve_save_path(output_dir, "photo.jpg")
    second = resolve_save_path(output_dir, "PHOTO.jpg")

    assert os.path.normcase(os.path.abspath(first)) != os.path.normcase(
        os.path.abspath(second)
    )
    assert first.name == "photo.jpg"
    assert second.name == "PHOTO2.jpg"
