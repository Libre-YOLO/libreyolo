"""Regression tests for collision-safe inference artifact allocation."""

from concurrent.futures import ThreadPoolExecutor
import os

import pytest

from libreyolo.utils.general import resolve_save_path


pytestmark = pytest.mark.unit


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
