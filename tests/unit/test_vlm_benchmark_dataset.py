"""Offline tests for the local-only VLM benchmark dataset manifest."""

from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from PIL import Image

from libreyolo.validation import vlm_benchmark_dataset as dataset_manifest

pytestmark = pytest.mark.unit


@dataclass(frozen=True)
class _LocalCOCO:
    source: Path
    images: Path
    contract: dataset_manifest._SourceContract


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_source(path: Path, source: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


_REVIEW_CHECKS = (
    "canonical_source",
    "image_attribution_sufficiency",
    "annotation_license_and_redistribution",
    "privacy_and_pii",
    "visual_quality",
    "selection_salt_freeze",
    "benchmark_suitability",
    "publication_upload_authorization",
)


def _review_payload(manifest_sha256: str) -> dict:
    return {
        "schema": "libreyolo.vlm-benchmark-dataset-review.v1",
        "manifest_sha256": manifest_sha256,
        "partition_role": "zero_shot_confidence_promotion",
        "status": "approved",
        "reviewer": "Local test reviewer",
        "reviewed_at": "2026-08-16T10:30:00Z",
        "checks": {check: True for check in _REVIEW_CHECKS},
    }


def _write_review(path: Path, payload: dict) -> bytes:
    encoded = (
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        + b"\n"
    )
    path.write_bytes(encoded)
    return encoded


def test_production_source_contract_pins_official_selected_image_bytes():
    contract = dataset_manifest._SOURCE_CONTRACT
    assert contract.image_archive_sha256 == (
        "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05"
    )
    assert contract.image_archive_size_bytes == 815_585_330
    assert contract.selected_image_identity_sha256 == (
        "73e35dbb1ce5058953bccbc99ab15db46474f36cc160046cbcac71350662d29c"
    )
    assert contract.selected_image_identity_size_bytes == 73_312
    assert contract.selected_image_bytes_total == 81_833_238
    assert dict(contract.partition_unrepresented_category_ids) == {
        "holdout100": (89,),
        "train400": (21, 38, 87, 89),
        "promotion500": (89,),
    }


@pytest.fixture(scope="module")
def local_coco(tmp_path_factory) -> _LocalCOCO:
    root = tmp_path_factory.mktemp("vlm-coco-license4")
    images_dir = root / "val2017"
    images_dir.mkdir()
    encoded = io.BytesIO()
    Image.new("RGB", (2, 2), (17, 23, 31)).save(encoded, format="JPEG")
    image_bytes = encoded.getvalue()

    images = []
    annotations = []
    for image_id in range(1, 504):
        filename = f"{image_id:012}.jpg"
        (images_dir / filename).write_bytes(image_bytes)
        images.append(
            {
                "license": 1 if image_id == 501 else 4,
                "file_name": filename,
                "coco_url": f"http://images.example.test/val2017/{filename}",
                "height": 2,
                "width": 2,
                "date_captured": "2017-01-01 00:00:00",
                "flickr_url": f"http://source.example.test/image/{image_id}",
                "id": image_id,
            }
        )
        annotations.append(
            {
                "segmentation": [],
                "area": 1,
                "iscrowd": 1 if image_id == 502 else 0,
                "ignore": 1 if image_id == 503 else 0,
                "image_id": image_id,
                "bbox": [0, 0, 1, 1],
                "category_id": ((image_id - 1) % 3) + 1,
                "id": image_id,
            }
        )

    source = {
        "info": {
            "description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "Synthetic offline test",
            "date_created": "2017/09/01",
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            },
            {
                "url": "http://creativecommons.org/licenses/by/2.0/",
                "id": 4,
                "name": "Attribution License",
            },
        ],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "supercategory": "object",
                "id": category_id,
                "name": f"class-{category_id}",
            }
            for category_id in range(1, 4)
        ],
    }
    source_path = root / "instances_val2017.json"
    _write_source(source_path, source)
    canonical = dataset_manifest._normalized_source_bytes(source)
    contract = dataset_manifest._SourceContract(
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
        canonical_size_bytes=len(canonical),
        image_count=503,
        annotation_count=503,
        category_count=3,
        license_image_count=502,
        eligible_image_count=500,
        available_category_count=3,
        unavailable_category_ids=(),
        coverage_seed_image_count=3,
    )
    return _LocalCOCO(source=source_path, images=images_dir, contract=contract)


@pytest.fixture
def pinned_local_source(monkeypatch, local_coco):
    monkeypatch.setattr(dataset_manifest, "_SOURCE_CONTRACT", local_coco.contract)
    return local_coco


def test_build_and_verify_metadata_only_partitioned_bundle(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in source.images.iterdir()
    }
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )

    expected_files = {
        "manifest.json",
        "ATTRIBUTION.jsonl",
        "ANNOTATION_NOTICE.txt",
        "annotations/instances_val2017_holdout100.json",
        "annotations/instances_val2017_train400.json",
        "annotations/instances_val2017_promotion500.json",
    }
    assert set(_bundle_bytes(artifacts.output_dir)) == expected_files
    assert all(not path.is_symlink() for path in artifacts.output_dir.rglob("*"))
    assert all(
        path.suffix.lower() in {".json", ".jsonl", ".txt"}
        for path in artifacts.output_dir.rglob("*")
        if path.is_file()
    )
    assert before == {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in source.images.iterdir()
    }

    manifest = _json(artifacts.manifest_path)
    assert manifest["schema"] == "libreyolo.vlm-benchmark-dataset.v1"
    assert manifest["source"]["image_bytes_included"] is False
    identity_rows = [
        {
            "image_id": row["image_id"],
            "file_name": row["file_name"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in manifest["images"]
    ]
    identity_payload = dataset_manifest._canonical_json(identity_rows)
    assert manifest["source"]["selected_image_identity"] == {
        "canonicalization": "selected-image-id-name-size-sha256-json-v1",
        "canonical_size_bytes": len(identity_payload),
        "selected_image_bytes_total": sum(row["size_bytes"] for row in identity_rows),
        "sha256": hashlib.sha256(identity_payload).hexdigest(),
        "publisher_archive_member_pin_enforced": False,
    }
    assert manifest["license_gate"] == {
        "required_image_license_id": 4,
        "spdx": "CC-BY-2.0",
        "name": "Attribution License",
        "url": "http://creativecommons.org/licenses/by/2.0/",
    }
    assert manifest["manual_review"]["status"] == "required-outside-manifest"
    assert manifest["manual_review"]["checks"] == [
        "canonical_source",
        "image_attribution_sufficiency",
        "annotation_license_and_redistribution",
        "privacy_and_pii",
        "visual_quality",
        "selection_salt_freeze",
        "benchmark_suitability",
        "publication_upload_authorization",
    ]
    assert manifest["annotation_license"]["spdx"] == "CC-BY-4.0"
    assert manifest["annotation_license"]["artifacts_are_modified_derivatives"] is True
    assert manifest["annotation_license"]["derived_artifacts"] == [
        "manifest.json",
        "ATTRIBUTION.jsonl",
        "annotations/instances_val2017_holdout100.json",
        "annotations/instances_val2017_train400.json",
        "annotations/instances_val2017_promotion500.json",
    ]
    assert len(manifest["images"]) == 500
    selected_ids = [row["image_id"] for row in manifest["images"]]
    assert len(set(selected_ids)) == 500
    assert 501 not in selected_ids
    assert 502 not in selected_ids
    assert 503 not in selected_ids
    assert all(row["license_id"] == 4 for row in manifest["images"])
    assert [row["rank_index"] for row in manifest["images"]] == list(range(500))

    holdout = _json(
        artifacts.output_dir / "annotations/instances_val2017_holdout100.json"
    )
    train = _json(artifacts.output_dir / "annotations/instances_val2017_train400.json")
    promotion = _json(
        artifacts.output_dir / "annotations/instances_val2017_promotion500.json"
    )
    holdout_ids = [row["id"] for row in holdout["images"]]
    train_ids = [row["id"] for row in train["images"]]
    assert holdout_ids == selected_ids[:100]
    assert train_ids == selected_ids[100:]
    assert [row["id"] for row in promotion["images"]] == selected_ids
    assert set(holdout_ids).isdisjoint(train_ids)
    assert {row["category_id"] for row in holdout["annotations"]} == {1, 2, 3}
    assert all(row["iscrowd"] == 0 for row in promotion["annotations"])
    assert all(row["ignore"] == 0 for row in promotion["annotations"])
    assert {
        row["name"]: (
            row["represented_category_count"],
            row["represented_category_ids"],
            row["unrepresented_category_ids"],
        )
        for row in manifest["partitions"]
    } == {
        "holdout100": (3, [1, 2, 3], []),
        "train400": (3, [1, 2, 3], []),
        "promotion500": (3, [1, 2, 3], []),
    }

    attribution = [
        json.loads(line)
        for line in (artifacts.output_dir / "ATTRIBUTION.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(attribution) == 500
    assert [row["image_id"] for row in attribution] == selected_ids
    assert all(row["creator"] is None and row["title"] is None for row in attribution)
    assert all(row["creator_supplied_by_source"] is False for row in attribution)
    notice = (artifacts.output_dir / "ANNOTATION_NOTICE.txt").read_text(
        encoding="utf-8"
    )
    assert "COCO Consortium" in notice
    assert "CC-BY-4.0" in notice
    assert "selected, filtered, reordered, and summarized" in notice
    assert "annotations_trainval2017.zip" in notice

    verified = dataset_manifest.verify_benchmark_dataset(
        artifacts.manifest_path, source.source, source.images
    )
    assert verified.manifest_sha256 == artifacts.manifest_sha256
    assert verified.selected_image_count == 500


def test_build_is_independent_of_source_record_order(tmp_path, pinned_local_source):
    source = pinned_local_source
    original = _json(source.source)
    reordered = {
        key: list(reversed(value)) if isinstance(value, list) else value
        for key, value in reversed(list(original.items()))
    }
    reordered_path = tmp_path / "reordered" / "instances_val2017.json"
    _write_source(reordered_path, reordered)

    first = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "first"
    )
    second = dataset_manifest.build_benchmark_dataset(
        reordered_path, source.images, tmp_path / "second"
    )

    assert _bundle_bytes(first.output_dir) == _bundle_bytes(second.output_dir)
    assert first.manifest_sha256 == second.manifest_sha256


@pytest.mark.parametrize("mutation", ["wrong_license", "path_escape"])
def test_build_rejects_invalid_source_evidence_before_output(
    tmp_path, pinned_local_source, mutation
):
    source = pinned_local_source
    modified = _json(source.source)
    if mutation == "wrong_license":
        license_row = next(row for row in modified["licenses"] if row["id"] == 4)
        license_row["url"] = "http://creativecommons.org/licenses/by-nc/2.0/"
        match = "not the pinned CC-BY-2.0"
    else:
        modified["images"][0]["file_name"] = "../outside.jpg"
        match = "local JPEG basename"
    source_path = tmp_path / mutation / "instances_val2017.json"
    _write_source(source_path, modified)
    output = tmp_path / f"output-{mutation}"

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match=match):
        dataset_manifest.build_benchmark_dataset(source_path, source.images, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "payload,match",
    [
        (b'{"info":{},"info":{}}', "duplicate key"),
        (b'{"info":NaN}', "not permitted"),
    ],
)
def test_build_rejects_non_strict_json_before_output(
    tmp_path, pinned_local_source, payload, match
):
    source_path = tmp_path / "invalid" / "instances_val2017.json"
    source_path.parent.mkdir()
    source_path.write_bytes(payload)
    output = tmp_path / "output"

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match=match):
        dataset_manifest.build_benchmark_dataset(
            source_path, pinned_local_source.images, output
        )
    assert not output.exists()


def test_verify_rejects_modified_metadata_artifact(tmp_path, pinned_local_source):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    promotion = artifacts.output_dir / "annotations/instances_val2017_promotion500.json"
    promotion.write_bytes(promotion.read_bytes() + b" ")

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="safety limit"):
        dataset_manifest.verify_benchmark_dataset(
            artifacts.manifest_path, source.source, source.images
        )


def test_verify_rejects_changed_selected_image(tmp_path, pinned_local_source):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    manifest = _json(artifacts.manifest_path)
    selected = source.images / manifest["images"][0]["file_name"]
    original = selected.read_bytes()
    changed = io.BytesIO()
    Image.new("RGB", (2, 2), (200, 10, 20)).save(changed, format="JPEG")
    try:
        selected.write_bytes(changed.getvalue())
        with pytest.raises(
            dataset_manifest.BenchmarkDatasetError,
            match="manifest does not match reconstructed source evidence",
        ):
            dataset_manifest.verify_benchmark_dataset(
                artifacts.manifest_path, source.source, source.images
            )
    finally:
        selected.write_bytes(original)


def test_verify_rejects_duplicate_manifest_keys(tmp_path, pinned_local_source):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    text = artifacts.manifest_path.read_text(encoding="utf-8")
    artifacts.manifest_path.write_text(
        text.replace('"schema":', '"schema":"duplicate","schema":', 1),
        encoding="utf-8",
    )

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="duplicate key"):
        dataset_manifest.verify_benchmark_dataset(
            artifacts.manifest_path, source.source, source.images
        )


def test_build_refuses_overwrite_and_cleans_failed_stage(
    tmp_path, pinned_local_source, monkeypatch
):
    source = pinned_local_source
    existing = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "existing"
    )
    with pytest.raises(dataset_manifest.BenchmarkDatasetOutputExistsError):
        dataset_manifest.build_benchmark_dataset(
            source.source, source.images, existing.output_dir
        )

    output = tmp_path / "failed"

    def fail_write(path, payload):
        del path, payload
        raise OSError("injected write failure")

    monkeypatch.setattr(dataset_manifest, "_write_bytes", fail_write)
    with pytest.raises(OSError, match="injected write failure"):
        dataset_manifest.build_benchmark_dataset(source.source, source.images, output)
    assert not output.exists()
    assert not (tmp_path / ".failed.lock").exists()
    assert not list(tmp_path.glob(".failed.tmp-*"))


def test_build_rejects_output_inside_image_tree(tmp_path, pinned_local_source):
    source = pinned_local_source
    output = source.images / "metadata"

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="disjoint"):
        dataset_manifest.build_benchmark_dataset(source.source, source.images, output)
    assert not output.exists()


def test_build_bounds_selected_image_reads(tmp_path, pinned_local_source, monkeypatch):
    source = pinned_local_source
    image_size = next(source.images.iterdir()).stat().st_size
    monkeypatch.setattr(dataset_manifest, "_MAX_IMAGE_BYTES", image_size - 1)
    output = tmp_path / "output"

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="safety limit"):
        dataset_manifest.build_benchmark_dataset(source.source, source.images, output)
    assert not output.exists()


def test_build_rejects_selected_images_above_pillow_pixel_limit(
    tmp_path, pinned_local_source, monkeypatch
):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 1)
    output = tmp_path / "output"

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="pixel limit"):
        dataset_manifest.build_benchmark_dataset(
            pinned_local_source.source,
            pinned_local_source.images,
            output,
        )
    assert not output.exists()


def test_build_enforces_selected_image_archive_member_pin(
    tmp_path, pinned_local_source, monkeypatch
):
    source = pinned_local_source
    unpinned = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "unpinned"
    )
    identity = _json(unpinned.manifest_path)["source"]["selected_image_identity"]
    pinned_contract = replace(
        source.contract,
        selected_image_identity_sha256=identity["sha256"],
        selected_image_identity_size_bytes=identity["canonical_size_bytes"],
        selected_image_bytes_total=identity["selected_image_bytes_total"],
    )
    monkeypatch.setattr(
        dataset_manifest,
        "_SOURCE_CONTRACT",
        pinned_contract,
    )
    pinned = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "pinned"
    )
    assert (
        _json(pinned.manifest_path)["source"]["selected_image_identity"][
            "publisher_archive_member_pin_enforced"
        ]
        is True
    )

    output = tmp_path / "mismatch"
    selected_name = _json(pinned.manifest_path)["images"][0]["file_name"]
    selected_path = source.images / selected_name
    original = selected_path.read_bytes()
    changed = io.BytesIO()
    Image.new("RGB", (2, 2), (200, 10, 20)).save(changed, format="JPEG")
    try:
        selected_path.write_bytes(changed.getvalue())
        with pytest.raises(
            dataset_manifest.BenchmarkDatasetError,
            match="pinned official archive members",
        ):
            dataset_manifest.build_benchmark_dataset(
                source.source, source.images, output
            )
    finally:
        selected_path.write_bytes(original)
    assert not output.exists()


def test_verify_benchmark_run_inputs_binds_bundle_partition_and_review(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    review_path = tmp_path / "review.json"
    review_payload = _review_payload(artifacts.manifest_sha256)
    review_payload["reviewer"] = "  Local test reviewer  "
    review_bytes = _write_review(review_path, review_payload)

    verified = dataset_manifest.verify_benchmark_run_inputs(
        artifacts.manifest_path,
        source.source,
        source.images,
        review_path,
    )

    manifest = _json(artifacts.manifest_path)
    promotion = _json(
        artifacts.output_dir / "annotations/instances_val2017_promotion500.json"
    )
    assert verified.manifest_path == artifacts.manifest_path.resolve()
    assert verified.manifest_sha256 == artifacts.manifest_sha256
    assert verified.source_annotations == source.source.resolve()
    assert verified.source_canonical_sha256 == source.contract.canonical_sha256
    assert (
        verified.source_file_sha256
        == hashlib.sha256(source.source.read_bytes()).hexdigest()
    )
    assert verified.source_file_size_bytes == source.source.stat().st_size
    assert verified.images_dir == source.images.resolve()
    assert (
        verified.selected_image_identity_sha256
        == (manifest["source"]["selected_image_identity"]["sha256"])
    )
    assert (
        verified.partition_name,
        verified.partition_role,
        verified.partition_start,
        verified.partition_stop,
    ) == ("promotion500", "zero_shot_confidence_promotion", 0, 500)
    assert (
        verified.annotation_path
        == (
            artifacts.output_dir / "annotations/instances_val2017_promotion500.json"
        ).resolve()
    )
    assert verified.annotation_sha256 == manifest["artifacts"]["promotion500"]["sha256"]
    assert (
        verified.annotation_size_bytes
        == manifest["artifacts"]["promotion500"]["size_bytes"]
    )
    assert verified.class_names == ("class-1", "class-2", "class-3")
    assert [row["image_id"] for row in verified.expected_images] == [
        row["id"] for row in promotion["images"]
    ]
    assert all(
        "sha256" in row and "size_bytes" in row for row in verified.expected_images
    )
    assert [row["id"] for row in verified.expected_categories] == [1, 2, 3]
    assert [row["id"] for row in verified.expected_annotations] == [
        row["id"] for row in promotion["annotations"]
    ]
    assert verified.review_attestation_path == review_path.resolve()
    assert (
        verified.review_attestation_sha256 == hashlib.sha256(review_bytes).hexdigest()
    )
    assert verified.review_attestation["reviewer"] == "Local test reviewer"
    assert dict(verified.review_attestation["checks"]) == {
        check: True for check in _REVIEW_CHECKS
    }
    with pytest.raises(TypeError):
        verified.expected_images[0]["image_id"] = 999
    with pytest.raises(TypeError):
        verified.expected_images[0]["annotation_ids"][0] = 999
    with pytest.raises(TypeError):
        verified.expected_annotations[0]["bbox"][0] = 999


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("schema", "other.v1"), "schema is unsupported"),
        (("manifest_sha256", "0" * 64), "does not bind"),
        (("partition_role", "fine_tune_training"), "required partition role"),
        (("status", "pending"), "not approved"),
        (("reviewer", " \t"), "non-empty string"),
        (("reviewer", "r" * 257), "exceeds 256"),
        (("reviewed_at", "2026-08-16T10:30:00+00:00"), "ending in 'Z'"),
        (("reviewed_at", "2026-02-30T10:30:00Z"), "valid UTC"),
    ],
)
def test_verify_benchmark_run_inputs_rejects_invalid_review_fields(
    tmp_path, pinned_local_source, mutation, match
):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    payload = _review_payload(artifacts.manifest_sha256)
    key, value = mutation
    payload[key] = value
    review_path = tmp_path / "review.json"
    _write_review(review_path, payload)

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match=match):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            review_path,
        )


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda value: value["checks"].pop("visual_quality"), "missing visual_quality"),
        (
            lambda value: value["checks"].__setitem__("unexpected", True),
            "unsupported unexpected",
        ),
        (
            lambda value: value["checks"].__setitem__("visual_quality", 1),
            "visual_quality.*must be true",
        ),
        (
            lambda value: value["checks"].__setitem__("visual_quality", False),
            "visual_quality.*must be true",
        ),
        (lambda value: value.__setitem__("comment", "extra"), "unsupported comment"),
    ],
)
def test_verify_benchmark_run_inputs_requires_exact_all_true_checks(
    tmp_path, pinned_local_source, mutate, match
):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    payload = _review_payload(artifacts.manifest_sha256)
    mutate(payload)
    review_path = tmp_path / "review.json"
    _write_review(review_path, payload)

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match=match):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            review_path,
        )


@pytest.mark.parametrize(
    "raw,match",
    [
        (
            b'{"schema":"a","schema":"b"}',
            "duplicate key",
        ),
        (b'{"reviewer":NaN}', "not permitted"),
    ],
)
def test_verify_benchmark_run_inputs_rejects_non_strict_review_json(
    tmp_path, pinned_local_source, raw, match
):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    review_path = tmp_path / "review.json"
    review_path.write_bytes(raw)

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match=match):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            review_path,
        )


def test_verify_benchmark_run_inputs_bounds_and_externalizes_review(
    tmp_path, pinned_local_source, monkeypatch
):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    payload = _review_payload(artifacts.manifest_sha256)
    external = tmp_path / "review.json"
    encoded = _write_review(external, payload)
    monkeypatch.setattr(
        dataset_manifest, "_MAX_REVIEW_ATTESTATION_BYTES", len(encoded) - 1
    )
    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="safety limit"):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            external,
        )

    monkeypatch.setattr(dataset_manifest, "_MAX_REVIEW_ATTESTATION_BYTES", len(encoded))
    inside = artifacts.output_dir / "review.json"
    inside.write_bytes(encoded)
    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="outside"):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            inside,
        )


def test_verify_benchmark_run_inputs_rejects_review_symlink_and_wrong_role(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    artifacts = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    review_path = tmp_path / "review.json"
    _write_review(review_path, _review_payload(artifacts.manifest_sha256))
    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="existing file"):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            tmp_path,
        )
    review_link = tmp_path / "review-link.json"
    try:
        review_link.symlink_to(review_path)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="symlink"):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            review_link,
        )
    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="partition role"):
        dataset_manifest.verify_benchmark_run_inputs(
            artifacts.manifest_path,
            source.source,
            source.images,
            review_path,
            required_role="fine_tune_training",
        )


def test_build_review_template_is_exactly_unapproved_and_rejected_for_run(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    output = tmp_path / "reviews" / "promotion500.json"

    result = dataset_manifest.build_unapproved_review_template(
        bundle.manifest_path,
        source.source,
        source.images,
        output,
    )

    expected = {
        "schema": "libreyolo.vlm-benchmark-dataset-review.v1",
        "manifest_sha256": bundle.manifest_sha256,
        "partition_role": "zero_shot_confidence_promotion",
        "status": "unapproved",
        "reviewer": "",
        "reviewed_at": None,
        "checks": {check: False for check in _REVIEW_CHECKS},
    }
    assert result == dataset_manifest.BenchmarkReviewTemplateArtifacts(
        output_path=output.resolve(),
        manifest_sha256=bundle.manifest_sha256,
        partition_role="zero_shot_confidence_promotion",
    )
    assert _json(output) == expected
    assert output.read_bytes() == dataset_manifest._json_file_bytes(expected)
    assert not output.is_symlink()

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="not approved"):
        dataset_manifest.verify_benchmark_run_inputs(
            bundle.manifest_path,
            source.source,
            source.images,
            output,
        )


def test_build_review_template_verifies_bundle_before_writing(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    output = tmp_path / "review.json"
    (bundle.output_dir / "ATTRIBUTION.jsonl").write_bytes(b"modified\n")

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="modified"):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            output,
        )

    assert not output.exists()


def test_build_review_template_refuses_existing_or_bundled_output(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    existing = tmp_path / "existing.json"
    existing.write_bytes(b"do not replace")

    with pytest.raises(dataset_manifest.BenchmarkDatasetOutputExistsError):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            existing,
        )
    assert existing.read_bytes() == b"do not replace"

    bundled = bundle.output_dir / "review.json"
    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="outside"):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            bundled,
        )
    assert not bundled.exists()


def test_build_review_template_refuses_symlinked_output_paths(
    tmp_path, pinned_local_source
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    target = tmp_path / "target.json"
    target.write_bytes(b"target")
    output_link = tmp_path / "review-link.json"
    real_parent = tmp_path / "real-reviews"
    real_parent.mkdir()
    parent_link = tmp_path / "review-parent-link"
    try:
        output_link.symlink_to(target)
        parent_link.symlink_to(real_parent, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable in this test environment: {exc}")

    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="symlink"):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            output_link,
        )
    assert target.read_bytes() == b"target"

    ambiguous = parent_link / "review.json"
    with pytest.raises(
        dataset_manifest.BenchmarkDatasetError, match="symlinked parent"
    ):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            ambiguous,
        )
    assert not (real_parent / "review.json").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
def test_build_review_template_refuses_junction_parent(tmp_path, pinned_local_source):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    target = tmp_path / "real-reviews"
    target.mkdir()
    junction = tmp_path / "review-junction"
    process = subprocess.run(
        ["cmd", "/d", "/c", "mklink", "/J", str(junction), str(target)],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        pytest.skip(f"Windows junctions are unavailable: {process.stderr.strip()}")
    try:
        with pytest.raises(
            dataset_manifest.BenchmarkDatasetError, match="symlinked parent or junction"
        ):
            dataset_manifest.build_unapproved_review_template(
                bundle.manifest_path,
                source.source,
                source.images,
                junction / "review.json",
            )
        assert not (target / "review.json").exists()
    finally:
        os.rmdir(junction)


def test_review_template_atomic_publish_never_overwrites_racing_output(
    tmp_path, pinned_local_source, monkeypatch
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    output = tmp_path / "review.json"

    def racing_link(_source, _destination, **_kwargs):
        output.write_bytes(b"racing writer")
        raise FileExistsError

    monkeypatch.setattr(dataset_manifest.os, "link", racing_link)
    with pytest.raises(dataset_manifest.BenchmarkDatasetOutputExistsError):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            output,
        )

    assert output.read_bytes() == b"racing writer"
    assert not list(tmp_path.glob(".review.json.*.tmp"))


def test_review_template_refuses_parent_replacement_during_publish(
    tmp_path, pinned_local_source, monkeypatch
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    parent = tmp_path / "reviews"
    displaced_parent = tmp_path / "reviews-displaced"
    output = parent / "review.json"
    if dataset_manifest._SUPPORTS_DIR_FD_PUBLICATION:
        real_link = dataset_manifest.os.link

        def replacing_link(*args, **kwargs):
            parent.rename(displaced_parent)
            parent.mkdir()
            return real_link(*args, **kwargs)

        monkeypatch.setattr(dataset_manifest.os, "link", replacing_link)
    else:
        real_mkstemp = dataset_manifest.tempfile.mkstemp

        def replacing_mkstemp(*args, **kwargs):
            parent.rename(displaced_parent)
            parent.mkdir()
            return real_mkstemp(*args, **kwargs)

        monkeypatch.setattr(dataset_manifest.tempfile, "mkstemp", replacing_mkstemp)
    with pytest.raises(dataset_manifest.BenchmarkDatasetError, match="parent changed"):
        dataset_manifest.build_unapproved_review_template(
            bundle.manifest_path,
            source.source,
            source.images,
            output,
        )

    assert not output.exists()
    assert not (displaced_parent / "review.json").exists()
    assert not list(parent.glob(".review.json.*.tmp"))
    assert not list(displaced_parent.glob(".review.json.*.tmp"))


def test_review_template_cli_reports_unapproved_and_has_no_approval_fields(
    tmp_path, pinned_local_source, capsys
):
    source = pinned_local_source
    bundle = dataset_manifest.build_benchmark_dataset(
        source.source, source.images, tmp_path / "bundle"
    )
    output = tmp_path / "review.json"
    arguments = [
        "review-template",
        "--manifest",
        str(bundle.manifest_path),
        "--annotations",
        str(source.source),
        "--images-dir",
        str(source.images),
        "--output",
        str(output),
    ]

    code = dataset_manifest.main(arguments)

    assert code == 0
    status = json.loads(capsys.readouterr().out)
    assert status == {
        "schema": "libreyolo.vlm-benchmark-dataset-status.v1",
        "status": "ok",
        "mode": "review-template",
        "output": str(output.resolve()),
        "manifest_sha256": bundle.manifest_sha256,
        "partition_role": "zero_shot_confidence_promotion",
        "approved": False,
    }
    assert _json(output)["status"] == "unapproved"

    for forbidden in ("--approve", "--status", "--reviewer", "--reviewed-at"):
        with pytest.raises(SystemExit):
            dataset_manifest.build_parser().parse_args(
                [*arguments, forbidden, "approved"]
            )
