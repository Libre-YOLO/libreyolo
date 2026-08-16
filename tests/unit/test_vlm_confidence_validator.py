"""Offline tests for the internal VLM confidence validation harness."""

import hashlib
import json
import math
from dataclasses import replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import yaml
from PIL import Image

pytestmark = [pytest.mark.unit, pytest.mark.vlm]

torch = pytest.importorskip("torch")

from libreyolo.validation.config import ValidationConfig  # noqa: E402
from libreyolo.validation.detection_validator import DetectionValidator  # noqa: E402
from libreyolo.validation.preprocessors import StandardValPreprocessor  # noqa: E402
from libreyolo.validation.vlm_benchmark_dataset import (  # noqa: E402
    VerifiedBenchmarkRunInputs,
)
from libreyolo.validation.vlm_confidence import compare_repeats  # noqa: E402
from libreyolo.validation.vlm_confidence_report import (  # noqa: E402
    compare_confidence_reports,
)
from libreyolo.validation.vlm_confidence_validator import (  # noqa: E402
    VLMConfidenceValidator,
)


@pytest.fixture(autouse=True)
def _identify_stub_peft_runtime(monkeypatch):
    """Keep fake live-PEFT tests independent of the optional VLM extra."""

    real_version = importlib_metadata.version

    def version(package):
        if package == "peft":
            return "offline-stub"
        return real_version(package)

    monkeypatch.setattr(
        "libreyolo.validation.vlm_confidence_validator.metadata.version", version
    )


def _view(boxes, scores, classes=None):
    classes = [0] * len(boxes) if classes is None else classes
    return {
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
        "num_detections": len(boxes),
    }


def _variants(
    boxes,
    candidate_scores,
    *,
    available=True,
    parsed=None,
    generation_payload=None,
):
    parsed = len(boxes) if parsed is None else parsed
    if generation_payload is None:
        generation_payload = repr((boxes, candidate_scores, available, parsed))
    return SimpleNamespace(
        generation_hash=hashlib.sha256(generation_payload.encode()).hexdigest(),
        candidate=_view(boxes, candidate_scores if available else [1.0] * len(boxes)),
        constant=_view(boxes, [1.0] * len(boxes)),
        item_scores=(tuple(candidate_scores) if available else None),
        parsed_items=parsed,
        score_available=available,
        fallback_reason=None if available else "detection_alignment",
    )


class _GenerationModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = torch.nn.Parameter(torch.tensor([0.125]))
        self.train()


class _PeftConfig:
    def __init__(self, lora_alpha=16):
        self.lora_alpha = lora_alpha

    def to_dict(self):
        return {
            "r": 8,
            "lora_alpha": self.lora_alpha,
            "target_modules": {"q_proj", "v_proj"},
            "inference_mode": False,
            "base_model_name_or_path": "mutable/local/cache",
            "revision": None,
        }


class _LivePeftModel(torch.nn.Module):
    def __init__(self, base_model, *, lora_alpha=16, runtime_scale=2.0):
        super().__init__()
        self.base_model = base_model
        for parameter in base_model.parameters():
            parameter.requires_grad_(False)
        self.adapter = torch.nn.Parameter(torch.tensor([0.125]))
        self.register_buffer("adapter_scale", torch.tensor([1.0]))
        self.peft_config = {"default": _PeftConfig(lora_alpha)}
        self.active_adapters = ["default"]
        self.disable_adapters = False
        self.merged_adapters = []
        self.scaling = {"default": runtime_scale}
        self.train()
        self.base_model.eval()

    def get_base_model(self):
        return self.base_model


class _StubVLM:
    FAMILY = "qwen3vl"
    FILENAME_PREFIX = "StubQwen"
    HF_REPOS = {"stub": "stub/base"}
    HF_REVISIONS = {"stub": "a" * 40}
    MAX_NEW_TOKENS = 16
    REPETITION_PENALTY = 1.1
    nb_classes = 1
    size = "stub"
    # Deliberately wrong for the live model: the validator must derive the
    # target device from the supplied module, not trust this wrapper field.
    device = torch.device("cuda:7")
    DEFAULT_SCORE = 1.0

    def __init__(self, paths, variants, processor_dir):
        self.paths = [str(path) for path in paths]
        self.variants = variants
        self.processor = SimpleNamespace(name_or_path=str(processor_dir))
        self.names = {0: "cat"}
        self.model = _GenerationModel()
        self.preprocessed = []
        self.forward_models = []
        self.forward_count = 0

    def _get_model_name(self):
        return "stub_vlm"

    def _get_input_size(self):
        return 100

    def set_classes(self, classes):
        self.names = {index: str(name) for index, name in enumerate(classes)}
        self.nb_classes = len(self.names)
        return self

    def _detection_prompt(self):
        return "Detect: " + ", ".join(self.names.values())

    def _preprocess(self, path, color_format="auto", input_size=None):
        assert color_format == "auto"
        assert input_size == (100, 100)
        self.preprocessed.append(path)
        index = self.paths.index(path)
        return {"input_ids": torch.tensor([[index]])}, None, (100, 100), 1.0

    def _forward_for_confidence_gate(self, inputs, *, model=None):
        self.forward_count += 1
        self.forward_models.append(model)
        return int(inputs["input_ids"].item())

    def _postprocess_score_variants(self, output, original_size):
        assert original_size == (100, 100)
        return self.variants[output]


def _write_base_snapshot(
    root,
    *,
    repo="stub/base",
    revision="a" * 40,
    weights=None,
    weight_map=None,
):
    root.mkdir(parents=True, exist_ok=True)
    (root / ".libreyolo_snapshot_complete").write_text(
        json.dumps({"repo": repo, "revision": revision}) + "\n",
        encoding="utf-8",
    )
    (root / "config.json").write_text(
        '{"architectures":["OfflineStub"]}\n', encoding="utf-8"
    )
    if weights is None:
        weights = {"model.safetensors": b"offline-stub-weights"}
    for name, payload in weights.items():
        (root / name).write_bytes(payload)
    if weight_map is not None:
        (root / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": 1}, "weight_map": weight_map},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    (root / "preprocessor_config.json").write_text(
        '{"processor_class":"OfflineStub"}\n', encoding="utf-8"
    )
    return root


def _stub_model(tmp_path, paths, variants):
    processor_dir = tmp_path / "processor"
    _write_base_snapshot(processor_dir)
    return _StubVLM(paths, variants, processor_dir)


class _Dataset:
    def __init__(self, paths):
        self.img_files = [Path(path) for path in paths]

    def __len__(self):
        return len(self.img_files)


class _Loader(list):
    def __init__(self, batches, dataset):
        super().__init__(batches)
        self.dataset = dataset


class _Evaluator:
    def __init__(self, map_value, map50_value, backend="offline-stub"):
        self.map_value = map_value
        self.map50_value = map50_value
        self.updates = []
        self.last_backend = backend

    def update(self, view, image_id):
        self.updates.append((view, image_id))

    def compute(self, save_json=None):
        assert save_json is None
        return {"mAP": self.map_value, "mAP50": self.map50_value}


class _Harness(VLMConfidenceValidator):
    def __init__(
        self,
        model,
        config,
        paths,
        targets,
        *,
        evaluator_backend="offline-stub",
        **kwargs,
    ):
        self._stub_paths = paths
        self._stub_targets = targets
        self._stub_evaluator_backend = evaluator_backend
        super().__init__(model, config, **kwargs)

    def _setup_dataloader(self):
        self._actual_imgsz = (100, 100)
        self.val_preproc = SimpleNamespace(uses_letterbox=False)
        self.class_names = ["cat"]
        self.nc = 1
        self.model.set_classes(self.class_names)
        dataset = _Dataset(self._stub_paths)
        batch = (
            torch.zeros((len(dataset), 3, 1, 1)),
            self._stub_targets,
            [(100, 100)] * len(dataset),
            list(range(1, len(dataset) + 1)),
        )
        return _Loader([batch], dataset)

    def _init_metrics(self):
        self.candidate_evaluator = _Evaluator(0.6, 0.8, self._stub_evaluator_backend)
        self.constant_evaluator = _Evaluator(0.4, 0.5, self._stub_evaluator_backend)
        self.coco_evaluator = self.candidate_evaluator
        images = {
            index + 1: {"id": index + 1, "width": 100, "height": 100}
            for index in range(len(self._stub_paths))
        }
        annotations = {}
        annotation_id = 1
        for image_index, image_targets in enumerate(self._stub_targets):
            for target in image_targets:
                x1, y1, x2, y2, class_id = target.tolist()
                if x2 <= x1 or y2 <= y1:
                    continue
                annotations[annotation_id] = {
                    "id": annotation_id,
                    "image_id": image_index + 1,
                    "category_id": int(class_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
                annotation_id += 1
        self._gt_coco_api = SimpleNamespace(
            imgs=images,
            anns=annotations,
            cats={0: {"id": 0, "name": "cat"}},
        )
        self._coco_label_to_category_id = None
        self._reset_confidence_records()


class _BoundHarness(_Harness):
    def __init__(self, *args, binding_mutation=None, **kwargs):
        self._binding_mutation = binding_mutation
        super().__init__(*args, **kwargs)

    def _setup_dataloader(self):
        loader = super()._setup_dataloader()
        verified = self._verified_dataset
        assert verified is not None
        loader.dataset.ids = [
            int(image["image_id"]) for image in verified.expected_images
        ]
        loader.dataset.coco = SimpleNamespace(
            imgs={
                int(image["image_id"]): {
                    "id": int(image["image_id"]),
                    "file_name": str(image["file_name"]),
                    "width": int(image["width"]),
                    "height": int(image["height"]),
                }
                for image in verified.expected_images
            }
        )
        if self._binding_mutation == "order":
            loader.dataset.ids.reverse()
        elif self._binding_mutation == "name":
            loader.dataset.coco.imgs[loader.dataset.ids[0]]["file_name"] = "wrong.jpg"
        elif self._binding_mutation == "path":
            wrong = verified.images_dir / "wrong.jpg"
            wrong.write_bytes(Path(loader.dataset.img_files[0]).read_bytes())
            loader.dataset.img_files[0] = wrong
        elif self._binding_mutation == "dimensions":
            loader.dataset.coco.imgs[loader.dataset.ids[0]]["width"] += 1
        elif self._binding_mutation == "short_batch":
            images, targets, sizes, image_ids = loader[0]
            loader[0] = (
                images[:1],
                targets[:1],
                sizes[:1],
                image_ids[:1],
            )
        return loader

    def _init_metrics(self):
        super()._init_metrics()
        verified = self._verified_dataset
        assert verified is not None
        self._coco_annotation_file = verified.annotation_path
        self._coco_label_to_category_id = {
            label: int(category["id"])
            for label, category in enumerate(verified.expected_categories)
        }
        if self._binding_mutation == "ground_truth":
            self._gt_coco_api.anns[1]["bbox"][0] += 1
        elif self._binding_mutation == "category":
            self._gt_coco_api.cats[0]["name"] = "dog"
        elif self._binding_mutation == "annotation_path":
            wrong = verified.annotation_path.parent / "same-bytes-wrong-path.json"
            wrong.write_bytes(verified.annotation_path.read_bytes())
            self._coco_annotation_file = wrong


def _file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_inputs(tmp_path, paths, targets):
    bundle = tmp_path / "bundle"
    annotation_dir = bundle / "annotations"
    annotation_dir.mkdir(parents=True)
    manifest = bundle / "manifest.json"
    manifest.write_text('{"verified":true}\n', encoding="utf-8")
    annotation = annotation_dir / "promotion.json"
    annotation.write_text('{"verified":true}\n', encoding="utf-8")
    source = tmp_path / "source.json"
    source.write_text('{"source":true}\n', encoding="utf-8")
    review = tmp_path / "review.json"
    review.write_text('{"approved":true}\n', encoding="utf-8")

    expected_annotations = []
    annotation_id = 1
    for image_index, image_targets in enumerate(targets):
        for target in image_targets:
            x1, y1, x2, y2, class_id = target.tolist()
            if x2 <= x1 or y2 <= y1:
                continue
            expected_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_index + 1,
                    "category_id": int(class_id),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "area": (x2 - x1) * (y2 - y1),
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
    expected_images = tuple(
        {
            "image_id": index + 1,
            "file_name": Path(path).name,
            "width": 100,
            "height": 100,
            "size_bytes": Path(path).stat().st_size,
            "sha256": _file_sha256(path),
        }
        for index, path in enumerate(paths)
    )
    review_payload = {
        "schema": "libreyolo.vlm-benchmark-dataset-review.v1",
        "manifest_sha256": _file_sha256(manifest),
        "partition_role": "zero_shot_confidence_promotion",
        "status": "approved",
        "reviewer": "Offline Reviewer",
        "reviewed_at": "2026-08-16T12:00:00Z",
        "checks": {},
    }
    return VerifiedBenchmarkRunInputs(
        manifest_path=manifest.resolve(),
        manifest_sha256=_file_sha256(manifest),
        source_annotations=source.resolve(),
        source_canonical_sha256="1" * 64,
        source_file_sha256=_file_sha256(source),
        source_file_size_bytes=source.stat().st_size,
        images_dir=Path(paths[0]).parent.resolve(),
        selected_image_identity_sha256="2" * 64,
        partition_name="promotion-test",
        partition_role="zero_shot_confidence_promotion",
        partition_start=0,
        partition_stop=len(paths),
        annotation_path=annotation.resolve(),
        annotation_sha256=_file_sha256(annotation),
        annotation_size_bytes=annotation.stat().st_size,
        class_names=("cat",),
        expected_images=expected_images,
        expected_categories=({"id": 0, "name": "cat"},),
        expected_annotations=tuple(expected_annotations),
        review_attestation_path=review.resolve(),
        review_attestation_sha256=_file_sha256(review),
        review_attestation=review_payload,
    )


def _verified_context(verified):
    review = verified.review_attestation
    return {
        "schema": "libreyolo.vlm-confidence-benchmark-context.test",
        "dataset": {
            "schema": "libreyolo.vlm-confidence-benchmark-dataset.v1",
            "manifest": {
                "schema": "libreyolo.vlm-benchmark-dataset.v1",
                "sha256": verified.manifest_sha256,
            },
            "source": {
                "canonical_annotation_sha256": verified.source_canonical_sha256,
                "file_sha256": verified.source_file_sha256,
                "file_size_bytes": verified.source_file_size_bytes,
                "selected_image_identity_sha256": (
                    verified.selected_image_identity_sha256
                ),
            },
            "partition": {
                "name": verified.partition_name,
                "role": verified.partition_role,
                "start": verified.partition_start,
                "stop": verified.partition_stop,
                "image_count": verified.partition_stop - verified.partition_start,
                "annotation_artifact": verified.annotation_path.relative_to(
                    verified.manifest_path.parent
                ).as_posix(),
                "annotation_size_bytes": verified.annotation_size_bytes,
                "annotation_sha256": verified.annotation_sha256,
            },
            "classes": {"count": 1, "names": ["cat"], "category_ids": [0]},
            "review": {
                "schema": review["schema"],
                "sha256": verified.review_attestation_sha256,
                "manifest_sha256": review["manifest_sha256"],
                "partition_role": review["partition_role"],
                "status": review["status"],
                "reviewer": review["reviewer"],
                "reviewed_at": review["reviewed_at"],
                "checks": review["checks"],
            },
        },
    }


def _config(tmp_path, **kwargs):
    save_dir = kwargs.pop("save_dir", tmp_path / "results")
    return ValidationConfig(
        data_dir=str(tmp_path),
        batch_size=2,
        num_workers=0,
        verbose=False,
        save_dir=str(save_dir),
        faster_coco_eval=False,
        **kwargs,
    )


def _targets():
    targets = torch.zeros((2, 4, 5), dtype=torch.float32)
    targets[0, 0] = torch.tensor([10, 10, 30, 30, 0])
    targets[1, 0] = torch.tensor([20, 20, 40, 40, 0])
    return targets


def _bound_validator(tmp_path, *, binding_mutation=None):
    images = tmp_path / "images"
    images.mkdir()
    paths = [images / "one.jpg", images / "two.jpg"]
    for path in paths:
        path.write_bytes(b"offline")
    targets = _targets()
    verified = _verified_inputs(tmp_path, paths, targets)
    model = _stub_model(tmp_path, paths, [_variants([], []), _variants([], [])])
    validator = _BoundHarness(
        model,
        _config(tmp_path),
        paths,
        targets,
        benchmark_context=_verified_context(verified),
        verified_dataset=verified,
        binding_mutation=binding_mutation,
    )
    return validator, model, verified, paths


def test_verified_dataset_binding_runs_and_records_portable_context(tmp_path):
    validator, model, verified, paths = _bound_validator(tmp_path)

    validator.run()

    assert model.forward_count == 2
    assert (
        validator.benchmark_config["benchmark_run"]["dataset"]
        == (_verified_context(verified)["dataset"])
    )
    serialized_context = json.dumps(
        validator.benchmark_config["benchmark_run"]["dataset"], sort_keys=True
    )
    assert str(tmp_path) not in serialized_context
    assert [item["file_name"] for item in validator.dataset_manifest["images"]] == [
        path.name for path in paths
    ]


def test_portable_dataset_context_requires_verified_evidence(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"offline")
    model = _stub_model(tmp_path, [path], [_variants([], [])])

    with pytest.raises(ValueError, match="requires matching verified_dataset"):
        _Harness(
            model,
            _config(tmp_path),
            [path],
            torch.zeros((1, 1, 5)),
            benchmark_context={"dataset": {}},
        )
    assert model.forward_count == 0


def test_verified_evidence_must_match_portable_dataset_context(tmp_path):
    validator, model, verified, paths = _bound_validator(tmp_path)
    mismatched = replace(verified, manifest_sha256="f" * 64)

    with pytest.raises(ValueError, match="does not match verified_dataset"):
        _BoundHarness(
            model,
            _config(tmp_path, save_dir=tmp_path / "mismatch"),
            paths,
            _targets(),
            benchmark_context=_verified_context(verified),
            verified_dataset=mismatched,
        )
    assert validator._verified_dataset is verified
    assert model.forward_count == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("ground_truth", "ground-truth annotations"),
        ("category", "ground-truth categories"),
        ("order", "image order"),
        ("name", "image name"),
        ("path", "image path"),
        ("annotation_path", "annotation path"),
        ("dimensions", "image dimensions"),
    ],
)
def test_verified_dataset_mismatch_fails_before_first_forward(
    tmp_path, mutation, message
):
    validator, model, _, _ = _bound_validator(tmp_path, binding_mutation=mutation)

    with pytest.raises(RuntimeError, match=message):
        validator.run()

    assert model.forward_count == 0


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("image", "Benchmark image 2.*before generation"),
        ("source", "source annotations.*before generation"),
        ("annotation", "annotation artifact.*before generation"),
        ("manifest", "manifest.*before generation"),
        ("review", "review attestation.*before generation"),
    ],
)
def test_verified_hash_mismatch_fails_before_first_forward(tmp_path, artifact, message):
    validator, model, verified, paths = _bound_validator(tmp_path)
    targets = {
        "image": paths[-1],
        "source": verified.source_annotations,
        "annotation": verified.annotation_path,
        "manifest": verified.manifest_path,
        "review": verified.review_attestation_path,
    }
    targets[artifact].write_bytes(b"changed-before-run")

    with pytest.raises(RuntimeError, match=message):
        validator.run()

    assert model.forward_count == 0


def test_verified_decoded_dimensions_fail_before_first_forward(tmp_path):
    validator, model, _, _ = _bound_validator(tmp_path)
    original = model._preprocess

    def wrong_size(*args, **kwargs):
        inputs, image, _, ratio = original(*args, **kwargs)
        return inputs, image, (99, 100), ratio

    model._preprocess = wrong_size

    with pytest.raises(RuntimeError, match="Decoded image dimensions"):
        validator.run()

    assert model.forward_count == 0


def test_verified_image_is_rehashed_after_its_generation(tmp_path):
    validator, model, _, paths = _bound_validator(tmp_path)
    original = model._forward_for_confidence_gate

    def mutate_current(inputs, *, model=None):
        output = original(inputs, model=model)
        paths[0].write_bytes(b"changed-during-forward")
        return output

    model._forward_for_confidence_gate = mutate_current

    with pytest.raises(RuntimeError, match="changed during generation"):
        validator.run()

    assert model.forward_count == 1


@pytest.mark.parametrize(
    ("artifact", "message"),
    [
        ("prior_image", "Benchmark image 1.*after generation"),
        ("source", "source annotations.*after generation"),
        ("annotation", "annotation artifact.*after generation"),
    ],
)
def test_verified_files_receive_a_final_full_rehash(tmp_path, artifact, message):
    validator, model, verified, paths = _bound_validator(tmp_path)
    original = model._forward_for_confidence_gate

    def mutate_after_prior_check(inputs, *, model=None):
        output = original(inputs, model=model)
        if output == 1:
            if artifact == "prior_image":
                paths[0].write_bytes(b"changed-after-its-own-check")
            elif artifact == "source":
                verified.source_annotations.write_bytes(b"changed-source")
            else:
                verified.annotation_path.write_bytes(b"changed-annotation")
        return output

    model._forward_for_confidence_gate = mutate_after_prior_check

    with pytest.raises(RuntimeError, match=message):
        validator.run()

    assert model.forward_count == 2


def test_verified_dataset_enforces_final_processed_image_count(tmp_path):
    validator, model, _, _ = _bound_validator(tmp_path, binding_mutation="short_batch")

    with pytest.raises(RuntimeError, match="image count"):
        validator.run()

    assert model.forward_count == 1


def test_serial_gate_reports_coco_deltas_quality_and_coverage(tmp_path):
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    for path in paths:
        path.write_bytes(b"offline")
    variants = [
        _variants([[10, 10, 30, 30], [50, 50, 70, 70]], [0.9, 0.1]),
        _variants([[20, 20, 40, 40]], [0.7], available=False),
    ]
    model = _stub_model(tmp_path, paths, variants)
    model.size = "2b"
    model.HF_REPOS = {"2b": "stub/base"}
    model.HF_REVISIONS = {"2b": "a" * 40}
    model.set_classes(["dog"])
    live_model = _LivePeftModel(model.model)
    runner_context = {
        "schema": "libreyolo.vlm-confidence-test-context.v1",
        "git": {"commit": "c" * 40, "dirty": False},
        "runtime": {
            "python": "3.11.0",
            "implementation": "CPython",
            "platform": "offline",
            "torch": "offline",
            "numpy": "offline",
            "pillow": "offline",
            "opencv": "offline",
            "packages": {
                "transformers": "offline",
                "huggingface_hub": "offline",
                "tokenizers": "offline",
                "safetensors": "offline",
                "pycocotools": "offline",
            },
            "cuda_runtime": None,
            "cudnn": None,
            "nvidia_driver": None,
            "cuda_available": False,
            "requested_device": "auto",
            "resolved_device": "cpu",
            "attention_backends": {"model": "offline"},
        },
        "determinism": {
            "seed": 0,
            "python_hash_seed": "0",
            "python_hash_randomization": False,
            "cublas_workspace_config": ":4096:8",
            "torch_deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }
    validator = _Harness(
        model,
        _config(tmp_path),
        paths,
        _targets(),
        generation_model=live_model,
        benchmark_context=runner_context,
        evaluator_backend="pycocotools offline-stub",
    )
    caller_rng_state = torch.random.get_rng_state().clone()

    metrics = validator.run()

    assert model.forward_count == 2
    assert model.preprocessed == [str(path) for path in paths]
    assert model.forward_models == [live_model, live_model]
    assert live_model.training is True
    assert live_model.base_model.training is False
    assert model.names == {0: "dog"}
    assert torch.equal(torch.random.get_rng_state(), caller_rng_state)
    assert metrics["metrics/vlm_confidence/candidate_mAP50-95"] == 0.6
    assert metrics["metrics/vlm_confidence/constant_mAP50-95"] == 0.4
    assert metrics["metrics/vlm_confidence/delta_mAP50-95"] == pytest.approx(0.2)
    assert metrics["metrics/vlm_confidence/delta_mAP50"] == pytest.approx(0.3)
    assert metrics["metrics/vlm_confidence/auroc"] == 1.0
    assert metrics["metrics/vlm_confidence/ranking_ap"] == 1.0
    assert metrics["metrics/vlm_confidence/scored_prediction_brier"] == pytest.approx(
        0.01
    )
    assert metrics["metrics/vlm_confidence/scored_prediction_ece"] == pytest.approx(0.1)
    assert metrics["metrics/vlm_confidence/scored_prediction_mce"] == pytest.approx(0.1)
    assert metrics["metrics/vlm_confidence/default_conf_tp_retention"] == 1.0
    assert metrics["metrics/vlm_confidence/default_conf_fp_retention"] == 0.0
    assert metrics[
        "metrics/vlm_confidence/default_conf_prediction_retention"
    ] == pytest.approx(2 / 3)
    assert metrics["metrics/vlm_confidence/response_score_coverage"] == 0.5
    assert metrics["metrics/vlm_confidence/detection_score_coverage"] == pytest.approx(
        2 / 3
    )
    assert metrics["metrics/vlm_confidence/prediction_score_coverage"] == pytest.approx(
        2 / 3
    )
    assert validator.confidence_run.matches == (True, False, True)
    assert validator.fallback_reasons == {"detection_alignment": 1}
    assert validator.benchmark_config["family"] == "qwen3vl"
    assert validator.benchmark_config["device"] == "cpu"
    assert validator.benchmark_config["base_revision"] == "a" * 40
    assert (
        validator.benchmark_config["confidence_method"]
        == "qwen_generation_policy_label_bbox_geomean_v1"
    )
    assert validator.benchmark_config["evaluation"] == {
        "max_det": 100,
        "faster_coco_eval": False,
        "imgsz": [100, 100],
        "backend": "pycocotools offline-stub",
        "label_to_category_id": None,
    }
    assert validator.benchmark_config["confidence_evaluation"] == {
        "iou_threshold": 0.5,
        "default_conf": 0.25,
        "fallback_score": 1.0,
        "calibration_bins": 10,
        "binning": "uniform_left_closed_v1",
        "population": "scored_postprocessed_predictions",
        "matching": "class_aware_max_cardinality_iou_v1",
    }
    assert len(validator.benchmark_config["checkpoint"]["sha256"]) == 64
    assert len(validator.benchmark_config["processor"]["sha256"]) == 64
    assert validator.benchmark_config["benchmark_run"] == runner_context

    report_path = tmp_path / "results" / "vlm_confidence_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert compare_confidence_reports(report_path, report_path).reproducible
    assert report["hashes"]["manifest"] == validator.confidence_run.manifest_hash
    assert report["hashes"]["generation"] == validator.confidence_run.generation_hash
    assert report["benchmark_config"] == validator.benchmark_config
    expected_image_hash = hashlib.sha256(b"offline").hexdigest()
    assert [item["sha256"] for item in report["dataset_manifest"]["images"]] == [
        expected_image_hash,
        expected_image_hash,
    ]
    assert report["predictions"][2]["candidate_score"] is None
    assert report["predictions"][2]["effective_score"] == 1.0

    assert report["calibration"]["population"] == "scored_postprocessed_predictions"
    assert report["calibration"]["total_predictions"] == 3
    assert report["calibration"]["scored_predictions"] == 2
    assert report["calibration"]["unscored_predictions"] == 1
    assert report["calibration"]["score_coverage"] == pytest.approx(2 / 3)
    assert report["calibration"]["bins"][1]["count"] == 1
    assert report["calibration"]["bins"][9]["correct"] == 1
    assert report["evaluator_metrics"] == {
        "candidate_mAP50": 0.8,
        "candidate_mAP50-95": 0.6,
        "constant_mAP50": 0.5,
        "constant_mAP50-95": 0.4,
    }
    for key in (
        "speed/preprocess_ms",
        "speed/inference_ms",
        "speed/postprocess_ms",
        "speed/total_ms",
        "speed/total_s",
        "speed/images_seen",
    ):
        assert report["metrics"][key] == metrics[key]
        assert report["metrics"][key] >= 0
    assert report["artifacts"]["reliability_plot"] is None

    assert [image_id for _, image_id in validator.candidate_evaluator.updates] == [1, 2]
    assert [image_id for _, image_id in validator.constant_evaluator.updates] == [1, 2]
    candidate_scores = [
        score
        for view, _ in validator.candidate_evaluator.updates
        for score in view["scores"]
    ]
    constant_scores = [
        score
        for view, _ in validator.constant_evaluator.updates
        for score in view["scores"]
    ]
    assert candidate_scores == [0.9, 0.1, 1.0]
    assert constant_scores == [1.0, 1.0, 1.0]


@pytest.mark.parametrize("option", ["augment", "cuda_graph", "allow_download_scripts"])
def test_unsupported_modes_fail_loudly(tmp_path, option):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"offline")
    model = _stub_model(tmp_path, [path], [_variants([], [])])
    config = _config(tmp_path, **{option: True})
    with pytest.raises(NotImplementedError, match=option):
        _Harness(
            model,
            config,
            [path],
            torch.zeros((1, 1, 5)),
        )


def test_save_plots_writes_candidate_reliability_svg(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"offline")
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    targets[0, 0] = torch.tensor([10, 10, 30, 30, 0])
    model = _stub_model(tmp_path, [path], [_variants([[10, 10, 30, 30]], [0.8])])
    validator = _Harness(
        model,
        _config(tmp_path, save_plots=True),
        [path],
        targets,
    )

    metrics = validator.run()

    plot = tmp_path / "results" / "vlm_confidence_reliability.svg"
    ElementTree.parse(plot)
    text = plot.read_text(encoding="utf-8")
    assert "Candidate token probability reliability (diagnostic)" in text
    assert "N=1/1 scored (100.0%)" in text
    assert "n=1; confidence=0.8000; accuracy=1.0000" in text
    assert not (tmp_path / "results" / "plots").exists()
    assert metrics["metrics/vlm_confidence/scored_prediction_brier"] == pytest.approx(
        0.04
    )
    assert "save_plots" not in validator.benchmark_config["evaluation"]
    report = json.loads(
        (tmp_path / "results" / "vlm_confidence_report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["schema"] == "libreyolo.vlm-confidence-report.v2"
    assert report["artifacts"]["reliability_plot"] == plot.name


def test_plot_emission_does_not_change_benchmark_identity(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"offline")
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    variants = [_variants([], [], generation_payload="same response")]
    first = _Harness(
        _stub_model(tmp_path, [path], variants),
        _config(tmp_path, save_dir=tmp_path / "without-plot"),
        [path],
        targets,
    )
    second = _Harness(
        _stub_model(tmp_path, [path], variants),
        _config(tmp_path, save_dir=tmp_path / "with-plot", save_plots=True),
        [path],
        targets,
    )

    first.run()
    second.run()

    assert first.confidence_run.configuration_hash == (
        second.confidence_run.configuration_hash
    )
    assert first.confidence_run.manifest_hash == second.confidence_run.manifest_hash


def test_missing_original_path_fails_before_generation(tmp_path):
    path = tmp_path / "missing.jpg"
    model = _stub_model(tmp_path, [path], [_variants([], [])])
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    live_model = _LivePeftModel(model.model)
    validator = _Harness(
        model,
        _config(tmp_path),
        [path],
        targets,
        generation_model=live_model,
    )

    with pytest.raises(FileNotFoundError, match="missing.jpg"):
        validator.run()
    assert model.forward_count == 0
    assert live_model.training is True


def test_missing_configured_dataset_directory_fails_loudly(tmp_path):
    missing = tmp_path / "missing-dataset"
    model = _stub_model(tmp_path, [], [])
    config = ValidationConfig(
        data_dir=str(missing),
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "results"),
    )
    validator = VLMConfidenceValidator(model, config)

    with pytest.raises(FileNotFoundError, match="missing-dataset"):
        validator.run()


def test_score_dependent_geometry_change_is_rejected(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"offline")
    variants = _variants([[10, 10, 30, 30]], [0.9])
    variants.candidate = _view([[11, 10, 30, 30]], [0.9])
    model = _stub_model(tmp_path, [path], [variants])
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    validator = _Harness(
        model,
        _config(tmp_path),
        [path],
        targets,
    )

    with pytest.raises(RuntimeError, match="changed.*geometry order"):
        validator.run()


def test_unsupported_vlm_family_is_refused(tmp_path):
    model = _stub_model(tmp_path, [], [])
    model.FAMILY = "internvl3"

    with pytest.raises(NotImplementedError, match="only.*qwen3vl"):
        VLMConfidenceValidator(model, _config(tmp_path))


def test_built_in_dataset_alias_is_left_to_inherited_resolver(tmp_path, monkeypatch):
    model = _stub_model(tmp_path, [], [])
    delegated = []

    def fake_setup(instance):
        delegated.append(instance.config.data)
        instance.class_names = ["cat"]
        return _Loader([], _Dataset([tmp_path / "resolved.jpg"]))

    monkeypatch.setattr(DetectionValidator, "_setup_dataloader", fake_setup)
    config = ValidationConfig(
        data="coco8.yaml",
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "results"),
    )
    validator = VLMConfidenceValidator(model, config)

    loader = validator._setup_dataloader()

    assert len(loader.dataset) == 1
    assert delegated == ["coco8.yaml"]
    assert validator.class_names == ["cat"]


def test_generated_response_hash_distinguishes_otherwise_identical_runs(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"same-image")
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    targets[0, 0] = torch.tensor([10, 10, 30, 30, 0])
    first_model = _stub_model(
        tmp_path,
        [path],
        [
            _variants(
                [[10, 10, 30, 30]],
                [0.9],
                generation_payload="first decoded response",
            )
        ],
    )
    second_model = _stub_model(
        tmp_path,
        [path],
        [
            _variants(
                [[10, 10, 30, 30]],
                [0.9],
                generation_payload="second decoded response",
            )
        ],
    )

    first = _Harness(
        first_model,
        _config(tmp_path, save_dir=tmp_path / "first"),
        [path],
        targets,
        generation_model=_LivePeftModel(first_model.model),
    )
    second = _Harness(
        second_model,
        _config(tmp_path, save_dir=tmp_path / "second"),
        [path],
        targets,
        generation_model=_LivePeftModel(second_model.model),
    )

    first_metrics = first.run()
    second.run()
    comparison = compare_repeats(first.confidence_run, second.confidence_run)

    assert comparison.same_manifest is True
    assert comparison.same_configuration is True
    assert comparison.same_prediction_structure is True
    assert comparison.same_generation is False
    assert comparison.reproducible is False
    assert math.isnan(first_metrics["metrics/vlm_confidence/default_conf_fp_retention"])
    first_report = json.loads(
        (tmp_path / "first" / "vlm_confidence_report.json").read_text(encoding="utf-8")
    )
    assert first_report["diagnostics"]["incorrect_retention"] is None


def test_actual_evaluator_backend_changes_configuration_identity(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"same-image")
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    variants = [_variants([], [], generation_payload="same response")]

    first = _Harness(
        _stub_model(tmp_path, [path], variants),
        _config(tmp_path, save_dir=tmp_path / "first-backend"),
        [path],
        targets,
        evaluator_backend="pycocotools 2.0.11",
    )
    second = _Harness(
        _stub_model(tmp_path, [path], variants),
        _config(tmp_path, save_dir=tmp_path / "second-backend"),
        [path],
        targets,
        evaluator_backend="faster-coco-eval 1.7.2",
    )

    first.run()
    second.run()

    assert first.confidence_run.generation_hash == second.confidence_run.generation_hash
    assert (
        first.confidence_run.configuration_hash
        != second.confidence_run.configuration_hash
    )
    assert first.benchmark_config["evaluation"]["backend"] == "pycocotools 2.0.11"
    assert second.benchmark_config["evaluation"]["backend"] == "faster-coco-eval 1.7.2"


def test_real_yolo_loader_writer_and_report_reader_round_trip(tmp_path, monkeypatch):
    pytest.importorskip("pycocotools")
    images = tmp_path / "images" / "val"
    labels = tmp_path / "labels" / "val"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image_path = images / "one.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    (labels / "one.txt").write_text("0 0.2 0.2 0.2 0.2\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/val",
                "nc": 1,
                "names": ["cat"],
                "download": "https://example.invalid/must-not-run.zip",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "libreyolo.data.utils.check_dataset",
        lambda *_args, **_kwargs: pytest.fail("confidence gate attempted a download"),
    )
    model = _stub_model(
        tmp_path,
        [image_path],
        [_variants([[10, 10, 30, 30]], [0.8])],
    )
    model._get_val_preprocessor = lambda img_size=None: StandardValPreprocessor(
        img_size=(img_size, img_size)
    )
    original_preprocess = model._preprocess

    def preprocess(path, color_format="auto", input_size=None):
        size = (input_size, input_size) if isinstance(input_size, int) else input_size
        return original_preprocess(
            path, color_format=color_format, input_size=tuple(size)
        )

    model._preprocess = preprocess
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=100,
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "real-results"),
        faster_coco_eval=False,
    )

    metrics = VLMConfidenceValidator(model, config).run()
    report_path = tmp_path / "real-results" / "vlm_confidence_report.json"

    assert metrics["metrics/vlm_confidence/candidate_mAP50-95"] == pytest.approx(1.0)
    assert compare_confidence_reports(report_path, report_path).reproducible


def test_missing_url_dataset_fails_without_download_or_generation(
    tmp_path, monkeypatch
):
    data_yaml = tmp_path / "missing.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path / "not-present"),
                "val": "images/val",
                "nc": 1,
                "names": ["cat"],
                "download": "https://example.invalid/must-not-run.zip",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "libreyolo.data.utils.check_dataset",
        lambda *_args, **_kwargs: pytest.fail("confidence gate attempted a download"),
    )
    model = _stub_model(tmp_path, [], [])
    model._get_val_preprocessor = lambda img_size=None: StandardValPreprocessor(
        img_size=(img_size, img_size)
    )
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=100,
        num_workers=0,
        verbose=False,
        save_dir=str(tmp_path / "missing-results"),
        faster_coco_eval=False,
    )

    with pytest.raises((FileNotFoundError, RuntimeError, ValueError)):
        VLMConfidenceValidator(model, config).run()

    assert model.forward_count == 0


@pytest.mark.parametrize(
    ("image_size", "raw_bbox", "clean_box", "names"),
    [
        ((100, 100), (-5, 10, 20, 20), (0, 10, 15, 30), ("cat", "dog")),
        ((1000, 333), (100, 100, 233, 200), (100, 100, 333, 300), ("cat",)),
    ],
)
def test_native_coco_writer_reader_uses_canonical_ordering_ground_truth(
    tmp_path, image_size, raw_bbox, clean_box, names
):
    pytest.importorskip("pycocotools")
    width, height = image_size
    image_dir = tmp_path / "images" / "custom_val"
    annotation_dir = tmp_path / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    image_path = image_dir / "one.jpg"
    Image.new("RGB", image_size, "white").save(image_path)
    (annotation_dir / "valid.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": image_path.name,
                        "width": width,
                        "height": height,
                    }
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 5,
                        "bbox": list(raw_bbox),
                        "area": raw_bbox[2] * raw_bbox[3],
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 5, "name": "cat"}],
            }
        ),
        encoding="utf-8",
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(tmp_path),
                "val": "images/custom_val",
                "annotations": {"val": "annotations/valid.json"},
                "nc": len(names),
                "names": list(names),
            }
        ),
        encoding="utf-8",
    )
    model = _stub_model(
        tmp_path,
        [image_path],
        [_variants([list(clean_box)], [0.8])],
    )
    model._get_val_preprocessor = lambda img_size=None: StandardValPreprocessor(
        img_size=(img_size, img_size)
    )
    model._preprocess = lambda path, color_format="auto", input_size=None: (
        {"input_ids": torch.tensor([[0]])},
        None,
        image_size,
        1.0,
    )
    model._postprocess_score_variants = lambda output, original_size: model.variants[
        output
    ]
    save_dir = tmp_path / "native-results"
    config = ValidationConfig(
        data=str(data_yaml),
        batch_size=1,
        imgsz=100,
        num_workers=0,
        verbose=False,
        save_dir=str(save_dir),
        faster_coco_eval=False,
    )

    VLMConfidenceValidator(model, config).run()
    report_path = save_dir / "vlm_confidence_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["dataset_manifest"]["ground_truth"] == [
        {"image_id": "1", "class_id": 0, "xyxy": list(clean_box)}
    ]
    assert report["benchmark_config"]["evaluation"]["label_to_category_id"] == {"0": 5}
    assert compare_confidence_reports(report_path, report_path).reproducible


def test_unfingerprintable_live_model_is_refused_before_generation(tmp_path):
    path = tmp_path / "one.jpg"
    path.write_bytes(b"offline")
    model = _stub_model(tmp_path, [path], [_variants([], [])])
    frozen = _LivePeftModel(model.model)
    frozen.adapter.requires_grad_(False)
    validator = _Harness(
        model,
        _config(tmp_path),
        [path],
        torch.zeros((1, 1, 5)),
        generation_model=frozen,
    )

    with pytest.raises(RuntimeError, match="trainable PEFT state"):
        validator.run()
    assert model.forward_count == 0


def test_processor_fingerprint_ignores_hf_cache_metadata(tmp_path):
    first = tmp_path / "first-processor"
    second = tmp_path / "second-processor"
    for root, timestamp in ((first, "1.0"), (second, "2.0")):
        (root / ".cache" / "huggingface" / "download").mkdir(parents=True)
        (root / "preprocessor_config.json").write_text(
            '{"processor_class":"Qwen"}\n', encoding="utf-8"
        )
        (root / ".cache" / "huggingface" / "download" / "config.metadata").write_text(
            timestamp, encoding="utf-8"
        )

    first_hash = VLMConfidenceValidator._directory_sha256(
        first, processor_artifacts=True
    )
    second_hash = VLMConfidenceValidator._directory_sha256(
        second, processor_artifacts=True
    )

    assert first_hash == second_hash


def _base_checkpoint_identity(tmp_path, model):
    validator = VLMConfidenceValidator(model, _config(tmp_path))
    return validator._checkpoint_identity(
        model.model,
        model.HF_REPOS[model.size],
        model.HF_REVISIONS[model.size],
    )


def test_base_snapshot_checkpoint_binds_weight_bytes_and_ignores_cache(tmp_path):
    model = _stub_model(tmp_path, [], [])
    snapshot = Path(model.processor.name_or_path)

    first = _base_checkpoint_identity(tmp_path, model)
    cache = snapshot / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "model.safetensors.metadata").write_text("mutable", encoding="utf-8")
    after_cache_change = _base_checkpoint_identity(tmp_path, model)

    assert first == after_cache_change
    assert first["kind"] == "pinned_hf_snapshot"
    assert first["source"] == "stub/base"
    assert first["revision"] == "a" * 40
    assert first["weight_files"] == ["model.safetensors"]
    assert len(first["sha256"]) == 64
    assert first["artifacts"][-1]["path"] == "model.safetensors"

    (snapshot / "model.safetensors").write_bytes(b"changed-weight-bytes")
    mutated = _base_checkpoint_identity(tmp_path, model)

    assert mutated["sha256"] != first["sha256"]
    assert mutated["artifacts"][-1]["sha256"] != first["artifacts"][-1]["sha256"]


def test_base_snapshot_accepts_exact_sharded_safetensors_index(tmp_path):
    snapshot = _write_base_snapshot(
        tmp_path / "sharded",
        weights={
            "model-00001-of-00002.safetensors": b"first-shard",
            "model-00002-of-00002.safetensors": b"second-shard",
        },
        weight_map={
            "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
            "model.layers.0.weight": "model-00001-of-00002.safetensors",
            "lm_head.weight": "model-00002-of-00002.safetensors",
        },
    )
    model = _StubVLM([], [], snapshot)

    identity = _base_checkpoint_identity(tmp_path, model)

    assert identity["format"] == "safetensors_sharded"
    assert identity["weight_files"] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    assert [artifact["path"] for artifact in identity["artifacts"]] == [
        "config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]


@pytest.mark.parametrize(
    ("repo", "revision"),
    [("wrong/base", "a" * 40), ("stub/base", "b" * 40)],
)
def test_base_snapshot_rejects_marker_mismatch(tmp_path, repo, revision):
    snapshot = _write_base_snapshot(tmp_path / "snapshot", repo=repo, revision=revision)
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="marker does not match"):
        _base_checkpoint_identity(tmp_path, model)


@pytest.mark.parametrize("failure", ["missing", "unreferenced"])
def test_base_snapshot_index_must_bind_exact_shard_set(tmp_path, failure):
    weights = {
        "model-00001-of-00002.safetensors": b"first-shard",
        "model-00002-of-00002.safetensors": b"second-shard",
    }
    weight_map = {
        "model.layers.0.weight": "model-00001-of-00002.safetensors",
        "model.layers.1.weight": "model-00002-of-00002.safetensors",
    }
    if failure == "missing":
        del weights["model-00002-of-00002.safetensors"]
    else:
        weights["model-00003-of-00003.safetensors"] = b"unreferenced"
    snapshot = _write_base_snapshot(
        tmp_path / "snapshot", weights=weights, weight_map=weight_map
    )
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="does not exactly bind"):
        _base_checkpoint_identity(tmp_path, model)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../outside.safetensors",
        "nested/shard.safetensors",
        "nested\\shard.safetensors",
        "C:escape.safetensors",
    ],
)
def test_base_snapshot_rejects_unsafe_index_paths(tmp_path, unsafe_name):
    snapshot = _write_base_snapshot(
        tmp_path / "snapshot",
        weights={"model-00001-of-00001.safetensors": b"only-shard"},
        weight_map={"model.weight": unsafe_name},
    )
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="unsafe shard path"):
        _base_checkpoint_identity(tmp_path, model)


@pytest.mark.parametrize(
    "index_payload",
    [
        "{",
        (
            '{"metadata":{},"weight_map":'
            '{"model.weight":"model-00001-of-00001.safetensors",'
            '"model.weight":"model-00001-of-00001.safetensors"}}'
        ),
    ],
)
def test_base_snapshot_rejects_malformed_or_duplicate_index_json(
    tmp_path, index_payload
):
    snapshot = _write_base_snapshot(
        tmp_path / "snapshot",
        weights={"model-00001-of-00001.safetensors": b"only-shard"},
        weight_map={"model.weight": "model-00001-of-00001.safetensors"},
    )
    (snapshot / "model.safetensors.index.json").write_text(
        index_payload, encoding="utf-8"
    )
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="malformed or duplicate JSON"):
        _base_checkpoint_identity(tmp_path, model)


def test_base_snapshot_rejects_symlinked_weight(tmp_path):
    snapshot = _write_base_snapshot(tmp_path / "snapshot")
    weight = snapshot / "model.safetensors"
    external = tmp_path / "external.safetensors"
    external.write_bytes(weight.read_bytes())
    weight.unlink()
    try:
        weight.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"Symlinks are unavailable in this test environment: {exc}")
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="must not be a symlink"):
        _base_checkpoint_identity(tmp_path, model)


@pytest.mark.parametrize("extra_name", ["pytorch_model.bin", "other.safetensors"])
def test_base_snapshot_rejects_ambiguous_weight_payloads(tmp_path, extra_name):
    snapshot = _write_base_snapshot(tmp_path / "snapshot")
    (snapshot / extra_name).write_bytes(b"ambiguous")
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="ambiguous|exactly"):
        _base_checkpoint_identity(tmp_path, model)


@pytest.mark.parametrize("sharded", [False, True])
def test_base_snapshot_rejects_empty_weight_payloads(tmp_path, sharded):
    if sharded:
        weights = {"model-00001-of-00001.safetensors": b""}
        weight_map = {"model.weight": "model-00001-of-00001.safetensors"}
    else:
        weights = {"model.safetensors": b""}
        weight_map = None
    snapshot = _write_base_snapshot(
        tmp_path / "snapshot", weights=weights, weight_map=weight_map
    )
    model = _StubVLM([], [], snapshot)

    with pytest.raises(RuntimeError, match="must not be empty"):
        _base_checkpoint_identity(tmp_path, model)


def test_base_snapshot_rejects_weight_mutation_during_hashing(tmp_path, monkeypatch):
    model = _stub_model(tmp_path, [], [])
    weight = Path(model.processor.name_or_path) / "model.safetensors"

    def mutate_while_hashing(stream, digest):
        digest.update(stream.read(4))
        with weight.open("ab") as writer:
            writer.write(b"!")
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    monkeypatch.setattr(
        VLMConfidenceValidator,
        "_update_file_digest",
        staticmethod(mutate_while_hashing),
    )

    with pytest.raises(RuntimeError, match="changed while it was fingerprinted"):
        _base_checkpoint_identity(tmp_path, model)


def test_loaded_adapter_checkpoint_identity_also_binds_base_snapshot(tmp_path):
    model = _stub_model(tmp_path, [], [])
    snapshot = Path(model.processor.name_or_path)
    checkpoint = tmp_path / "adapter-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "adapter_config.json").write_text(
        '{"peft_type":"LORA"}\n', encoding="utf-8"
    )
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    (checkpoint / "preprocessor_config.json").write_text("{}\n", encoding="utf-8")
    model._checkpoint_dir = checkpoint
    model.processor.name_or_path = str(checkpoint)
    model.model.config = SimpleNamespace(_name_or_path=str(snapshot))

    first = _base_checkpoint_identity(tmp_path, model)
    (snapshot / "model.safetensors").write_bytes(b"different-base")
    second = _base_checkpoint_identity(tmp_path, model)

    assert first["kind"] == "adapter_checkpoint_with_base_snapshot"
    assert first["adapter"] == second["adapter"]
    assert first["base"]["sha256"] != second["base"]["sha256"]
    assert first["sha256"] != second["sha256"]


def test_full_checkpoint_directory_identity_remains_self_contained(tmp_path):
    model = _stub_model(tmp_path, [], [])
    checkpoint = tmp_path / "full-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}\n", encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"full-checkpoint")
    model._checkpoint_dir = checkpoint

    identity = _base_checkpoint_identity(tmp_path, model)

    assert identity["kind"] == "checkpoint_directory"
    assert identity["files"] == 2
    assert len(identity["sha256"]) == 64


def test_peft_config_changes_live_checkpoint_identity(tmp_path):
    configs = []
    for alpha in (16, 32):
        model = _stub_model(tmp_path, [], [])
        validator = _Harness(
            model,
            _config(tmp_path, save_dir=tmp_path / f"alpha-{alpha}"),
            [],
            torch.zeros((0, 1, 5)),
            generation_model=_LivePeftModel(model.model, lora_alpha=alpha),
        )
        validator._actual_imgsz = (100, 100)
        validator.class_names = ["cat"]
        validator._coco_label_to_category_id = None
        configs.append(validator._build_benchmark_config())

    first_checkpoint = configs[0]["checkpoint"]
    second_checkpoint = configs[1]["checkpoint"]
    assert (
        first_checkpoint["trainable_state"]["sha256"]
        == second_checkpoint["trainable_state"]["sha256"]
    )
    assert first_checkpoint["sha256"] != second_checkpoint["sha256"]


def test_frozen_base_changes_live_checkpoint_identity(tmp_path):
    configs = []
    for base_value in (0.125, 9.0):
        model = _stub_model(tmp_path, [], [])
        with torch.no_grad():
            model.model.adapter.fill_(base_value)
        validator = _Harness(
            model,
            _config(tmp_path, save_dir=tmp_path / f"base-{base_value}"),
            [],
            torch.zeros((0, 1, 5)),
            generation_model=_LivePeftModel(model.model),
        )
        validator._actual_imgsz = (100, 100)
        validator.class_names = ["cat"]
        validator._coco_label_to_category_id = None
        configs.append(validator._build_benchmark_config()["checkpoint"])

    assert (
        configs[0]["trainable_state"]["sha256"]
        == configs[1]["trainable_state"]["sha256"]
    )
    assert (
        configs[0]["parameter_state"]["sha256"]
        != configs[1]["parameter_state"]["sha256"]
    )
    assert configs[0]["sha256"] != configs[1]["sha256"]


def test_runtime_lora_scaling_changes_live_checkpoint_identity(tmp_path):
    configs = []
    for scale in (2.0, 1.0):
        model = _stub_model(tmp_path, [], [])
        validator = _Harness(
            model,
            _config(tmp_path, save_dir=tmp_path / f"scale-{scale}"),
            [],
            torch.zeros((0, 1, 5)),
            generation_model=_LivePeftModel(model.model, runtime_scale=scale),
        )
        validator._actual_imgsz = (100, 100)
        validator.class_names = ["cat"]
        validator._coco_label_to_category_id = None
        configs.append(validator._build_benchmark_config()["checkpoint"])

    assert (
        configs[0]["parameter_state"]["sha256"]
        == configs[1]["parameter_state"]["sha256"]
    )
    assert configs[0]["sha256"] != configs[1]["sha256"]


def test_constant_view_must_preserve_geometry_order_and_fallback_scores(tmp_path):
    model = _stub_model(tmp_path, [], [])
    validator = VLMConfidenceValidator(model, _config(tmp_path))
    reversed_geometry = _variants([[10, 10, 20, 20], [30, 30, 40, 40]], [0.9, 0.1])
    reversed_geometry.constant = _view([[30, 30, 40, 40], [10, 10, 20, 20]], [1.0, 1.0])

    with pytest.raises(RuntimeError, match="geometry order"):
        validator._score_independent_prediction_records(reversed_geometry, "1")

    malformed_fallback = _variants([[10, 10, 20, 20]], [0.2], available=False)
    malformed_fallback.candidate = _view([[10, 10, 20, 20]], [0.2])
    with pytest.raises(RuntimeError, match="exact constant-score fallback"):
        validator._score_independent_prediction_records(malformed_fallback, "1")


def test_evaluator_ground_truth_manifest_includes_crowd_semantics(tmp_path):
    model = _stub_model(tmp_path, [], [])
    validator = VLMConfidenceValidator(model, _config(tmp_path))
    base_annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 0,
        "bbox": [0, 0, 10, 10],
        "area": 100,
    }

    def manifest(iscrowd):
        validator._gt_coco_api = SimpleNamespace(
            imgs={1: {"id": 1, "width": 100, "height": 100}},
            cats={0: {"id": 0, "name": "cat"}},
            anns={1: {**base_annotation, "iscrowd": iscrowd}},
        )
        return validator._evaluator_ground_truth_manifest()

    assert manifest(0) != manifest(1)
    VLMConfidenceValidator._require_plain_ordering_ground_truth(manifest(0))
    with pytest.raises(NotImplementedError, match="crowd/ignore"):
        VLMConfidenceValidator._require_plain_ordering_ground_truth(manifest(1))


def test_crowd_annotations_are_refused_before_generation(tmp_path):
    path = tmp_path / "crowd.jpg"
    path.write_bytes(b"offline")
    targets = torch.zeros((1, 1, 5), dtype=torch.float32)
    targets[0, 0] = torch.tensor([10, 10, 30, 30, 0])
    model = _stub_model(tmp_path, [path], [_variants([], [])])

    class _CrowdHarness(_Harness):
        def _init_metrics(self):
            super()._init_metrics()
            self._gt_coco_api.anns[1]["iscrowd"] = 1

    validator = _CrowdHarness(model, _config(tmp_path), [path], targets)

    with pytest.raises(NotImplementedError, match="crowd/ignore"):
        validator.run()
    assert model.forward_count == 0


def test_validator_remains_internal():
    import libreyolo.validation as validation

    assert not hasattr(validation, "VLMConfidenceValidator")
