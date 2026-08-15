"""Offline tests for the internal VLM confidence validation harness."""

import hashlib
import json
import math
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


def _stub_model(tmp_path, paths, variants):
    processor_dir = tmp_path / "processor"
    processor_dir.mkdir(exist_ok=True)
    (processor_dir / "preprocessor_config.json").write_text(
        '{"processor_class":"OfflineStub"}\n', encoding="utf-8"
    )
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


def test_serial_gate_reports_coco_deltas_quality_and_coverage(tmp_path):
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    for path in paths:
        path.write_bytes(b"offline")
    variants = [
        _variants([[10, 10, 30, 30], [50, 50, 70, 70]], [0.9, 0.1]),
        _variants([[20, 20, 40, 40]], [0.7], available=False),
    ]
    model = _stub_model(tmp_path, paths, variants)
    model.set_classes(["dog"])
    live_model = _LivePeftModel(model.model)
    validator = _Harness(
        model,
        _config(tmp_path),
        paths,
        _targets(),
        generation_model=live_model,
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
        "backend": "offline-stub",
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


@pytest.mark.parametrize(
    "option",
    ["augment", "cuda_graph"],
)
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


def test_real_yolo_loader_writer_and_report_reader_round_trip(tmp_path):
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
            }
        ),
        encoding="utf-8",
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


@pytest.mark.parametrize(
    ("image_size", "raw_bbox", "clean_box", "names"),
    [
        ((100, 100), (-5, 10, 20, 20), (0, 10, 20, 30), ("cat", "dog")),
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
