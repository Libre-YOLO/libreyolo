import pytest

from libreyolo.backends.openvino import OpenVINOBackend

pytestmark = pytest.mark.unit


class _FakeOutput:
    def __init__(self, shape):
        self._shape = shape

    @property
    def shape(self):
        if isinstance(self._shape, Exception):
            raise self._shape
        return self._shape


class _FakeModel:
    def __init__(self, shape):
        self.inputs = [_FakeOutput(shape)]


def test_openvino_backend_reads_static_input_imgsz():
    model = _FakeModel([1, 3, 384, 384])

    assert OpenVINOBackend._read_static_input_imgsz(model) == 384


def test_openvino_backend_reads_rectangular_static_input_imgsz():
    model = _FakeModel([1, 3, 320, 640])

    assert OpenVINOBackend._read_static_input_imgsz(model) == (320, 640)


def test_openvino_backend_ignores_dynamic_input_shape():
    model = _FakeModel(RuntimeError("to_shape was called on a dynamic shape"))

    assert OpenVINOBackend._read_static_input_imgsz(model) is None


@pytest.mark.parametrize(
    "shape",
    [
        [1, 3],
        [1, 3, -1, -1],
        [1, 3, "?", "?"],
    ],
)
def test_openvino_backend_ignores_non_static_input_imgsz(shape):
    model = _FakeModel(shape)

    assert OpenVINOBackend._read_static_input_imgsz(model) is None


def test_openvino_backend_reads_rectangular_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        "\n".join(
            [
                "model_family: yolo9",
                "imgsz: 640",
                "imgsz_h: 320",
                "imgsz_w: 640",
                "nc: 2",
            ]
        )
    )

    parsed = OpenVINOBackend._read_metadata(metadata_path)

    assert parsed[5] == (320, 640)


def test_openvino_backend_reads_runtime_metadata(tmp_path):
    metadata_path = tmp_path / "metadata.yaml"
    metadata_path.write_text(
        "\n".join(
            [
                "model_family: resnet",
                "task: classify",
                "crop_pct: 0.95",
                "interpolation: bicubic",
                "nms: 'true'",
            ]
        )
    )

    runtime_metadata = OpenVINOBackend._read_metadata(metadata_path)[9]

    assert runtime_metadata == {
        "crop_pct": pytest.approx(0.95),
        "interpolation": "bicubic",
        "embedded_nms": True,
    }


def test_openvino_backend_finds_named_raw_output():
    class _NamedOutput:
        def __init__(self, *names):
            self._names = names

        def get_names(self):
            return self._names

        def get_any_name(self):
            return self._names[0]

    backend = OpenVINOBackend.__new__(OpenVINOBackend)
    backend.compiled_model = type(
        "_CompiledModel",
        (),
        {"outputs": [_NamedOutput("output"), _NamedOutput("raw")]},
    )()

    assert backend._find_output_index("raw") == 1
