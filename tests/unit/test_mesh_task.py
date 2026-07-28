"""Unit tests for the body-mesh task contract.

Everything here runs on fabricated payloads: the task plumbing, the result
container, the camera conversions and the metrics are all exercised without a
trained model, so they stay meaningful regardless of which mesh families ship.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from libreyolo.models.sam3dbody.camera import (
    crop_cam_to_full_image,
    default_focal_length,
    perspective_project,
)
from libreyolo.tasks import (
    TASKS,
    detect_task_suffix,
    normalize_task,
    resolve_task,
    task_to_suffix,
)
from libreyolo.utils.results import Boxes, Meshes, Results
from libreyolo.validation.mesh_metrics import (
    mesh_metrics,
    mpjpe,
    pa_mpjpe,
    procrustes_align,
    pve,
)


pytestmark = pytest.mark.unit

N_BETAS = 45
N_BODY_POSE = 130


def make_meshes(n: int = 2, with_geometry: bool = True) -> Meshes:
    """Build a fabricated mesh payload shaped like an MHR prediction."""
    geometry = {}
    if with_geometry:
        geometry = {
            "vertices": torch.randn(n, 64, 3),
            "faces": torch.randint(0, 64, (30, 3)),
            "joints3d": torch.randn(n, 70, 3),
            "joints2d": torch.rand(n, 70, 2) * 100,
        }
    return Meshes(
        torch.randn(n, 3),
        torch.randn(n, N_BODY_POSE),
        torch.randn(n, N_BETAS),
        torch.randn(n, 3),
        body_model="mhr",
        conf=torch.rand(n),
        focal_length=torch.full((n,), 1400.0),
        extras={"scale": torch.randn(n, 28)},
        orig_shape=(200, 300),
        **geometry,
    )


def make_results(n: int = 2) -> Results:
    boxes = Boxes(torch.rand(n, 4) * 100, torch.rand(n), torch.zeros(n))
    return Results(
        boxes=boxes,
        orig_shape=(200, 300),
        names={0: "person"},
        meshes=make_meshes(n),
    )


class TestTaskRegistration:
    def test_mesh_is_a_task(self):
        assert "mesh" in TASKS

    @pytest.mark.parametrize(
        "alias", ["mesh", "body-mesh", "body_mesh", "hmr", "human-mesh-recovery"]
    )
    def test_aliases_normalize(self, alias):
        assert normalize_task(alias) == "mesh"

    def test_suffix_roundtrip(self):
        assert task_to_suffix("mesh") == "mesh"
        assert detect_task_suffix("LibreSAM3DBodyd3-mesh.pt") == "mesh"

    def test_detect_suffix_ignores_unrelated_names(self):
        assert detect_task_suffix("LibreYOLO9s.pt") is None

    def test_resolve_task_precedence(self):
        # Explicit task beats the filename.
        assert (
            resolve_task(
                explicit_task="mesh",
                filename_task="pose",
                supported_tasks=("mesh", "pose"),
            )
            == "mesh"
        )
        # Checkpoint metadata beats the filename.
        assert (
            resolve_task(
                checkpoint_task="mesh",
                filename_task="detect",
                supported_tasks=("mesh", "detect"),
            )
            == "mesh"
        )

    def test_unsupported_task_rejected(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_task(explicit_task="mesh", supported_tasks=("detect",))

    def test_smpl_is_not_silently_aliased(self):
        # Nothing shipped here is SMPL, and quietly accepting the name would
        # imply an interoperability the payload does not provide.
        with pytest.raises(ValueError):
            normalize_task("smpl")


class TestMeshesPayload:
    def test_shapes_and_counts(self):
        m = make_meshes(3)
        assert len(m) == 3
        assert m.num_betas == N_BETAS
        assert m.num_vertices == 64
        assert m.num_joints == 70
        assert m.body_model == "mhr"
        assert m.has_vertices

    def test_params_bundle_includes_extras(self):
        params = make_meshes().params
        assert {"global_orient", "body_pose", "betas", "transl"} <= set(params)
        assert "scale" in params

    def test_slicing_selects_one_person_but_keeps_topology(self):
        m = make_meshes(3)
        one = m[0]
        assert len(one) == 1
        assert one.global_orient.shape == (1, 3)
        assert one.joints2d.shape == (1, 70, 2)
        # Face connectivity is shared by everyone in the image and must not be
        # sliced along with the per-person rows.
        assert one.faces.shape == m.faces.shape
        assert one.extras["scale"].shape == (1, 28)

    def test_device_and_dtype_moves_cover_topology(self):
        m = make_meshes(2).numpy()
        assert isinstance(m.vertices, np.ndarray)
        assert isinstance(m.faces, np.ndarray)
        assert isinstance(m.extras["scale"], np.ndarray)

    def test_cpu_roundtrip_preserves_values(self):
        m = make_meshes(2)
        assert torch.allclose(m.cpu().betas, m.betas)

    def test_parameters_only_payload_is_valid(self):
        # A model may emit parameters without decoding geometry; that must
        # remain representable rather than forcing empty vertex arrays.
        m = make_meshes(2, with_geometry=False)
        assert not m.has_vertices
        assert m.num_vertices == 0
        assert len(m) == 2

    def test_save_obj_writes_valid_wavefront(self, tmp_path):
        m = make_meshes(2)
        out = tmp_path / "person.obj"
        m.save_obj(out, index=1)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert sum(1 for line in lines if line.startswith("v ")) == 64
        assert sum(1 for line in lines if line.startswith("f ")) == 30
        # OBJ indices are 1-based, so no face may reference vertex 0.
        for line in lines:
            if line.startswith("f "):
                assert all(int(i) >= 1 for i in line.split()[1:])

    def test_save_obj_without_geometry_is_a_clear_error(self, tmp_path):
        m = make_meshes(1, with_geometry=False)
        with pytest.raises(ValueError, match="no mesh geometry"):
            m.save_obj(tmp_path / "x.obj")

    def test_save_obj_rejects_out_of_range_index(self, tmp_path):
        with pytest.raises(IndexError):
            make_meshes(2).save_obj(tmp_path / "x.obj", index=5)


class TestResultsIntegration:
    def test_results_carries_meshes(self):
        r = make_results(2)
        assert r.meshes is not None
        assert len(r) == 2

    def test_results_slicing_keeps_rows_aligned(self):
        r = make_results(3)
        one = r[1]
        assert len(one.boxes) == 1
        assert len(one.meshes) == 1

    def test_summary_reports_parameters_but_not_vertices(self):
        row = make_results(2).summary()[0]
        assert "mesh" in row
        mesh_row = row["mesh"]
        assert mesh_row["body_model"] == "mhr"
        assert len(mesh_row["global_orient"]) == 3
        assert len(mesh_row["betas"]) == N_BETAS
        assert mesh_row["num_vertices"] == 64
        # Tens of thousands of coordinates per person do not belong in JSON.
        assert "vertices" not in mesh_row

    def test_summary_normalizes_joints(self):
        r = make_results(1)
        raw = r.summary()[0]["mesh"]["joints2d"]
        norm = r.summary(normalize=True)[0]["mesh"]["joints2d"]
        assert max(norm["x"]) <= 1.0
        assert max(raw["x"]) > 1.0

    def test_to_json_serializes(self):
        import json

        parsed = json.loads(make_results(2).to_json())
        assert parsed[0]["mesh"]["body_model"] == "mhr"

    def test_repr_mentions_meshes(self):
        assert "meshes=" in repr(make_results(1))


class TestWrapResults:
    def _wrap(self, payload):
        from libreyolo.models.base.inference import _build_meshes

        return _build_meshes(payload, (200, 300))

    def test_builds_from_minimal_payload(self):
        m = self._wrap(
            {
                "body_model": "mhr",
                "global_orient": np.zeros((2, 3), dtype=np.float32),
                "body_pose": np.zeros((2, N_BODY_POSE), dtype=np.float32),
                "betas": np.zeros((2, N_BETAS), dtype=np.float32),
                "transl": np.zeros((2, 3), dtype=np.float32),
            }
        )
        assert len(m) == 2
        assert m.orig_shape == (200, 300)

    def test_missing_parameters_are_rejected(self):
        with pytest.raises(ValueError, match="missing"):
            self._wrap({"body_model": "mhr", "global_orient": np.zeros((1, 3))})

    def test_missing_body_model_is_rejected(self):
        with pytest.raises(ValueError, match="body_model"):
            self._wrap(
                {
                    "global_orient": np.zeros((1, 3)),
                    "body_pose": np.zeros((1, N_BODY_POSE)),
                    "betas": np.zeros((1, N_BETAS)),
                    "transl": np.zeros((1, 3)),
                }
            )

    def test_full_wrap_pairs_meshes_with_person_boxes(self):
        """The runner must return meshes row-aligned with their boxes."""
        from libreyolo.models.base.inference import InferenceRunner

        class DummyModel:
            names = {0: "person"}
            task = "mesh"

        runner = InferenceRunner(DummyModel())
        result = runner._wrap_results(
            {
                "boxes": np.array([[10.0, 10.0, 50.0, 90.0], [60.0, 5.0, 95.0, 99.0]]),
                "scores": np.array([0.9, 0.8]),
                "classes": np.array([0.0, 0.0]),
                "meshes": {
                    "body_model": "mhr",
                    "global_orient": np.zeros((2, 3), dtype=np.float32),
                    "body_pose": np.zeros((2, N_BODY_POSE), dtype=np.float32),
                    "betas": np.zeros((2, N_BETAS), dtype=np.float32),
                    "transl": np.zeros((2, 3), dtype=np.float32),
                },
            },
            (300, 200),
            None,
            None,
        )
        assert result.meshes is not None
        assert len(result.boxes) == len(result.meshes) == 2
        assert result.orig_shape == (200, 300)
        assert result.names == {0: "person"}

    def test_wrap_without_boxes_still_returns_meshes(self):
        from libreyolo.models.base.inference import InferenceRunner

        class DummyModel:
            names = {0: "person"}
            task = "mesh"

        result = InferenceRunner(DummyModel())._wrap_results(
            {
                "meshes": {
                    "body_model": "mhr",
                    "global_orient": np.zeros((1, 3), dtype=np.float32),
                    "body_pose": np.zeros((1, N_BODY_POSE), dtype=np.float32),
                    "betas": np.zeros((1, N_BETAS), dtype=np.float32),
                    "transl": np.zeros((1, 3), dtype=np.float32),
                }
            },
            (300, 200),
            None,
            None,
        )
        assert result.boxes is None
        assert len(result.meshes) == 1

    def test_unknown_keys_are_kept_as_extras(self):
        m = self._wrap(
            {
                "body_model": "mhr",
                "global_orient": np.zeros((1, 3)),
                "body_pose": np.zeros((1, N_BODY_POSE)),
                "betas": np.zeros((1, N_BETAS)),
                "transl": np.zeros((1, 3)),
                "hand_pose": np.zeros((1, 108)),
            }
        )
        assert "hand_pose" in m.extras


class TestCamera:
    def test_center_point_projects_to_principal_point(self):
        pixels = perspective_project(
            torch.tensor([[[0.0, 0.0, 5.0]]]), 1000.0, image_size=(480, 640)
        )
        assert torch.allclose(pixels.flatten(), torch.tensor([320.0, 240.0]))

    def test_known_offset(self):
        # x = 1 m at z = 5 m with f = 1000 px lands 200 px right of center.
        pixels = perspective_project(
            torch.tensor([[[1.0, 0.0, 5.0]]]), 1000.0, image_size=(480, 640)
        )
        assert torch.allclose(pixels.flatten(), torch.tensor([520.0, 240.0]))

    def test_translation_argument_matches_pre_adding_it(self):
        points = torch.randn(2, 5, 3) + torch.tensor([0.0, 0.0, 6.0])
        transl = torch.randn(2, 3) * 0.1
        a = perspective_project(points + transl[:, None, :], 900.0, image_size=(480, 640))
        b = perspective_project(points, 900.0, image_size=(480, 640), translation=transl)
        assert torch.allclose(a, b, atol=1e-5)

    def test_points_behind_camera_stay_finite(self):
        pixels = perspective_project(
            torch.tensor([[[0.1, 0.1, -2.0]]]), 1000.0, image_size=(480, 640)
        )
        assert torch.isfinite(pixels).all()

    def test_requires_principal_point_or_image_size(self):
        with pytest.raises(ValueError, match="principal_point or image_size"):
            perspective_project(torch.zeros(1, 1, 3), 1000.0)

    def test_rejects_wrong_rank(self):
        with pytest.raises(ValueError, match=r"\(N, P, 3\)"):
            perspective_project(torch.zeros(4, 3), 1000.0, image_size=(10, 10))

    def test_crop_camera_depth_matches_closed_form(self):
        # z = 2f / (box_size * scale)
        transl = crop_cam_to_full_image(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[320.0, 240.0]]),
            torch.tensor([200.0]),
            (480, 640),
            1000.0,
        )
        assert transl[0, 2] == pytest.approx(10.0)
        assert transl[0, 0] == pytest.approx(0.0)

    def test_off_center_box_shifts_translation(self):
        # A box 100 px right of center at 10 m with f = 1000 sits 1 m right.
        transl = crop_cam_to_full_image(
            torch.tensor([[1.0, 0.0, 0.0]]),
            torch.tensor([[420.0, 240.0]]),
            torch.tensor([200.0]),
            (480, 640),
            1000.0,
        )
        assert transl[0, 0] == pytest.approx(1.0)

    def test_default_focal_length_is_the_diagonal(self):
        assert default_focal_length(480, 640) == pytest.approx(800.0)


class TestMeshMetrics:
    def test_identical_inputs_score_zero(self):
        joints = torch.randn(4, 17, 3)
        assert float(mpjpe(joints, joints)) == pytest.approx(0.0, abs=1e-6)
        assert float(pa_mpjpe(joints, joints)) == pytest.approx(0.0, abs=1e-6)

    def test_root_alignment_cancels_translation(self):
        joints = torch.randn(4, 17, 3)
        shifted = joints + torch.tensor([1.0, -2.0, 3.0])
        assert float(mpjpe(shifted, joints)) == pytest.approx(0.0, abs=1e-5)

    def test_known_error_value(self):
        joints = torch.randn(2, 17, 3)
        moved = joints.clone()
        moved[:, 1:, 0] += 0.1
        # 16 of 17 joints move by exactly 0.1 after root alignment.
        assert float(mpjpe(moved, joints)) == pytest.approx(0.1 * 16 / 17, abs=1e-6)

    def test_procrustes_removes_rotation_scale_and_translation(self):
        joints = torch.randn(3, 17, 3)
        angle = 0.7
        rotation = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed = (joints @ rotation.T) * 2.5 + torch.tensor([5.0, -3.0, 1.0])
        assert float(pa_mpjpe(transformed, joints)) == pytest.approx(0.0, abs=1e-5)
        # The un-aligned metric should be large, proving the test is not
        # trivially passing because the transform did nothing.
        assert float(mpjpe(transformed, joints)) > 1.0

    def test_procrustes_does_not_align_away_a_reflection(self):
        # A mirrored body is a different pose, so a reflection must still cost.
        joints = torch.randn(2, 17, 3)
        reflected = joints.clone()
        reflected[..., 0] *= -1
        assert float(pa_mpjpe(reflected, joints)) > 0.1

    def test_procrustes_output_shape(self):
        joints = torch.randn(2, 17, 3)
        assert procrustes_align(joints + 1.0, joints).shape == joints.shape

    def test_mask_restricts_the_average(self):
        joints = torch.randn(2, 17, 3)
        moved = joints.clone()
        moved[:, 1:, 0] += 0.1
        mask = torch.zeros(2, 17, dtype=torch.bool)
        mask[:, 0] = True  # only the (unmoved) root counts
        assert float(mpjpe(moved, joints, mask=mask)) == pytest.approx(0.0, abs=1e-6)

    def test_empty_mask_does_not_divide_by_zero(self):
        joints = torch.randn(1, 17, 3)
        mask = torch.zeros(1, 17, dtype=torch.bool)
        assert float(mpjpe(joints + 1.0, joints, mask=mask)) == 0.0

    def test_pve_cancels_translation(self):
        verts = torch.randn(2, 200, 3)
        assert float(pve(verts + torch.tensor([1.0, 2.0, 3.0]), verts)) == pytest.approx(
            0.0, abs=1e-5
        )

    def test_shape_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="must match"):
            mpjpe(torch.randn(2, 17, 3), torch.randn(2, 16, 3))

    def test_metric_bundle_is_reported_in_millimeters(self):
        joints = torch.randn(2, 17, 3)
        moved = joints.clone()
        moved[:, 1:, 0] += 0.1  # 0.1 m -> ~94 mm after root alignment
        metrics = mesh_metrics(moved, joints)
        assert metrics["metrics/mpjpe"] == pytest.approx(0.1 * 16 / 17 * 1000, abs=1e-3)
        assert "metrics/pve" not in metrics

    def test_metric_bundle_includes_pve_when_vertices_given(self):
        joints = torch.randn(2, 17, 3)
        verts = torch.randn(2, 50, 3)
        metrics = mesh_metrics(joints, joints, verts, verts)
        assert metrics["metrics/pve"] == pytest.approx(0.0, abs=1e-3)


class TestDrawing:
    def test_draw_mesh_returns_an_image_of_the_same_size(self):
        from PIL import Image

        from libreyolo.utils.drawing import draw_mesh

        img = Image.new("RGB", (300, 200))
        out = draw_mesh(
            img,
            joints2d=np.random.rand(2, 70, 2) * 100,
            vertices2d=np.random.rand(2, 500, 2) * 100,
        )
        assert out.size == img.size

    def test_draw_mesh_handles_missing_inputs(self):
        from PIL import Image

        from libreyolo.utils.drawing import draw_mesh

        img = Image.new("RGB", (64, 64))
        assert draw_mesh(img).size == img.size

    def test_vertex_cloud_is_decimated(self):
        from PIL import Image

        from libreyolo.utils.drawing import draw_mesh

        # 18k vertices per person is the real MHR count; drawing must stay
        # bounded rather than scaling with it.
        img = Image.new("RGB", (128, 128))
        out = draw_mesh(img, vertices2d=np.random.rand(1, 18439, 2) * 128,
                        max_vertices=100)
        assert out.size == img.size
