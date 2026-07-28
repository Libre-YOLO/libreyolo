"""Integration checks for the MHR body model.

Marked ``external_data``: these need the ~700 MB Apache-2.0 MHR release asset,
so they are excluded from the fast unit run. Point ``LIBREYOLO_MHR_PATH`` at an
existing copy, or let the first run fetch it.

The assertions are physical rather than golden-value: a rest-pose body should
be roughly human-sized, translation should move it rigidly by exactly the
requested amount, and pose and identity parameters should each change the
geometry. Those catch the unit and convention mistakes that actually happen
here (centimeters vs meters, the decimeter translation factor, parameter blocks
concatenated in the wrong order) without pinning numbers that a legitimate
upstream asset update would change.
"""

from __future__ import annotations

import pytest
import torch

from libreyolo.models.sam3dbody.mhr_body import (
    MHRBodyModel,
    default_mhr_path,
    load_mhr_body_model,
)

pytestmark = pytest.mark.external_data


@pytest.fixture(scope="module")
def body_model() -> MHRBodyModel:
    path = default_mhr_path()
    if not path.exists():
        pytest.skip(
            f"MHR body model not present at {path}; "
            "set LIBREYOLO_MHR_PATH or run ensure_mhr_model()."
        )
    return load_mhr_body_model(path, device="cpu", download=False)


def zeros(batch: int) -> dict:
    return {
        "global_orient": torch.zeros(batch, 3),
        "body_pose": torch.zeros(batch, MHRBodyModel.NUM_BODY_POSE),
        "betas": torch.zeros(batch, MHRBodyModel.NUM_BETAS),
        "scales": torch.zeros(batch, MHRBodyModel.NUM_SCALES),
    }


def test_rest_pose_shapes(body_model):
    vertices, joints = body_model(**zeros(2))
    assert vertices.shape == (2, MHRBodyModel.NUM_VERTICES, 3)
    assert joints.shape == (2, MHRBodyModel.NUM_JOINTS, 3)


def test_rest_pose_is_human_sized(body_model):
    # Output must be meters, not the centimeters the rig works in.
    vertices, _ = body_model(**zeros(1))
    height = float(vertices[0, :, 1].max() - vertices[0, :, 1].min())
    assert 1.4 < height < 2.1, f"rest-pose body is {height:.2f} m tall"


def test_translation_moves_the_body_rigidly(body_model):
    # Verifies the decimeter scaling applied to translation on the way in.
    base, _ = body_model(**zeros(2))
    offset = torch.tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]])
    moved, _ = body_model(**zeros(2), transl=offset)
    assert torch.allclose(moved - base, offset[:, None, :], atol=1e-4)


def test_pose_parameters_change_geometry(body_model):
    base, _ = body_model(**zeros(2))
    params = zeros(2)
    params["body_pose"] = torch.randn(2, MHRBodyModel.NUM_BODY_POSE) * 0.1
    posed, _ = body_model(**params)
    assert (posed - base).abs().max() > 1e-4


def test_identity_parameters_change_geometry(body_model):
    base, _ = body_model(**zeros(2))
    params = zeros(2)
    params["betas"] = torch.randn(2, MHRBodyModel.NUM_BETAS) * 0.5
    reshaped, _ = body_model(**params)
    assert (reshaped - base).abs().max() > 1e-4


@pytest.mark.parametrize(
    "field,width",
    [("body_pose", 99), ("betas", 10), ("scales", 24)],
)
def test_wrong_parameter_widths_are_rejected(body_model, field, width):
    params = zeros(1)
    params[field] = torch.zeros(1, width)
    with pytest.raises(ValueError, match=field):
        body_model(**params)


def test_decoded_mesh_round_trips_through_the_result_payload(body_model):
    """The decoder output should drop straight into a Meshes slot."""
    from libreyolo.models.sam3dbody.camera import perspective_project
    from libreyolo.utils.results import Meshes

    params = zeros(1)
    transl = torch.tensor([[0.0, 0.0, 4.0]])
    vertices, joints = body_model(**params, transl=transl)
    joints2d = perspective_project(joints, 1000.0, image_size=(480, 640))

    meshes = Meshes(
        params["global_orient"],
        params["body_pose"],
        params["betas"],
        transl,
        body_model="mhr",
        vertices=vertices,
        joints3d=joints,
        joints2d=joints2d,
        orig_shape=(480, 640),
    )
    assert len(meshes) == 1
    assert meshes.num_vertices == MHRBodyModel.NUM_VERTICES
    assert meshes.num_joints == MHRBodyModel.NUM_JOINTS
    # A body 4 m in front of the camera should project inside the frame.
    assert (joints2d[..., 0] > 0).any() and (joints2d[..., 0] < 640).any()
