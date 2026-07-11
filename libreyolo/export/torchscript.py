"""
TorchScript export implementation.

Exports PyTorch models to TorchScript format via tracing.
"""

import json

import torch


def export_torchscript(
    nn_model, dummy, *, output_path: str, metadata: dict | None = None
) -> str:
    """Export a PyTorch model to TorchScript format.

    Args:
        nn_model: The PyTorch nn.Module to export.
        dummy: Dummy input tensor for tracing.
        output_path: Destination file path for the .torchscript file.
        metadata: Optional metadata dict to embed as an extra file.

    Returns:
        The output_path string.
    """
    # Several export wrappers cache shape-dependent anchors during their first
    # forward. A second trace-check therefore observes a different Python path
    # even though the recorded fixed-shape graph is correct. Runtime parity
    # tests validate the saved module directly, so avoid this false-negative
    # retrace here.
    traced = torch.jit.trace(nn_model, dummy, check_trace=False)
    extra_files = {}
    if metadata:
        extra_files["libreyolo_metadata.json"] = json.dumps(metadata)
    torch.jit.save(traced, output_path, _extra_files=extra_files)
    return output_path
