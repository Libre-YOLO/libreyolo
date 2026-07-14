"""Neural network architecture for YOLOv9 end-to-end (NMS-free).

Builds on the LibreYOLO9 head (see ``..yolo9.nn``, ported from
MultimediaTechLab/YOLO, MIT License). The dual one-to-many / one-to-one
branch scheme implements the dual-assignment training strategy from the
YOLOv10 paper (Wang et al., 2024, arXiv:2405.14458).
"""

import torch

from ..yolo9.nn import DDetect, LibreYOLO9Model, YOLO9_CONFIGS


class YOLO9E2EDetect(DDetect):
    """YOLOv9 detect head with a one-to-one branch for NMS-free inference.

    Training: both a dense one-to-many branch (standard TAL, topk=10) and a
    one-to-one branch (topk=1) are run in parallel, with gradients blocked on
    the backbone features fed to the one-to-one branch (detach). The dual-
    branch loss sums both sets of losses.

    Inference: only the one-to-one branch is active; top-K selection replaces
    NMS so the model can be used without any post-processing graph ops.

    Color space / normalization: RGB 0–1, same as standard YOLOv9.
    """

    def __init__(self, nc=80, ch=(), reg_max=16, stride=(), use_group=True):
        super().__init__(
            nc=nc, ch=ch, reg_max=reg_max, stride=stride, use_group=use_group
        )
        self.one2one_cv2 = self._build_box_towers(
            ch,
            self._box_hidden_channels,
            self._box_output_channels,
            self._box_groups,
        )
        self.one2one_cv3 = self._build_class_towers(
            ch, self._class_hidden_channels, nc
        )
        self._copy_branch_weights(self.cv2, self.one2one_cv2)
        self._copy_branch_weights(self.cv3, self.one2one_cv3)
        self._init_one2one_bias()

    @staticmethod
    def _copy_branch_weights(source, target):
        """Start the auxiliary branch from the same initialized weights."""
        target.load_state_dict(source.state_dict())

    def _init_one2one_bias(self):
        """Initialize the one-to-one branch with the same detection priors as
        the dense branch (box 1.0, class -10 — see ``DDetect._init_bias``)."""
        for box_tower, class_tower in zip(self.one2one_cv2, self.one2one_cv3):
            box_tower[-1].bias.data.fill_(1.0)
            class_tower[-1].bias.data.fill_(-10.0)

    def _get_loss_fn(self, device):
        """Lazily initialize the dual-branch loss."""
        if self._loss_fn is None:
            from .loss import YOLO9E2ELoss

            self._loss_fn = YOLO9E2ELoss(
                num_classes=self.nc,
                reg_max=self.reg_max,
                strides=self.stride.tolist(),
                image_size=None,
                device=device,
            )
        return self._loss_fn

    def _forward_head(self, x, cv2, cv3):
        outputs = []
        for i in range(self.nl):
            outputs.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return outputs

    def _inference(self, x):
        return self._decode_inference(x), x

    def forward(self, x, targets=None, img_size=None):
        """Run dual-branch training or single one-to-one inference."""
        if self.training:
            dense_outputs = self._forward_head(x, self.cv2, self.cv3)
            exclusive_outputs = self._forward_head(
                [xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3
            )

            if targets is not None:
                loss_fn = self._get_loss_fn(x[0].device)
                if img_size is not None:
                    loss_fn.update_anchors(list(img_size))
                return loss_fn(dense_outputs, exclusive_outputs, targets)
            return {"one2many": dense_outputs, "one2one": exclusive_outputs}

        # Inference: use only the one-to-one branch
        exclusive_outputs = self._forward_head(x, self.one2one_cv2, self.one2one_cv3)
        return self._inference(exclusive_outputs)


class LibreYOLO9E2EModel(LibreYOLO9Model):
    """YOLOv9 model with a one-to-one head for NMS-free inference."""

    def __init__(self, config="c", reg_max=16, nb_classes=80, img_size=640):
        super().__init__(
            config=config, reg_max=reg_max, nb_classes=nb_classes, img_size=img_size
        )
        head_channels = YOLO9_CONFIGS[config]["head_channels"]
        self.head = YOLO9E2EDetect(
            nc=nb_classes, ch=head_channels, reg_max=reg_max, stride=(8, 16, 32)
        )


__all__ = ["YOLO9E2EDetect", "LibreYOLO9E2EModel"]
