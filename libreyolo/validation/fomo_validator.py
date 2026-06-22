"""FOMO validator combining standard point validation and grid validation."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from ..models.fomo.loss import FOMOLoss
from ..models.fomo.utils import decode_points_from_logits
from .point_validator import PointValidator

__all__ = ["FOMOValidator"]


class FOMOValidator(PointValidator):
    """FOMO-specific validator combining standard point validation and grid validation."""

    def __init__(
        self,
        model: Any,
        config: Optional[Any] = None,
        grid_size: int | None = None,
        fg_weight: float = 100.0,
        conf_thresholds: tuple[float, ...] = (0.25, 0.35, 0.50, 0.65, 0.80, 0.90),
        nms_radii: tuple[int, ...] = (1, 2),
        distance_tolerance: float = 1.5,
        **kwargs,
    ) -> None:
        super().__init__(model, config, **kwargs)
        self.grid_size = grid_size if grid_size is not None else self._infer_grid_size(model)
        self.fg_weight = fg_weight
        self.conf_thresholds = conf_thresholds
        self.nms_radii = nms_radii
        self.distance_tolerance = distance_tolerance

        self.val_loss_fn = FOMOLoss(
            num_classes=self.nc,
            fg_weight=self.fg_weight,
            device=self.device,
        ).to(self.device)
        self.val_loss_fn.eval()

    @staticmethod
    def _infer_grid_size(model: Any) -> int:
        input_size = None
        get_input_size = getattr(model, "_get_input_size", None)
        if callable(get_input_size):
            input_size = get_input_size()
        if input_size is None:
            inner_model = getattr(model, "model", model)
            input_size = getattr(inner_model, "imgsz", None)
        if input_size is None:
            return 12
        return int(input_size) // 8

    def _init_metrics(self) -> None:
        super()._init_metrics()
        self.last_logits: torch.Tensor | None = None
        self.grid_stats = {
            (float(threshold), int(nms_radius)): {
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "total_dist": 0.0,
            }
            for threshold in self.conf_thresholds
            for nms_radius in self.nms_radii
        }
        self.val_loss_total = 0.0
        self.batch_count = 0

    def _inference(self, images: torch.Tensor) -> Any:
        preds = super()._inference(images)
        self.last_logits = preds
        return preds

    def _update_grid_stats(
        self,
        decoded: list[torch.Tensor],
        targets_cpu: torch.Tensor,
        stats: Dict[str, float],
    ) -> None:
        batch_size = targets_cpu.shape[0]
        for b in range(batch_size):
            rows = decoded[b]

            fg_mask = targets_cpu[b] >= 1
            ys, xs = torch.where(fg_mask)
            if ys.numel():
                true_cls = targets_cpu[b][ys, xs] - 1
                trues_xy = torch.stack((xs, ys), dim=1).float().numpy()
                true_cls_np = true_cls.numpy()
            else:
                trues_xy = np.zeros((0, 2))
                true_cls_np = np.zeros(0, dtype=np.int64)

            if len(rows) > 0:
                preds_xy = rows[:, :2].numpy()
                preds_cls = (rows[:, 2].long() - 1).numpy()
            else:
                preds_xy = np.zeros((0, 2))
                preds_cls = np.zeros(0, dtype=np.int64)

            if len(preds_xy) == 0 and len(trues_xy) == 0:
                continue
            if len(preds_xy) == 0:
                stats["fn"] += len(trues_xy)
                continue
            if len(trues_xy) == 0:
                stats["fp"] += len(preds_xy)
                continue

            dist_mat = cdist(preds_xy, trues_xy)
            for pi in range(len(preds_cls)):
                for ti in range(len(true_cls_np)):
                    if preds_cls[pi] != true_cls_np[ti]:
                        dist_mat[pi, ti] = np.inf

            row_ind, col_ind = linear_sum_assignment(
                np.where(np.isfinite(dist_mat), dist_mat, 1e9)
            )
            matched_preds = set()
            matched_trues = set()
            for r, c in zip(row_ind, col_ind):
                d = dist_mat[r, c]
                if np.isfinite(d) and d <= self.distance_tolerance:
                    stats["tp"] += 1
                    stats["total_dist"] += d
                    matched_preds.add(r)
                    matched_trues.add(c)
            stats["fp"] += len(preds_xy) - len(matched_preds)
            stats["fn"] += len(trues_xy) - len(matched_trues)

    def _update_metrics(self, preds: Any, targets: Any, img_info: Any, img_ids: Any = None) -> None:
        super()._update_metrics(preds, targets, img_info, img_ids)

        logits = getattr(self, "last_logits", None)
        if logits is None:
            raise RuntimeError("FOMOValidator._update_metrics called before _inference")
        batch_size, _, grid_h, grid_w = logits.shape
        if grid_h != self.grid_size or grid_w != self.grid_size:
            raise ValueError(
                f"FOMO validator grid_size={self.grid_size} does not match "
                f"logit grid {grid_h}x{grid_w}"
            )
        grid_targets = torch.zeros(
            (batch_size, self.grid_size, self.grid_size),
            dtype=torch.long,
            device=self.device,
        )

        for b in range(batch_size):
            orig_h, orig_w = img_info[b]
            gt_row = targets[b]
            if isinstance(gt_row, torch.Tensor):
                gt_row = gt_row.cpu().numpy()
            xy_norm, classes = self._parse_gt_points(gt_row, orig_h, orig_w)

            for (xn, yn), cls in zip(xy_norm, classes):
                gx = int(xn * self.grid_size)
                gy = int(yn * self.grid_size)
                gx = min(max(gx, 0), self.grid_size - 1)
                gy = min(max(gy, 0), self.grid_size - 1)
                grid_targets[b, gy, gx] = int(cls) + 1

        with torch.no_grad():
            loss_out = self.val_loss_fn(logits, grid_targets)
            self.val_loss_total += float(loss_out["total_loss"].item())
            self.batch_count += 1

        logits_cpu = logits.detach().cpu()
        targets_cpu = grid_targets.detach().cpu()
        for threshold in self.conf_thresholds:
            for nms_radius in self.nms_radii:
                decoded = decode_points_from_logits(
                    logits_cpu,
                    conf_threshold=threshold,
                    nms_radius=nms_radius,
                )
                stats = self.grid_stats[(float(threshold), int(nms_radius))]
                self._update_grid_stats(decoded, targets_cpu, stats)

    def _compute_metrics(self) -> Dict[str, float]:
        metrics = super()._compute_metrics()

        avg_val_loss = self.val_loss_total / max(self.batch_count, 1)
        best_grid_res = None

        for threshold in self.conf_thresholds:
            for nms_radius in self.nms_radii:
                stats = self.grid_stats[(float(threshold), int(nms_radius))]
                total_tp = int(stats["tp"])
                total_fp = int(stats["fp"])
                total_fn = int(stats["fn"])
                total_dist = stats["total_dist"]

                prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
                rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
                mean_dist = total_dist / max(total_tp, 1)

                result = {
                    "threshold": float(threshold),
                    "nms_radius": int(nms_radius),
                    "precision": prec,
                    "recall": rec,
                    "f1": f1,
                    "mean_dist": mean_dist,
                    "tp": total_tp,
                    "fp": total_fp,
                    "fn": total_fn,
                }
                if best_grid_res is None or f1 > best_grid_res["f1"]:
                    best_grid_res = result

        metrics["metrics/val_loss"] = avg_val_loss
        if best_grid_res is not None:
            metrics.update({
                "metrics/grid_F1": best_grid_res["f1"],
                "metrics/grid_precision": best_grid_res["precision"],
                "metrics/grid_recall": best_grid_res["recall"],
                "metrics/grid_mean_distance": best_grid_res["mean_dist"],
                "metrics/grid_TP": float(best_grid_res["tp"]),
                "metrics/grid_FP": float(best_grid_res["fp"]),
                "metrics/grid_FN": float(best_grid_res["fn"]),
                "decode/threshold": best_grid_res["threshold"],
                "decode/nms_radius": float(best_grid_res["nms_radius"]),
            })

        return metrics
