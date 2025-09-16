import torch
from torchmetrics import Metric

class SignAccuracy(Metric):
    full_state_update = False
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total",   default=torch.tensor(0.0), dist_reduce_fx="sum")

    @staticmethod
    def _unpack_target(target):
        if isinstance(target, (tuple, list)):
            y = target[0]; w = target[1] if len(target) > 1 else None
        elif isinstance(target, dict):
            y = target.get("target", target.get("y", target))
            w = target.get("weight", None)
        else:
            y = target; w = None
        return y, w

    def update(self, y_pred, target):
        # DeepAR의 y_pred는 분포 파라미터 dict. 평균은 'loc'
        mu = y_pred["loc"]  # (B,T) or (N,)
        y, w = self._unpack_target(target)
        if mu.shape != y.shape:
            y = y.view_as(mu)

        pred_pos = (mu > 0).float()
        y_pos    = (y  > 0).float()
        hit = (pred_pos == y_pos).float()

        if w is not None:
            if w.shape != hit.shape:
                w = w.view_as(hit)
            self.correct += (hit * w).sum()
            self.total   += w.sum().clamp_min(1.0)
        else:
            self.correct += hit.sum()
            self.total   += torch.tensor(float(hit.numel()), device=hit.device)

    def compute(self):
        return (self.correct / self.total).clamp(0.0, 1.0)
