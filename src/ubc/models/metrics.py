import torch
import torchmetrics as tm

__all__ = ["EpochLoss"]

class EpochLoss(tm.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("loss", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("batches", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, loss: torch.Tensor):
        self.loss += loss.item()
        self.batches += 1

    def compute(self):
        return self.loss.float() / self.batches