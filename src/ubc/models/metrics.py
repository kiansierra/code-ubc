import torch
import torchmetrics as tm
from torchmetrics.classification.stat_scores import MulticlassStatScores
from torchmetrics.functional.classification.accuracy import _accuracy_reduce

from ..data import label2idx

__all__ = ["EpochLoss", "ClassBalancedAccuracy"]


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
    
    
class ClassBalancedAccuracy(MulticlassStatScores):
    
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    
    
    def compute(self) -> torch.Tensor:
        """Compute accuracy based on inputs passed in to ``update`` previously."""
        tp, fp, tn, fn = self._final_state()
        class_acc =  _accuracy_reduce(tp, fp, tn, fn, average='none', multidim_average=self.multidim_average)
        output = {k: class_acc[v] for k,v  in label2idx.items()}
        output['bac'] =  _accuracy_reduce(tp, fp, tn, fn, average=self.average, multidim_average=self.multidim_average)
        return output
