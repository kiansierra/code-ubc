from typing import Dict, List, Tuple

import torch
import torch.nn as nn

__all__ = ["Mixup", "CutMix"]


class Mixup(nn.Module):
    def __init__(self, keys: List[str], alpha: float = 0.4, prob: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        self.keys = keys
        self.main_key = "image"

    def forward(self, batch: Dict[str, torch.Tensor], skip: bool = False) -> Tuple[Dict[str, torch.Tensor], bool]:
        if skip:
            return batch, False
        if torch.rand(1) > self.prob:
            return batch, False
        lam = torch.distributions.beta.Beta(self.alpha, self.alpha).sample().item()
        lam = max(lam, 1 - lam)
        bs = batch[self.main_key].size(0)
        index = torch.randperm(bs).to(batch[self.main_key].device)
        intersect_keys = list(set(self.keys) & set(batch.keys()))
        for key in intersect_keys:
            batch[key] = lam * batch[key] + (1 - lam) * batch[key][index]
        return batch, True


class CutMix(nn.Module):
    def __init__(self, aux_keys: List[str], image_key: str = "image", beta: float = 1.0, prob: float = 0.5):
        super(CutMix, self).__init__()
        self.beta = beta
        self.prob = prob
        self.image_key = image_key
        self.aux_keys = aux_keys

    def forward(self, batch: Dict[str, torch.Tensor], skip: bool = False) -> Tuple[Dict[str, torch.Tensor], bool]:
        if skip:
            return batch, False
        if torch.rand(1) > self.prob:
            return batch, False
        images = batch[self.image_key]
        lam = torch.distributions.beta.Beta(self.beta, self.beta).sample().item()
        rand_index = torch.randperm(images.size()[0]).to(images.device)
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam, images.device)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        intersect_keys = list(set(self.aux_keys) & set(batch.keys()))
        for key in intersect_keys:
            target_a = batch[key]
            target_b = batch[key][rand_index]
            batch[key] = target_a * lam + target_b * (1.0 - lam)
        return batch, True

    @staticmethod
    def rand_bbox(size, lam, device):
        W = size[2]
        H = size[3]
        cut_rat = torch.sqrt(torch.Tensor([1.0 - lam])).to(device)
        cut_w = (W * cut_rat).type(torch.long)
        cut_h = (H * cut_rat).type(torch.long)
        cx = torch.randint(0, W, (1,)).to(device)
        cy = torch.randint(0, H, (1,)).to(device)
        bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
        bby1 = torch.clamp(cy - cut_h // 2, 0, H)
        bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
        bby2 = torch.clamp(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2
