from typing import Any

import pytorch_lightning as pl
import timm
import torch
import torchmetrics as tm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F

from .metrics import EpochLoss
from .optimization_utils import get_optimizer, get_scheduler


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )

class BaseLightningModel(pl.LightningModule):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        model_config = config["model"]
        metrics = tm.MetricCollection(
            tm.Accuracy(num_classes=model_config["num_classes"], task="multiclass", average="macro"),
            tm.Precision(num_classes=model_config["num_classes"], task="multiclass", average="macro"),
            tm.Recall(num_classes=model_config["num_classes"], task="multiclass", average="macro"),
            EpochLoss(),
        )
        self.train_metric = metrics.clone(prefix="train/")
        self.val_metric = metrics.clone(prefix="val/")
        
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images = batch["image"]
        labels = batch["label"]
        output = self(images)
        loss = F.cross_entropy(output["logits"], labels)
        self.log("train/loss", loss, prog_bar=True)
        self.train_metric.update(preds=output["probs"], target=labels, loss=loss)
        return loss

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metric.compute()
        self.train_metric.reset()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        images = batch["image"]
        labels = batch["label"]
        output = self(images)
        loss = F.cross_entropy(output["logits"], labels)
        self.val_metric.update(preds=output["probs"], target=labels, loss=loss)
        return super().validation_step()

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        self.val_metric.reset()
        self.log_dict(metrics, prog_bar=True, sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = self(batch["image"])
        return {'image_id': batch['image_id'], 'probs': output['probs']}
    

    def configure_optimizers(self) -> Any:
        optimizer = get_optimizer(self.config, self)
        scheduler = get_scheduler(self.config, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class TimmModel(BaseLightningModel):
    def __init__(self, config) -> None:
        super().__init__(config)
        model_config = config["model"]
        self.backbone = timm.create_model(model_config["backbone"], pretrained=model_config["pretrained"])
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, model_config["num_classes"])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        features = self.backbone(images)
        pooled_features = self.pooling(features).flatten(1)
        logits = self.linear(pooled_features)
        output = {"logits": logits, "features": pooled_features, "probs": self.softmax(logits)}
        return output

