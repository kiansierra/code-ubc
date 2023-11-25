from typing import Any, List, Optional

import pytorch_lightning as pl
import timm
import torch
import torchmetrics as tm
# from fvcore.common.registry import Registry
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F

from ..utils import Registry
from .metrics import ClassBalancedAccuracy, EpochLoss
from .optimization_utils import get_optimizer, get_scheduler

__all__ = ["TimmModel", "TimmVITModel", "MODEL_REGISTRY", "BaseLightningModel"]


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        pooled = F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)
        return pooled.squeeze(-1).squeeze(-1)

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
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__()
        self.config = config
        self.example_input_array = torch.zeros((1, 3, config["img_size"], config["img_size"]))

        model_config = config["model"]
        metrics = tm.MetricCollection(
            tm.Accuracy(num_classes=model_config["num_classes"], task="multiclass", average="macro"),
            tm.Precision(num_classes=model_config["num_classes"], task="multiclass", average="macro"),
            tm.Recall(num_classes=model_config["num_classes"], task="multiclass", average="macro"),
            ClassBalancedAccuracy(num_classes=model_config["num_classes"], average="macro"),
            EpochLoss(),
        )
        self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights) if weights else None, reduction="none")
        self.train_metric_global = metrics.clone(prefix="train/global_")
        self.train_metric = metrics.clone(prefix="train/wsi_")
        self.train_metric_tma = metrics.clone(prefix="train/tma_")
        self.val_metric_global = metrics.clone(prefix="val/global_")
        self.val_metric = metrics.clone(prefix="val/wsi_")
        self.val_metric_tma = metrics.clone(prefix="val/tma_")

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images = batch["image"]
        labels = batch["label"]
        output = self(images)
        loss = self.loss_fn(output["logits"], labels)
        self.log("train/loss", loss.mean(), prog_bar=True)
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.train_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.train_metric.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.train_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        return loss.mean()

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metric.compute()
        metrics_tma = self.train_metric_tma.compute()
        metrics_global = self.train_metric_global.compute()

        self.train_metric.reset()
        self.train_metric_tma.reset()
        self.train_metric_global.reset()

        averaged_metrics = {
            k.replace("wsi_", ""): (0.5 * v + 0.5 * metrics_tma[k.replace("wsi_", "tma_")]) for k, v in metrics.items()
        }
        all_metrics = {**averaged_metrics, **metrics, **metrics_tma, **metrics_global}
        self.log_dict(all_metrics, prog_bar=True, sync_dist=True)
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        images = batch["image"]
        labels = batch["label"]
        output = self(images)
        loss = self.loss_fn(output["logits"], labels)
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.val_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.val_metric.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.val_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        return super().validation_step()

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric.compute()
        metrics_tma = self.val_metric_tma.compute()
        metrics_global = self.val_metric_global.compute()

        self.val_metric.reset()
        self.val_metric_tma.reset()
        self.val_metric_global.reset()

        averaged_metrics = {
            k.replace("wsi_", ""): (0.5 * v + 0.5 * metrics_tma[k.replace("wsi_", "tma_")]) for k, v in metrics.items()
        }
        all_metrics = {**averaged_metrics, **metrics, **metrics_tma, **metrics_global}
        self.log_dict(all_metrics, prog_bar=True, sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = self(batch["image"])
        return {"image_id": batch["image_id"], "probs": output["probs"]}

    def configure_optimizers(self) -> Any:
        optimizer = get_optimizer(self.config, self)
        scheduler = get_scheduler(self.config, optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


MODEL_REGISTRY = Registry("MODELS", BaseLightningModel)


@MODEL_REGISTRY.register()
class TimmModel(BaseLightningModel):
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__(config, weights=weights)
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
        pooled_features = self.pooling(features)
        logits = self.linear(pooled_features)
        output = {"logits": logits, "features": pooled_features, "probs": self.softmax(logits)}
        return output


@MODEL_REGISTRY.register()
class TimmVITModel(BaseLightningModel):
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__(config, weights=weights)
        model_config = config["model"]
        self.backbone = timm.create_model(
            model_config["backbone"], pretrained=model_config["pretrained"], num_classes=0
        )
        in_features = self.backbone.num_features
        # self.backbone.classifier = nn.Identity()
        # self.backbone.global_pool = nn.Identity()
        self.linear = nn.Linear(in_features, model_config["num_classes"])
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        features = self.backbone(images)
        logits = self.linear(features)
        output = {"logits": logits, "features": features, "probs": self.softmax(logits)}
        return output
