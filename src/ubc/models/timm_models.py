from typing import Any, List, Optional

import timm
import torch
import torchmetrics as tm
from einops import rearrange

# from fvcore.common.registry import Registry
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F

import wandb

from ..data.constants import idx2label
from .cut_augmentations import CutMix, Mixup
from .metrics import ClassBalancedAccuracy, EpochLoss
from .model_registry import MODEL_REGISTRY, ConfigLightningModel
from .losses import LOSSES

__all__ = ["TimmModel", "TimmVITModel", "BaseLightningModel"]


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


def create_table(image_ids: torch.Tensor, images: torch.Tensor, labels: torch.Tensor, probs: torch.Tensor):
    images = images.permute(0, 2, 3, 1).cpu().numpy()
    columns = ["image_id", "image", "label", *idx2label.values()]
    if probs.shape[1] != len(idx2label):
        columns = columns[:-1]
    table = []
    image_ids = image_ids.cpu().numpy()
    probs = probs.cpu().numpy()
    labels = labels.cpu().numpy()
    for image_id, image, label, prob in zip(image_ids, images, labels, probs):
        row = [image_id, wandb.Image(image), idx2label[label]]
        row.extend(prob)
        table.append(row)
    return columns, table


class BaseLightningModel(ConfigLightningModel):
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__(config=config)
        self.example_input_array = torch.zeros((config["batch_size"], 3, config["img_size"], config["img_size"]))
        model_config = config["model"]
        num_classes = model_config["num_classes"]
        self.num_classes = num_classes
        metrics = tm.MetricCollection(
            tm.Accuracy(num_classes=num_classes, task="multiclass", average="macro"),
            tm.Precision(num_classes=num_classes, task="multiclass", average="macro"),
            tm.Recall(num_classes=num_classes, task="multiclass", average="macro"),
            ClassBalancedAccuracy(num_classes=num_classes, average="macro"),
            EpochLoss(),
        )

        loss_config = model_config["loss"]
        self.loss_fn = LOSSES.get(loss_config.name)(
            weight=torch.tensor(weights) if loss_config.use_weights else None,
            label_smoothing=loss_config.label_smoothing,
        )
        self.train_metric_global = metrics.clone(prefix="train/global/")
        self.train_metric_wsi = metrics.clone(prefix="train/wsi/")
        self.train_metric_tma = metrics.clone(prefix="train/tma/")
        self.val_metric_global = metrics.clone(prefix="val/global/")
        self.val_metric_wsi = metrics.clone(prefix="val/wsi/")
        self.val_metric_tma = metrics.clone(prefix="val/tma/")
        self.cutmix = CutMix(aux_keys=["label", "weight"], image_key="image", **model_config["cutmix"])
        self.mixup = Mixup(keys=["image", "label", "weight"], **model_config["mixup"])

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["label"]
        batch["label"] = F.one_hot(batch["label"], self.num_classes).float()
        batch, skip = self.mixup(batch)
        batch, skip = self.cutmix(batch, skip)
        images = batch["image"]
        output = self(images)
        loss = self.loss_fn(output["logits"], batch["label"], batch.get("weight", None))
        self.log("train/loss", loss.mean(), prog_bar=True)
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.train_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.train_metric_wsi.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.train_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        if batch_idx == 0:
            columns, data = create_table(batch["image_id"], images, labels, output["probs"].detach())
            self.trainer.logger.log_table("train/images", columns=columns, data=data)
        return loss.mean()

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metric_wsi.compute()
        metrics_tma = self.train_metric_tma.compute()
        metrics_global = self.train_metric_global.compute()

        self.train_metric_wsi.reset()
        self.train_metric_tma.reset()
        self.train_metric_global.reset()

        averaged_metrics = {
            k.replace("wsi/", "balanced/"): (0.5 * v + 0.5 * metrics_tma[k.replace("wsi/", "tma/")])
            for k, v in metrics.items()
        }
        all_metrics = {**averaged_metrics, **metrics, **metrics_tma, **metrics_global}
        self.log_dict(all_metrics, prog_bar=False, sync_dist=True)
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        labels = batch["label"]
        batch["label"] = F.one_hot(batch["label"], self.num_classes).float()
        images = batch["image"]
        output = self(images)
        loss = self.loss_fn(output["logits"], labels, batch.get("weight", None))
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.val_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.val_metric_wsi.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.val_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        if batch_idx == 0:
            columns, data = create_table(batch["image_id"], images, labels, output["probs"])
            self.trainer.logger.log_table("val/images", columns=columns, data=data)
        return super().validation_step()

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metric_wsi.compute()
        metrics_tma = self.val_metric_tma.compute()
        metrics_global = self.val_metric_global.compute()

        self.val_metric_wsi.reset()
        self.val_metric_tma.reset()
        self.val_metric_global.reset()

        averaged_metrics = {
            k.replace("wsi/", "balanced/"): (0.5 * v + 0.5 * metrics_tma[k.replace("wsi/", "tma/")])
            for k, v in metrics.items()
        }
        all_metrics = {**averaged_metrics, **metrics, **metrics_tma, **metrics_global}
        self.log_dict(all_metrics, prog_bar=True, sync_dist=True)
        return super().on_validation_epoch_end()

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        output = self(batch["image"])
        return {"image_id": batch["image_id"], "probs": output["probs"]}


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


@MODEL_REGISTRY.register()
class TimmBasicModel(BaseLightningModel):
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__(config, weights=weights)
        model_config = config["model"]
        global_pool = model_config.get("global_pool", "avg")

        self.backbone = timm.create_model(
            model_config["backbone"],
            pretrained=model_config["pretrained"],
            num_classes=model_config["num_classes"],
            global_pool=global_pool if global_pool != "gem" else "avg",
        )
        if global_pool == "gem":
            self.backbone.global_pool = GeM()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, images):
        features = self.backbone.forward_features(images)
        logits = self.backbone.forward_head(features)
        output = {"logits": logits, "features": features, "probs": self.softmax(logits)}
        return output


@MODEL_REGISTRY.register()
class TimmModelBCE(BaseLightningModel):
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__(config, weights=weights)
        model_config = config["model"]

        self.backbone = timm.create_model(
            model_config["backbone"],
            pretrained=model_config["pretrained"],
            num_classes=model_config["num_classes"],
            global_pool=model_config.get("global_pool", "avg"),
        )
        self.softmax = nn.Softmax(dim=1)
        self.cutmix = CutMix(aux_keys=["label", "weight"], image_key="image", beta=1.0, prob=0.5)
        self.mixup = Mixup(keys=["image", "label", "weight"], alpha=1.0, prob=0.5)
        self.loss_fn = nn.BCEWithLogitsLoss(weight=torch.tensor(weights) if weights else None, reduction="none")
        self.num_classes = model_config["num_classes"]

    def forward(self, images):
        features = self.backbone.forward_features(images)
        logits = self.backbone.forward_head(features)
        output = {"logits": logits, "features": features, "probs": self.softmax(logits)}
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch["label"] = F.one_hot(batch["label"], self.num_classes).float()
        batch, skip = self.mixup(batch)
        batch, skip = self.cutmix(batch, skip)
        output = self(batch["image"])
        loss = self.loss_fn(output["logits"], batch["label"])
        if "weight" in batch:
            loss = loss * batch["weight"].unsqueeze(-1) / batch["weight"].mean()
        self.log("train/loss", loss.mean(), prog_bar=True)
        return loss.mean()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        images = batch["image"]
        label = batch["label"]
        batch["label"] = F.one_hot(batch["label"], self.num_classes).float()
        labels = batch["label"]
        output = self(images)
        loss = self.loss_fn(output["logits"], labels)
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.val_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.val_metric_wsi.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.val_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        if batch_idx == 0:
            columns, data = create_table(batch["image_id"], images, label, output["probs"])
            self.trainer.logger.log_table("images", columns=columns, data=data)
        return


@MODEL_REGISTRY.register()
class TileModel(BaseLightningModel):
    def __init__(self, config: DictConfig, weights: Optional[List[int]] = None) -> None:
        super().__init__(config, weights=weights)
        model_config = config["model"]
        num_tiles = 16
        self.example_input_array = {
            "images": torch.zeros((1, num_tiles, 3, config["img_size"], config["img_size"])),
            "x_pos": torch.zeros((1, num_tiles)).long(),
            "y_pos": torch.zeros((1, num_tiles)).long(),
        }

        self.backbone = timm.create_model("tf_efficientnet_b0_ns", pretrained=True)
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = GeM()
        self.x_embed = nn.Embedding(model_config["num_tiles"], self.backbone.num_features)
        self.y_embed = nn.Embedding(model_config["num_tiles"], self.backbone.num_features)
        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(self.backbone.num_features, model_config["num_heads"], batch_first=True)
                for _ in range(model_config["num_layers"])
            ]
        )

        self.head = nn.Linear(self.backbone.num_features, model_config["num_classes"])
        self.softmax = nn.Softmax(dim=1)
        # self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        bs = x.shape[0]
        x = rearrange(x, "b n c h w -> (b n) c h w")
        features = self.backbone(x)
        features = self.pooling(features)
        features = rearrange(features, "(b n) c -> b n c", b=bs)
        return features

    def forward(self, images: torch.Tensor, x_pos: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
        features = self.get_features(images)
        x_embed = self.x_embed(x_pos)
        y_embed = self.y_embed(y_pos)
        features = features + x_embed + y_embed
        for layer in self.layers:
            features = layer(features)
        features = features.mean(dim=1)
        logits = self.head(features)
        output = {"logits": logits, "features": features, "probs": self.softmax(logits)}
        return output

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images = batch["image"]
        labels = batch["label"]
        output = self(images, batch["pos_x"], batch["pos_y"])
        loss = self.loss_fn(output["logits"], labels)
        self.log("train/loss", loss.mean(), prog_bar=True)
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.train_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.train_metric_wsi.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.train_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        return loss.mean()

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        images = batch["image"]
        labels = batch["label"]
        output = self(images, batch["pos_x"], batch["pos_y"])
        loss = self.loss_fn(output["logits"], labels)
        tma_index = torch.where(batch["is_tma"] == 1)[0]
        wsi_index = torch.where(batch["is_tma"] == 0)[0]
        self.val_metric_global.update(preds=output["probs"], target=labels, loss=loss.mean())
        if len(wsi_index):
            self.val_metric_wsi.update(
                preds=output["probs"][wsi_index], target=labels[wsi_index], loss=loss[wsi_index].mean()
            )
        if len(tma_index):
            self.val_metric_tma.update(
                preds=output["probs"][tma_index], target=labels[tma_index], loss=loss[tma_index].mean()
            )
        return
