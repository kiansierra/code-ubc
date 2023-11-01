from typing import Any, Optional

import pytorch_lightning as pl
import timm
import torch
import torchmetrics as tm
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1)*p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)
        
    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)
        
    def __repr__(self):
        return self.__class__.__name__ + \
                '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + \
                ', ' + 'eps=' + str(self.eps) + ')'


class TimmModel(pl.LightningModule):
    
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        model_config = config['model']
        self.backbone = timm.create_model(model_config['backbone'], pretrained=model_config['pretrained'])
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.backbone.global_pool = nn.Identity()
        self.pooling = GeM()
        self.linear = nn.Linear(in_features, model_config['num_classes'])
        self.softmax = nn.Softmax(dim=1)
        self.metric = tm.Accuracy(num_classes=model_config['num_classes'], task='multiclass', average='macro')
        
    def forward(self, images):
        features = self.backbone(images)
        pooled_features = self.pooling(features).flatten(1)
        logits = self.linear(pooled_features)
        output = {'logits': logits, 'features': pooled_features, 'probs': self.softmax(logits)}
        return output
    
    def on_train_epoch_start(self) -> None:
        self.train_epoch_loss = 0
        self.train_batches = 0
        return super().on_train_epoch_start()
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        images = batch['image']
        labels = batch['label']
        output = self(images)
        loss = F.cross_entropy(output['logits'], labels)
        self.log('train/loss', loss, prog_bar=True)
        self.train_epoch_loss += loss.item()
        self.train_batches += 1
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.log('train/epoch_loss', self.train_epoch_loss/self.train_batches, prog_bar=True)
        return super().on_train_epoch_end()
    
    def on_validation_epoch_start(self) -> None:
        self.val_epoch_loss = 0
        self.val_batches = 0
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT | None:
        images = batch['image']
        labels = batch['label']
        output = self(images)
        loss = F.cross_entropy(output['logits'], labels)
        self.metric.update(output['probs'], labels)
        self.val_epoch_loss += loss.item()
        self.val_batches += 1
        return super().validation_step()
    
    def on_validation_epoch_end(self) -> None:
        metric = self.metric.compute()
        self.metric.reset()
        self.log('val/balanced-accuracy', metric, prog_bar=True)
        self.log('val/epoch_loss', self.val_epoch_loss/self.val_batches, prog_bar=True)
        return super().on_validation_epoch_end()
    
    def configure_optimizers(self) -> Any:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return [optimizer], []