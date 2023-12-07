from typing import Any, List, Optional
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from .model_registry import MODEL_REGISTRY, ConfigLightningModel
import segmentation_models_pytorch as smp
import torch.nn as nn
from torchmetrics import Accuracy, Dice

@MODEL_REGISTRY.register()
class Unet(ConfigLightningModel):
    
    def __init__(self, config: DictConfig, weights: List[int] | None = None) -> None:
        super().__init__(config)
        img_clases, mask_classes = 3, 3
        image_unet = smp.Unet(**config.model.params, classes=img_clases)
        self.encoder = image_unet.encoder
        mask_unet = smp.Unet(**config.model.params, classes=mask_classes)
        self.image_decoder = image_unet.decoder
        self.mask_decoder = mask_unet.decoder
        self.image_segmentation_head = image_unet.segmentation_head
        self.mask_segmentation_head = mask_unet.segmentation_head
        self.image_loss = nn.MSELoss()
        self.mask_loss = nn.CrossEntropyLoss()
        self.dice = Dice(num_classes=mask_classes)

    def encode(self, x):
        return self.encoder(x)
    
    def get_image_outputs(self, features):
        return self.image_segmentation_head(self.image_decoder(*features))
    
    def get_mask_outputs(self, features):
        return self.mask_segmentation_head(self.mask_decoder(*features))
    
    def forward(self, x):
        encoder_features = self.encode(x)
        output = {}
        output["image"] = self.get_image_outputs(encoder_features)
        output["mask"] = self.get_mask_outputs(encoder_features)
        return output
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self(batch['image'])
        img_loss = self.image_loss(outputs["image"], batch["image"])
        mask_loss = self.mask_loss(outputs["mask"], batch["mask"].float())
        loss = img_loss + mask_loss
        self.log_dict({"train_loss": loss, "img_loss": img_loss, "mask_loss": mask_loss})
        return loss
    
    def on_validation_epoch_start(self) -> None:
        self.epoch_img_loss = 0
        self.num_val_batches = 0
        return super().on_validation_epoch_start()
    
    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self(batch['image'])
        self.epoch_img_loss += self.image_loss(outputs["image"], batch["image"])
        # self.dice(outputs["mask"], batch["mask"])
        self.num_val_batches += 1
        return 
        
    def on_validation_epoch_end(self) -> None:
        # dice = self.dice.compute()
        val_img_loss = self.epoch_img_loss / self.num_val_batches
        self.log_dict({"val/epoch_image_loss": val_img_loss, "val/dice": 0})
        return super().on_validation_epoch_end()
        
        