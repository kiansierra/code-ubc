import torch
import torchmetrics as tm
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import BaseLightningModel

__all__ = ["train_loop"]


def train_epoch(
    epoch: int,
    dataloader: DataLoader,
    model: BaseLightningModel,
    optimizer: Optimizer,
    scheduler: _LRScheduler,
    device,
):
    model.train()
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0
    for num, batch in bar:

        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = model.loss_fn(outputs["logits"], targets).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        _, predicted = torch.max(outputs["probs"], 1)
        acc = torch.sum(predicted == targets)
        recall_nn = tm.Recall(task="multiclass", average="macro", num_classes=5).cuda()
        recall = recall_nn(predicted, targets)
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += acc.item()
        running_recall += recall.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        epoch_recall = running_recall / dataset_size
        bar.set_postfix(
            Epoch=epoch,
            Train_Loss=epoch_loss,
            Train_Acc=epoch_acc,
            Train_Recall=epoch_recall,
            LR=optimizer.param_groups[0]["lr"],
        )
    return


@torch.no_grad()
def validate_epoch(epoch: int, dataloader: DataLoader, model: BaseLightningModel, device):
    model.eval()
    bar = tqdm(enumerate(dataloader), total=len(dataloader))
    dataset_size = 0
    running_loss = 0.0
    running_acc = 0.0
    running_recall = 0.0
    for num, batch in bar:
        images = batch["image"].to(device)
        targets = batch["label"].to(device)
        outputs = model(images)
        loss = model.loss_fn(outputs["logits"], targets)
        _, predicted = torch.max(outputs["probs"], 1)
        acc = torch.sum(predicted == targets)
        recall_nn = tm.Recall(task="multiclass", average="macro", num_classes=5).cuda()
        recall = recall_nn(predicted, targets)
        batch_size = images.size(0)
        running_loss += loss.mean().item() * batch_size
        running_acc += acc.item()
        running_recall += recall.item() * batch_size
        dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        epoch_acc = running_acc / dataset_size
        epoch_recall = running_recall / dataset_size
        bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc, Valid_Recall=epoch_recall)

    return


def train_loop(
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    model: BaseLightningModel,
    config: DictConfig,
    device,
):
    optimizers, scheduler_dict = model.configure_optimizers()
    optimizer = optimizers[0]
    scheduler = scheduler_dict[0]["scheduler"]
    model.to(device)
    for epoch in range(config.trainer.max_epochs):
        train_epoch(epoch, train_dataloader, model, optimizer, scheduler, device)
        validate_epoch(epoch, validation_dataloader, model, device)

    return
