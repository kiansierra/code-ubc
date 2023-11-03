import os
from pathlib import Path
from typing import List, Optional

from loguru import logger

import wandb

from .additional_types import BatchElement, DataElement, DeviceType

__all__ = ["batch_to_device", "reverse_batch", "upload_to_wandb"]


def batch_to_device(batch: BatchElement, device: DeviceType = "cuda") -> BatchElement:
    """Moves all tensors to corresponding device"""
    return {k: v.to(device) for k, v in batch.items()}


def reverse_batch(batch: BatchElement) -> List[DataElement]:
    """Converts the Batched Dictionary to a list of dictionarys of length batch_size.

    Args:
        batch (BatchElement): Dictionary of batched elements

    Returns:
        List[DataElement]: List of dictionary Elements
    """
    return [dict(zip(batch, t)) for t in zip(*batch.values())]


def upload_to_wandb(
    folder: Path,
    name: str,
    pattern: str = "*",
    artifact_type: str = "model",
    job_type="upload",
    config=None,
    return_artifact: bool = False,
    log_folders: bool = False,
    end_wandb_run: bool = True,
) -> Optional[wandb.Artifact]:
    """
    Uploads files in a folder to wandb
    Args:
        folder (Path): Folder to upload
        name (str): Name of the artifact
        pattern (str, optional): Pattern to match files. Defaults to "*".
        artifact_type (str, optional): Type of artifact. Defaults to "model".
        job_type (str, optional): Type of job. Defaults to "upload".
        config ([type], optional): Config to log. Defaults to None.
        return_artifact (bool, optional): Whether to return the artifact. Defaults to False.
        log_folders (bool, optional): Whether to log sub-folders. Defaults to False.
        end_wandb_run (bool, optional): Whether to end the wandb run. Defaults to True.
    Returns:
        Optional[wandb.Artifact]: Artifact if return_artifact is True
    """
    folder = Path(folder)
    run = wandb.run
    if run is None:
        WANDB_USER = os.environ["WANDB_USER"]
        PROJECT_NAME = os.environ["WANDB_PROJECT"]
        run = wandb.init(entity=WANDB_USER, project=PROJECT_NAME, job_type=job_type, config=config)
    logger.info(f"uploading {artifact_type}:{name} to project {run.project}")
    competition_data = wandb.Artifact(name, type=artifact_type)
    for path in folder.glob(pattern):
        if path.is_dir() and log_folders:
            competition_data.add_dir(path, name=path.name)
            logger.info(f"uploaded folder {path.name}")
            continue
        if path.is_file():
            competition_data.add_file(path, name=path.name)
            logger.info(f"uploaded file {path.name}")
    if return_artifact:
        return competition_data
    run.log_artifact(competition_data)
    if end_wandb_run:
        run.finish()
    return competition_data
