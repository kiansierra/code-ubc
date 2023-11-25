from pathlib import Path
from typing import Dict, List, Optional

from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig


def get_configs(folder:Optional[str] = None) -> Dict[str, DictConfig]:
    config_dir = Path(__file__).parent
    if folder:
        config_dir = config_dir / folder
    config_names = [file.name.replace(".yaml", "") for file in config_dir.glob("*.yaml")]
    GlobalHydra().clear()
    if not GlobalHydra().is_initialized():
        initialize_config_dir(config_dir=str(config_dir), version_base="1.2")
    outputs =  {name: compose(config_name=name) for name in config_names}
    return outputs



def list_configs(folder:Optional[str] = None) -> List[str]:
    config_dir = Path(__file__).parent
    if folder:
        config_dir = config_dir / folder
    config_names = [file.name.replace(".yaml", "") for file in config_dir.glob("*.yaml")]
    return config_names

def get_config(config_name:str, folder:Optional[str] = None) -> DictConfig:
    config_dir = Path(__file__).parent
    if folder:
        config_dir = config_dir / folder
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=config_name)
    return cfg
