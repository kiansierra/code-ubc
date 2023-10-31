import importlib
import importlib.metadata as importlib_metadata
import logging

logger = logging.getLogger(__name__)

__all__ = ["is_composer_available"]

_composer_available = importlib.util.find_spec("composer") is not None

try:
    _composer_version = importlib_metadata.version("composer")
    logger.debug(f"Successfully imported composer version {_composer_version }")
except importlib_metadata.PackageNotFoundError:
    _composer_available = False


def is_composer_available():
    return _composer_available
