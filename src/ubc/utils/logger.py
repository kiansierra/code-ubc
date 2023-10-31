import logging


def format_logging():
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")
    # add formatter to stream_handler
    stream_handler.setFormatter(formatter)
    # add stream_handler to logger
    return stream_handler


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if logger.hasHandlers():
        logger.handlers.clear()
    logger.addHandler(format_logging())
    return logger
