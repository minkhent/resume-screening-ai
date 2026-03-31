import os
import sys
import logging

def setup_logger(log_file: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(os.path.join(save_dir, log_file))
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    logger.handlers = []
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger