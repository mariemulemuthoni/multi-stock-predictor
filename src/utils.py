import logging
import os
import yaml
import random
import numpy as np
import tensorflow as tf

def load_config(config_path="config/config.yaml"):
    """Parses the YAML config file into a dictionary."""
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def setup_logger(name, log_file="logs/app.log"):
    """
    Configures a hierarchical logger with file and stream handlers.
    Ensures operational visibility across the pipeline.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Persist logs to disk
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Output logs to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

def set_seeds(seed=42):
    """
    Enforces deterministic behavior across numpy, python, and tensorflow
    to ensure reproducibility of experiments.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)