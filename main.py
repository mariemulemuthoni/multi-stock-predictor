"""
Main Execution Entry Point
--------------------------
Orchestrates the end-to-end training pipeline:
1. Loads configuration.
2. Initializes the logging subsystem.
3. Triggers the Training Manager.
"""

import sys
from src.utils import load_config, setup_logger
from src.trainer import Trainer

def main():
    # Load config
    try:
        config = load_config("config/config.yaml")
    except FileNotFoundError:
        print("Critical Error: Configuration file 'config/config.yaml' not found.")
        sys.exit(1)

    # Initialize logging subsystem (writes to logs/app.log and console)
    logger = setup_logger("Main")
    logger.info("Initializing Stock Prediction System...")

    try:
        trainer = Trainer(config)
        trainer.train_all()
        logger.info("System execution completed successfully.")
        
    except Exception as e:
        logger.critical(f"System crashed due to unhandled exception: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()