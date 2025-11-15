"""
Logging utilities for the adversarial patch pipeline.
"""
import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = 'adversarial_patch',
    log_dir: str = 'logs',
    level: int = logging.INFO,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        log_to_file: Whether to log to file
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


