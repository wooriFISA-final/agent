"""Logging configuration"""
import logging
import sys
from pathlib import Path

def setup_logger(
    name: str = "agent_system",
    level: str = "INFO",
    log_file: str = "logs/agent_system.log"
) -> logging.Logger:
    """Setup logger"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
