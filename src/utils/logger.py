"""Logging utilities for the fraud detection system."""

import sys
from pathlib import Path
from loguru import logger
from typing import Optional


def setup_logger(
    log_file: Optional[str] = None,
    level: str = "INFO",
    rotation: str = "100 MB",
    retention: str = "10 days",
    format_string: Optional[str] = None
) -> None:
    """Setup logger with file and console output.
    
    Args:
        log_file: Path to log file. If None, logs to console only.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate log file
        retention: How long to keep old log files
        format_string: Custom format string for logs
    """
    # Remove default handler
    logger.remove()
    
    # Default format
    if format_string is None:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add console handler
    logger.add(
        sys.stderr,
        format=format_string,
        level=level,
        colorize=True
    )
    
    # Add file handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format=format_string,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip"
        )
    
    logger.info(f"Logger initialized with level: {level}")


def get_logger():
    """Get logger instance.
    
    Returns:
        Logger instance
    """
    return logger
