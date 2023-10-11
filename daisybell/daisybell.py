import logging
import logging.handlers
import os
from typing import Generator

from transformers import Pipeline

from daisybell.scanners import ScannerRegistry


def scan(model: Pipeline, params: dict = {}) -> Generator:
    """
    Scans a model for problems.

    Parameters:
        model: A HuggingFace Pipeline to scan.

    Returns:
        A (name, kind, description, {output}) tuple of all scanners that are applicable to the model.
    """

    root_logger = logging.getLogger("daisybell")
    root_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))

    for scanner_class in ScannerRegistry.registered_scanners:
        scanner = scanner_class(root_logger)
        if scanner.can_scan(model):
            yield scanner.name, scanner.kind, scanner.description, scanner.scan(model, params)
