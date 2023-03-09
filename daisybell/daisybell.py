from importlib import import_module
import inspect
import logging
import logging.handlers
from logging import Logger
import os
from typing import Any, Callable, Generator, List

from transformers import Pipeline

from daisybell.scanners import ScannerRegistry

# MODULES = ["daisybell.scanners"]

# def register_scanner(scanner, logger: Logger):
#     breakpoint()
#     scanner(logger)

# def collect_scanners(scanner_modules: List[str], logger: Logger) -> List[type]:
#     scanners = list()
#     for module_name in scanner_modules:
#         breakpoint()
#         module = import_module(module_name)
#         classes = [(name, obj) for name, obj in inspect.getmembers(module) if inspect.isclass(obj)]
#         for c in classes:
#             if issubclass(c[1],ScannerBase) and c[0] != 'ScannerBase':
#                 register_scanner(c[1], logger)

#     return scanners

def scan(model: Pipeline, params: dict = {}) -> Generator:
    """
    Scans a model for problems.

    Parameters:
        model: A HuggingFace Pipeline to scan.

    Returns:
        A (name, kind, description, {output}) tuple of all scanners that are applicable to the model.
    """

    handler = logging.handlers.WatchedFileHandler(
    os.environ.get("LOGFILE", "./daisybell.log"))
    formatter = logging.Formatter(logging.BASIC_FORMAT)
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    root_logger.addHandler(handler)

    # registered_scanners = collect_scanners(MODULES, root_logger)

    for scanner_class in ScannerRegistry.registered_scanners:
        scanner = scanner_class(root_logger)
        if scanner.can_scan(model):
            yield scanner.name, scanner.kind, scanner.description, scanner.scan(
                model, params
            )
