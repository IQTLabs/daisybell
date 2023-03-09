from logging import Logger
from typing import Any, Callable, List

import pandas as pd
from transformers import Pipeline

class ScannerRegistry(type):
    registered_scanners: List[type] = list()

    def __init__(cls, name, bases, attrs):
        super().__init__(cls)
        if name != 'ScannerBase' :
            ScannerRegistry.registered_scanners.append(cls)

class ScannerBase(metaclass=ScannerRegistry):
    def __init__(self, name: str, kind: str, description: str, logger: Logger) -> None:
        self.name = name
        self.kind = kind
        self.description = description
        self.logger = logger

    def can_scan(self, model: Pipeline) -> bool:
        self.logger.warning('Base class can_scan should not be directly invoked')
        return False

    def scan(self, model: Pipeline, params: dict) -> pd.DataFrame:
        self.logger.warning('Base class scan should not be directly invoked')
        return None

def scanner(name: str, kind: str, description: str) -> Callable:
    """
    Register a new scanner. This function is intended to be used as a decorator.

        Parameters:
            name: The name of the scanner.
            kind: What kind of scanner it is (eg. bias).
            descriptor: A short description of the scanner.
    """
    def inner(scanner: Any):
        scanner.__scanner_name__ = name
        scanner.__scanner_kind__ = kind
        scanner.__scanner_description__ = description
        # REGISTERED_SCANNERS.append(scanner())

    return inner