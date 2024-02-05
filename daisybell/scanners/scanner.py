from logging import Logger
from typing import List

from transformers import Pipeline


class ScannerRegistry(type):
    registered_scanners: List[type] = list()

    def __init__(cls, name, bases, attrs):
        super().__init__(cls)
        if name != "ScannerBase":
            ScannerRegistry.registered_scanners.append(cls)


class ScannerBase(metaclass=ScannerRegistry):
    def __init__(self, name: str, kind: str, description: str, logger: Logger) -> None:
        self.name = name
        self.kind = kind
        self.description = description
        self.logger = logger

    def can_scan(self, model: Pipeline) -> bool:
        raise NotImplementedError("Base class can_scan should not be directly invoked")

    def scan(self, model: Pipeline, params: dict) -> dict:
        raise NotImplementedError("Base class scan should not be directly invoked")
