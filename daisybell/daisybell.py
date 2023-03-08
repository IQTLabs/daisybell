from typing import Any, Callable, Generator, Sequence, Tuple

import pandas as pd
from transformers import Pipeline, pipeline

REGISTERED_SCANNERS = []

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
        REGISTERED_SCANNERS.append(scanner())

    return inner


def scan(model: Pipeline, params: dict = {}) -> Generator:
    """
    Scans a model for problems.

    Parameters:
        model: A HuggingFace Pipeline to scan.

    Returns:
        A (name, kind, description, {output}) tuple of all scanners that are applicable to the model.
    """
    for scanner in REGISTERED_SCANNERS:
        if scanner.can_scan(model):
            yield scanner.__scanner_name__, scanner.__scanner_kind__, scanner.__scanner_description__, scanner.scan(
                model, params
            )
