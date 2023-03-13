import logging
import pytest

from daisybell.scanners import ScannerBase, ScannerRegistry

@pytest.fixture
def invalid_scanner():

    class InvalidScanner(ScannerBase):
        def __init__(self, logger: logging.Logger):
            super().__init__("InvalidScanner", "invalid", "Scanner to test for correct behavior to base class when dealing with improper inheitance", logger)

    logger = logging.getLogger("daisybell-test")
    invalid = InvalidScanner(logger)
    yield invalid

def test_base_can_scan(invalid_scanner):
    with pytest.raises(NotImplementedError):
        invalid_scanner.can_scan("fakemodel")

def test_base_scan(invalid_scanner):
    with pytest.raises(NotImplementedError):
        invalid_scanner.scan("fakemodel", {})
        