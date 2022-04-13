from aiscan import scan
from transformers import pipeline


def test_scan():
    res = scan(pipeline("fill-mask", model="roberta-base"))
    assert list(res)[0][0] == "masking-human-bias"
