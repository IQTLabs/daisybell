from aiscan import scan
from transformers import pipeline


def test_scan():
    res = scan(pipeline("fill-mask", model="roberta-base"))
    assert list(res)[0][0] == "masking-human-bias"
    res = scan(
        pipeline(
            "zero-shot-classification", model="cross-encoder/nli-distilroberta-base"
        )
    )
    assert list(res)[0][0] == "zero-shot-human-bias"
