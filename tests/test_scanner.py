from daisybell import scan
from transformers import pipeline


def test_scanning_masking_human_bias():
    res = scan(pipeline("fill-mask", model="roberta-base"))
    assert list(res)[0][0] == "masking-human-language-bias"


def test_scanning_zero_shot_human_bias():
    res = scan(
        pipeline(
            "zero-shot-classification", model="cross-encoder/nli-distilroberta-base"
        )
    )
    assert list(res)[0][0] == "zero-shot-human-bias"
