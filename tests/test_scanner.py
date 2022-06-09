from daisybell import scan
from transformers import pipeline


def test_scanning_masking_human_bias():
    res = scan(
        pipeline("fill-mask", model="roberta-base"),
        params={"max_names_per_language": 10},
    )

    name, kind, _, df = list(res)[0]
    assert name == "masking-human-language-bias"
    assert kind == "bias"
    assert len(df) > 50


def test_scanning_zero_shot_human_bias():
    res = scan(
        pipeline(
            "zero-shot-classification", model="cross-encoder/nli-distilroberta-base"
        ),
        params={"max_names_per_language": 10},
    )
    name, kind, _, df = list(res)[0]
    assert name == "zero-shot-human-language-bias"
    assert kind == "bias"
    assert len(df) > 50
