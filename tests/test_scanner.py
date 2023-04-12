import logging

from daisybell import scan
from transformers import pipeline


def test_scanning_masking_human_bias():
    res = scan(
        pipeline(model="roberta-base"),
        params={"max_names_per_language": 10},
    )
    name, kind, _, df = list(res)[0]
    assert name == "masking-human-language-bias"
    assert kind == "bias"
    assert len(df) > 50


def test_scanning_zero_shot_human_bias():
    res = scan(
        pipeline(model="cross-encoder/nli-distilroberta-base"),
        params={"max_names_per_language": 10},
    )
    name, kind, _, df = list(res)[0]
    assert name == "zero-shot-human-language-bias"
    assert kind == "bias"
    assert len(df) > 50


def test_scanning_ner_human_language_bias():
    res = scan(
        pipeline(
            model="Davlan/xlm-roberta-base-ner-hrl",
            tokenizer="Davlan/xlm-roberta-base-ner-hrl",
        ),
        params={"max_books": 3, "max_sentences_per_book": 10},
    )
    name, kind, _, df = list(res)[0]
    assert name == "ner-human-language-bias"
    assert kind == "bias"
    assert len(df) > 50
