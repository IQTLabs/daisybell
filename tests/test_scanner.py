import os 

import pytest

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
        pipeline(model="Davlan/xlm-roberta-base-ner-hrl"),
        params={"max_books": 3, "max_sentences_per_book": 10},
    )
    name, kind, _, df = list(res)[0]
    assert name == "ner-human-language-bias"
    assert kind == "bias"
    assert len(df) > 50


@pytest.mark.skipif(
    "GITHUB_ACTIONS" in os.environ, reason="This test is too slow for GitHub Actions"
)
def test_chatbot_ai_alignment():
    res = scan(
        pipeline(model="StabilityAI/stablelm-tuned-alpha-3b"),
        params={},
    )
    name, kind, _, df = list(res)[0]
    assert name == "chatbot-ai-alignment"
    assert kind == "alignment"
    assert df.iloc[0]["score"] > 0.5
