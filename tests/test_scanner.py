import pytest

from daisybell import scan
from transformers import pipeline


@pytest.fixture
def fake_chat_bot():
    """
    This is a fake chat bot that always returns the same response.
    We use it because a real chat bot would be too slow to run in a test.
    """

    class FakeTokenizer:
        eos_token_id = None

    class FakeChatBot:
        task = "text-generation"
        tokenizer = FakeTokenizer()

        def __call__(self, *args, **kwds):
            return [{"generated_text": "ethical and moral values"}]

    yield FakeChatBot()


def test_scanning_masking_human_bias():
    res = scan(
        pipeline(model="roberta-base"),
        params={"max_names_per_language": 10},
    )
    name, kind, _, result = list(res)[0]
    assert name == "masking-human-language-bias"
    assert kind == "bias"
    assert "scores" in result
    assert "details" in result
    rows, cols = result["details"][0]["df"].shape
    assert cols == 2
    assert rows > 20


def test_scanning_zero_shot_human_bias():
    res = scan(
        pipeline(model="cross-encoder/nli-distilroberta-base"),
        params={"max_names_per_language": 10},
    )
    name, kind, _, result = list(res)[0]
    assert name == "zero-shot-human-language-bias"
    assert kind == "bias"
    assert "scores" in result
    assert "details" in result
    rows, cols = result["details"][0]["df"].shape
    assert cols == 2
    assert rows > 20


def test_scanning_ner_human_language_bias():
    res = scan(
        pipeline(model="Davlan/xlm-roberta-base-ner-hrl"),
        params={"max_books": 3, "max_sentences_per_book": 10},
    )
    name, kind, _, result = list(res)[0]
    assert name == "ner-human-language-bias"
    assert kind == "bias"
    assert "scores" in result
    assert "details" in result
    rows, cols = result["details"][0]["df"].shape
    assert cols == 2
    assert rows > 20


def test_chatbot_ai_alignment(fake_chat_bot):
    res = scan(
        fake_chat_bot,
        params={},
    )
    name, kind, _, result = list(res)[0]
    assert name == "chatbot-ai-alignment"
    assert kind == "alignment"
    assert "scores" in result
    assert {
        "scores": [
            {"name": "simple alignment score", "score": 1.0},
            {"name": "jailbreak alignment score", "score": 1.0},
        ]
        == result["scores"]
    }
