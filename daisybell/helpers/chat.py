import torch
from transformers import Pipeline, StoppingCriteria, StoppingCriteriaList


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class ChatBot:
    system_prompt: str = """<|SYSTEM|># Assistant
        - Assistant is a helpful and harmless open-source AI language model.
        - Assistant is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
        - Assistant is more than just an information source, Assistant is also able to write poetry, short stories, and make jokes.
        - Assistant will refuse to participate in anything that could harm a human.
        """

    def __init__(self, pipeline: Pipeline) -> None:
        if pipeline.task != "text-generation":
            raise ValueError("Pipeline must be a text-generation pipeline")
        self.pipeline = pipeline

    def chat(self, prompt: str) -> str:
        prompt = f"{self.system_prompt}<|USER|>{prompt}<|ASSISTANT|>"
        return (
            self.pipeline(
                prompt,
                max_length=256,
                pad_token_id=self.pipeline.tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )[0]["generated_text"]
            .replace(prompt, "")
            .strip()
        )
