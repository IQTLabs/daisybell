from collections import defaultdict
from copy import deepcopy
from logging import Logger

import pandas as pd
import pysbd
from seqeval.metrics import recall_score
from tqdm.auto import tqdm
from transformers import Pipeline

from daisybell.helpers.common import handle_common_params_to_masking_and_zeroshot
from daisybell.helpers.dataset import emit_books, replace_entities
from daisybell.scanners import ScannerBase

NAME = "ner-human-language-bias"
KIND = "bias"
DESCRIPTION = "Scanning for language bias in NER based models. WARNING! THIS SCANNER IS EXPERIMENTAL."


class NerLanguageBias(ScannerBase):
    def __init__(self, logger: Logger):
        super().__init__(NAME, KIND, DESCRIPTION, logger)

    def can_scan(self, model: Pipeline) -> bool:
        try:
            return model.task == "token-classification"
        except Exception:
            return False

    def scan(self, model: Pipeline, params: dict) -> pd.DataFrame:
        (
            suffix,
            max_names_per_language,
            wikidata,
        ) = handle_common_params_to_masking_and_zeroshot(params)

        max_sentences_per_book = params.get("max_sentences_per_book", 999999999)
        max_books = params.get("max_books", 999999999)

        language_names = {}
        for language in wikidata["language"].unique():
            # get the names for the current language (up to the maximum number specified)
            names = wikidata[wikidata["language"] == language]["name"].iloc[
                :max_names_per_language
            ]
            # filter out names that are not one word
            names = names[names.str.count(" ") == 0]
            # Some languages just don't have enough examples, this skips them
            if len(names) < 10:
                continue
            language_names[language] = names

        splitter = pysbd.Segmenter(language="en", clean=True)
        control_entities = defaultdict(list)
        test_entities = defaultdict(list)
        for book_idx, (book_name, book_content) in enumerate(emit_books(params), 1):
            print(f"Scanning using {book_name}...")
            text = splitter.segment(book_content)
            for sent_idx, sentence in enumerate(
                tqdm(
                    text,
                    total=min(len(text), max_sentences_per_book),
                    desc="Sentences",
                    unit="sent",
                )
            ):
                # control ner output
                control = []
                for entity in model(sentence):
                    if entity["entity"] == "B-PER":
                        control.append(entity)
                control_entity = [entity["entity"] for entity in control]
                print(f"==> Control entity: {control_entity}")

                for language in language_names.keys():
                    transformed = replace_entities(
                        sentence,
                        (
                            (
                                entity["start"],
                                entity["end"],
                                language_names[language]
                                .sample(random_state=18 + i)
                                .item()
                                + suffix,
                            )
                            for i, entity in enumerate(control)
                        ),
                    )
                    test_result = []
                    for entity in model(transformed):
                        if entity["entity"] == "B-PER":
                            test_result.append(entity["entity"])

                    new_control_entity = deepcopy(control_entity)
                    if len(control_entity) == 0 and len(test_result) == 0:
                        continue
                    elif len(control_entity) > len(test_result):
                        test_result += ["O"] * (len(control_entity) - len(test_result))
                    elif len(control_entity) < len(test_result):
                        new_control_entity += ["O"] * (
                            len(test_result) - len(control_entity)
                        )
                    control_entities[language].append(new_control_entity)
                    test_entities[language].append(test_result)

                if sent_idx == max_sentences_per_book:
                    break
            if book_idx == max_books:
                break

        return (
            pd.DataFrame(
                {
                    "Language": language_names.keys(),
                    "Recall": (
                        recall_score(
                            control_entities[language],
                            test_entities[language],
                            average="macro",
                        )
                        for language in language_names.keys()
                    ),
                }
            )
            .sort_values("Recall")
            .reset_index(drop=True)
        )
