from typing import Tuple

import pandas as pd

from daisybell.helpers.dataset import handle_wikidata_dataset


def handle_common_params_to_masking_and_zeroshot(
    params: dict,
) -> Tuple[str, int, pd.DataFrame]:
    """
    Handles the common parameters to masking and zeroshot scanners.

    Parameters:
        params: The parameters passed to daisybell.

    Returns:
        A tuple of the suffix, the maximum number of names per language, and the wikidata dataframe.
    """
    if params.get("suffix"):
        suffix = params["suffix"]
    else:
        suffix = ""
    if params.get("max_names_per_language"):
        max_names_per_language = params["max_names_per_language"]
    else:
        max_names_per_language = 999999999  # If this number is exceeded we got bigger problems
    return suffix, max_names_per_language, pd.read_csv(handle_wikidata_dataset(params))
