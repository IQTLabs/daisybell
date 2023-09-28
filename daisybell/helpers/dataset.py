import json
import os
import tarfile
from pathlib import Path
from typing import Generator, Optional, List, Tuple
from urllib.request import urlretrieve


def handle_dataset(url: str, alterative_path: Optional[str] = None) -> os.PathLike:
    """
    Handles the dataset.

    Parameters:
        url: The url of the dataset.
        alterative_path: An alternative path to the dataset if not using the default (~/.iqtlabs).

    Returns:
        The path to the dataset.
    """
    if alterative_path:
        output_path = Path(alterative_path)
    else:
        (Path.home() / ".iqtlabs").mkdir(exist_ok=True)
        output_path = Path.home() / ".iqtlabs" / os.path.basename(url)
    if not output_path.exists():
        urlretrieve(
            url,
            output_path,
        )
    return output_path


def handle_books_dataset(params: dict) -> os.PathLike:
    """
    Downloads the books dataset or provides the cached copy.

    Parameters:
        params: The parameters passed to daisybell.

    Returns:
        A pandas DataFrame with the books dataset.
    """
    books_url = "https://iqtlabs-aia-datasets.s3.amazonaws.com/public_domain_books.tar.gz"
    return handle_dataset(books_url, params.get("books_path"))


def handle_wikidata_dataset(params: dict) -> os.PathLike:
    """
    Downloads the wikidata dataset or provides the cached copy.

    Parameters:
        params: The parameters passed to daisybell.

    Returns:
        A pandas DataFrame with the wikidata dataset.
    """
    wikidata_url = "https://iqtlabs-aia-datasets.s3.amazonaws.com/wikidata_person_names-v1.csv.gz"
    return handle_dataset(wikidata_url, params.get("wikidata_person_names_path"))


def emit_books(params: dict) -> Generator:
    """
    Emit the books in a tar file.
    books_path: The path to the tar file.
    :return: An iterator of tuples of the form (book_name, book_content).
    """
    with tarfile.open(handle_books_dataset(params)) as tar:
        for member in tar.getmembers():
            yield member.name, json.loads(tar.extractfile(member).read())[0]["content"]  # pyright: ignore


def replace_entities(text: str, substrings: List[Tuple[int, int, str]]) -> str:
    """
    Replace substrings in a string with a given replacement string.
    :param text: The string to replace substrings in.
    :param substrings: An iterator of tuples of the form (start, end, replacement).
    :return: The string with the substrings replaced.
    """
    # Sort the substrings by start index. This allows us to replace substrings
    # in linear time without making new copies of the string per replacement.
    # When a string is an entire book, this can save a lot of time.
    substrings = sorted(substrings, key=lambda x: x[0])
    result = []
    current_index = 0
    for start, end, substring in substrings:
        result.append(text[current_index : start + 1])
        result.append(substring)
        current_index = end
    result.append(text[current_index:])
    return "".join(result)
