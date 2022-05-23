from curses import meta
from typing import Tuple, Dict, Any, List
from os import PathLike
from pathlib import Path
import json
from argparse import ArgumentParser
from transformers import pipeline
from tabulate import tabulate
import pandas as pd

from daisybell import scan


def create_file_output(
    scan_output: Tuple[str, str, str, pd.DataFrame],
    output_path: PathLike,
    model_name: str,
    model_params: dict = None,
):
    """
    Creates scanner output as human and machine readable files.

    Parameters:
        scan_output: The output of a scan().
        output_path: A directory to put the files.
        model_name: The name of the model.
        model_params: Any parameters for the scanners and model.
    """
    Path(output_path).mkdir(exist_ok=True, parents=True)
    model_metadata: Dict[str, Any] = dict()
    model_metadata["name"] = model_name
    if model_params:
        model_metadata["params"] = model_params
    model_metadata["scanners"] = list()

    for name, kind, desc, df in scan_output:
        scan_metadata: Dict[str, Any] = dict()
        scan_metadata["name"] = name
        scan_metadata["kind"] = kind
        scan_metadata["description"] = desc
        model_metadata["scanners"].append(scan_metadata)
        df.to_csv(Path(output_path) / f"{name}.csv", index=False)

    with open(Path(output_path) / "metadata.json", "w") as fd:
        json.dump(model_metadata, fd)


def main():
    """The entry point of daisybell."""

    parser = ArgumentParser(
        description="Scans machine learning models for AI Assurance problems"
    )
    parser.add_argument(
        "--huggingface", action="store", help="name of HuggingFace model to scan"
    )
    parser.add_argument("--task", action="store", help="what HuggingFace task to try")
    parser.add_argument(
        "--params",
        action="store",
        help="Configure the behavior of scanners",
        type=json.loads,
    )
    parser.add_argument(
        "--output",
        action="store",
        help="Output scanner results to a directory",
    )

    args = parser.parse_args()
    if args.huggingface and args.task:
        if args.params:
            params = args.params
        else:
            params = {}
        mask = pipeline(args.task, model=args.huggingface)
        print(f"Starting scan on model: {args.huggingface}...")
        scan_output = list(scan(mask, params))
        for name, kind, desc, df in scan_output:
            title = f"Results of {kind} scannner: {name} ({desc})"
            dashes = "=" * len(title)
            print(f"{title}\n{dashes}")
            print(tabulate(df, headers="keys", tablefmt="psql"))
        if args.output:
            print(f"Saving output to {args.output}...")
            create_file_output(scan_output, args.output, args.huggingface, args.params)
