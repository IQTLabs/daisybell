from typing import Union, Tuple, Optional, List, Dict, Any
from os import PathLike
from pathlib import Path
import json
import warnings
from argparse import ArgumentParser
from transformers import pipeline
from tabulate import tabulate
import pandas as pd

from daisybell import scan

warnings.filterwarnings("ignore", category=UserWarning)


def create_file_output(
    scan_output: Union[Tuple[str, str, str, pd.DataFrame], List[Any]],
    output_path: PathLike,
    model_name: str,
    model_params: Optional[dict] = None,
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

    for name, kind, desc, result in scan_output:
        scan_metadata: Dict[str, Any] = dict()
        scan_metadata["name"] = name
        scan_metadata["kind"] = kind
        scan_metadata["description"] = desc
        model_metadata["scanners"].append(scan_metadata)
        if isinstance(result, pd.DataFrame):
            result.to_csv(Path(output_path) / f"{name}.csv", index=False)
        elif isinstance(result, dict):
            for detail in result.get("details", []):
                detail["df"].to_csv(
                    Path(output_path) / f"{name}_{detail['name'].replace(' ', '_')}.csv",
                    index=False,
                )
            with open(Path(output_path) / f"{name}_scores.csv", "w") as fd:
                fd.write("name,score\n")
                for score in result["scores"]:
                    fd.write(f"{score['name']},{score['score']}\n")
    with open(Path(output_path) / "metadata.json", "w") as fd:
        json.dump(model_metadata, fd)


def main():  # noqa C901
    """The entry point of daisybell."""

    parser = ArgumentParser(description="Scans machine learning models for AI Assurance problems")
    parser.add_argument("model", action="store", help="name of HuggingFace model to scan")
    parser.add_argument("--task", "-t", action="store", help="what HuggingFace task to try")
    parser.add_argument(
        "--params",
        "-p",
        action="store",
        help="Configure the behavior of scanners",
        type=json.loads,
    )
    parser.add_argument(
        "--output",
        "-o",
        action="store",
        help="Output scanner results to a directory",
    )
    parser.add_argument(
        "--tokenizer",
        "-k",
        action="store",
        help="name of HuggingFace tokenizer to use (only when --task/-t is set)",
    )
    parser.add_argument(
        "--device",
        "-d",
        action="store",
        default="cpu",
        help="device to use for scanning (default: cpu)",
    )
    parser.add_argument(
        "--score-only",
        "-s",
        action="store_true",
        help="only output the scores of scanners to standard output, in comma-separated format",
    )

    args = parser.parse_args()
    if args.params:
        params = args.params
    else:
        params = {}
    if args.task:
        if args.tokenizer:
            model = pipeline(
                args.task,
                model=args.model,
                tokenizer=args.tokenizer,
                device=args.device,
            )
        else:
            model = pipeline(args.task, model=args.model, device=args.device)
    else:
        model = pipeline(model=args.model, device=args.device)
    print(f"Starting scan on model: {args.model}...")
    scan_output = list(scan(model, params))
    for name, kind, desc, result in scan_output:
        if not args.score_only:
            title = f"Results of {kind} scannner: {name} ({desc})"
            dashes = "=" * len(title)
            print(f"{title}\n{dashes}")
        if isinstance(result, pd.DataFrame):
            print(tabulate(result, headers="keys", tablefmt="psql"))
        elif isinstance(result, dict):
            if args.score_only:
                for score in result["scores"]:
                    print(f"{name},{score['name']},{score['score']}")
                continue
            for detail in result.get("details", []):
                print(f"{detail['name']}:\n{tabulate(detail['df'], headers='keys', tablefmt='psql')}")
            print(f"Scores:\n{tabulate(result['scores'], headers='keys', tablefmt='psql')}")
    if args.output:
        print(f"Saving output to {args.output}...")
        create_file_output(scan_output, args.output, args.model, args.params)
