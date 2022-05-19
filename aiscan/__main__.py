import json
from argparse import ArgumentParser
from transformers import pipeline
from aiscan import scan
from tabulate import tabulate


def main():
    """The entry point of aiscan."""

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
    args = parser.parse_args()
    if args.huggingface and args.task:
        if args.params:
            params = args.params
        else:
            params = {}
        mask = pipeline(args.task, model=args.huggingface)
        print(f"Starting scan on model: {args.huggingface}...")
        for name, kind, desc, df in scan(mask, params):
            title = f"Results of {kind} scannner: {name} ({desc})"
            dashes = "=" * len(title)
            print(f"{title}\n{dashes}")
            print(tabulate(df, headers="keys", tablefmt="psql"))
