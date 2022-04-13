from argparse import ArgumentParser
from transformers import pipeline
from aiscan import scan


def main():
    """The entry point of aiscan."""

    parser = ArgumentParser(
        description="Scans machine learning models for AI Assurance problems"
    )
    parser.add_argument(
        "--huggingface", action="store", help="name of HuggingFace model to scan"
    )
    args = parser.parse_args()
    if args.huggingface:
        mask = pipeline("fill-mask", model="roberta-base")
        print(f"Scanning on {args.huggingface}")
        print(list(scan(mask)))
