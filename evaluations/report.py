"""Parses the JSON files produced by ehri_chat/evaluate.py and prints them as a table.

Usage:
    python -m evaluations.report
    python -m evaluations.report --format csv --output evaluations/report.csv
    python -m evaluations.report --model gemini --mode rag
"""

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

EVALUATIONS_DIR = Path(__file__).resolve().parent
JUDGE_DIR = EVALUATIONS_DIR / "llm_as_a_judge"
TOKENS_DIR = EVALUATIONS_DIR / "tokens_usage"

REPORT_SUFFIX = "_report.json"
TOKENS_SUFFIX = "_tokens_usage.json"

# e.g. query_camps_with_archive_germany_rag_gemini_2026-06-18_16.16.22.326487
BASENAME_RE = re.compile(
    r"^query_(?P<query>.+)_(?P<mode>vanilla|rag|graphrag|mcp)_(?P<model>[a-zA-Z0-9]+)_"
    r"\d{4}-\d{2}-\d{2}_\d{2}\.\d{2}\.\d{2}\.\d+$"
)

HEADERS = ["model", "mode", "query", "CR", "AR", "G", "tokens in", "tokens out"]

# Mirrors the order of the `queries` dict in ehri_chat/evaluate.py so reports
# read in the same order the evaluation run was defined in. Keep in sync.
QUERY_ORDER = [
    "situation_belgium",
    "roma_fate_france",
    "comparison_countries",
    "listing_institutions",
    "museums_holocaust",
    "camps_with_archive_germany",
    "deportations_antwerp",
    "raids_amsterdam",
    "visas_countries",
    "person_information",
]


@dataclass
class Row:
    model: str
    mode: str
    query: str
    cr: str
    ar: str
    g: str
    tokens_in: int
    tokens_out: int

    def as_list(self):
        return [self.model, self.mode, self.query, self.cr, self.ar, self.g, self.tokens_in, self.tokens_out]


def load_rows() -> list[Row]:
    rows = []
    for report_path in sorted(JUDGE_DIR.glob(f"*{REPORT_SUFFIX}")):
        basename = report_path.name[: -len(REPORT_SUFFIX)]
        match = BASENAME_RE.match(basename)
        if not match:
            print(f"Skipping unrecognized report filename: {report_path.name}", file=sys.stderr)
            continue

        tokens_path = TOKENS_DIR / f"{basename}{TOKENS_SUFFIX}"
        if not tokens_path.exists():
            print(f"Skipping {report_path.name}: no matching tokens usage file", file=sys.stderr)
            continue

        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            report = json.loads(report_path.read_text(encoding="cp1252"))

        try:
            tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            tokens = json.loads(tokens_path.read_text(encoding="cp1252"))


        rows.append(Row(
            model=match["model"],
            mode=match["mode"],
            query=match["query"],
            cr=report["context_relevance"]["score"],
            ar=report["answer_relevance"]["score"],
            g=report["groundedness"]["score"],
            tokens_in=tokens["input_tokens"],
            tokens_out=tokens["output_tokens"],
        ))
    return rows


def filter_rows(rows: list[Row], model: str | None, mode: str | None, query: str | None) -> list[Row]:
    if model:
        rows = [r for r in rows if r.model == model]
    if mode:
        rows = [r for r in rows if r.mode == mode]
    if query:
        rows = [r for r in rows if r.query == query]
    return rows


def query_sort_key(query: str):
    """Position in QUERY_ORDER, falling back to alphabetical for unknown queries."""
    try:
        return (QUERY_ORDER.index(query), "")
    except ValueError:
        return (len(QUERY_ORDER), query)


def sort_rows(rows: list[Row], sort_by: str) -> list[Row]:
    if sort_by == "query":
        return sorted(rows, key=lambda r: query_sort_key(r.query))

    attr_by_header = {
        "model": "model", "mode": "mode", "cr": "cr", "ar": "ar", "g": "g",
        "tokens in": "tokens_in", "tokens out": "tokens_out",
    }
    attr = attr_by_header.get(sort_by, "model")
    # Query order is always the secondary sort key, so rows within the same
    # model/mode/score group still read in the canonical query order.
    return sorted(rows, key=lambda r: (getattr(r, attr), query_sort_key(r.query)))


def print_table(rows: list[Row]):
    table = [HEADERS] + [[str(v) for v in r.as_list()] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(HEADERS))]

    def format_row(row):
        return " | ".join(value.ljust(widths[i]) for i, value in enumerate(row))

    separator = "-+-".join("-" * w for w in widths)
    print(format_row(table[0]))
    print(separator)
    for row in table[1:]:
        print(format_row(row))


def write_csv(rows: list[Row], output: Path):
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        for r in rows:
            writer.writerow(r.as_list())


def main():
    parser = argparse.ArgumentParser(description="Tabulate EHRI chat evaluation results.")
    parser.add_argument("--model", help="Filter by model/provider (e.g. mistral, gemini)")
    parser.add_argument("--mode", help="Filter by mode (vanilla, rag, graphrag, mcp)")
    parser.add_argument("--query", help="Filter by query key (e.g. comparison_countries)")
    parser.add_argument("--sort-by", default="query", choices=[h.lower() for h in HEADERS],
                         help="Column to sort by")
    parser.add_argument("--format", default="table", choices=["table", "csv"], help="Output format")
    parser.add_argument("--output", type=Path, help="Write output to this file instead of stdout")
    args = parser.parse_args()

    rows = load_rows()
    rows = filter_rows(rows, args.model, args.mode, args.query)
    rows = sort_rows(rows, args.sort_by)

    if not rows:
        print("No matching evaluation results found.", file=sys.stderr)
        return

    if args.format == "csv":
        if args.output:
            write_csv(rows, args.output)
        else:
            writer = csv.writer(sys.stdout)
            writer.writerow(HEADERS)
            for r in rows:
                writer.writerow(r.as_list())
    else:
        if args.output:
            import contextlib
            with args.output.open("w", encoding="utf-8") as f, contextlib.redirect_stdout(f):
                print_table(rows)
        else:
            print_table(rows)


if __name__ == "__main__":
    main()
