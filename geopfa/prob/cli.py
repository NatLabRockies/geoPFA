"""Command-line entry point for a processed, configuration-driven PFA run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_probabilistic_config
from .data import load_processed_pfa
from .runner import run_probabilistic


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geopfa-prob",
        description="Run the probabilistic method from processed geoPFA data.",
    )
    run = parser.add_subparsers(dest="command", required=True).add_parser(
        "run",
        help="Load processed layers and execute the probabilistic model.",
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        help="PFA JSON containing criteria and a probabilistic block.",
    )
    run.add_argument(
        "--processed-data-dir",
        type=Path,
        required=True,
        help="Root of the criteria/component/*_processed.csv layer tree.",
    )
    run.add_argument(
        "--crs",
        required=True,
        help="CRS assigned to processed layer coordinates (for example EPSG:26911).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its process exit code."""
    args = _build_parser().parse_args(argv)
    config_path = args.config.resolve()
    data_dir = args.processed_data_dir.resolve()
    if not config_path.is_file():
        print(f"error: config file not found: {config_path}", file=sys.stderr)
        return 1
    try:
        config = load_probabilistic_config(config_path)
        pfa, artifacts = load_processed_pfa(
            config_path, data_dir, crs=args.crs
        )
        result = run_probabilistic(
            pfa,
            config,
            input_artifacts=artifacts,
        )
    except (TypeError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.skipped:
        print("probabilistic run skipped (config.enabled = false)")
    else:
        print(
            f"fitted {len(result.components)} component(s); "
            f"outputs written to {config.output_dir}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
