"""Command-line interface for the probabilistic method.

Provides ``geopfa-prob`` (entry point) and ``python -m geopfa.prob`` for
running the probabilistic method from a config file.

The PFA dict is loaded from a pickle path declared in the config JSON
under the ``pfa_pickle`` key — this is the serialised output of the
standard geoPFA pre-processing pipeline (notebook cell that saves
``pfa.pkl``).

Examples
--------
Run on a pre-processed PFA dict::

    geopfa-prob run --config path/to/config.json

The config JSON must contain a ``probabilistic`` block (see
``docs/probabilistic_method.md`` for the full schema) and a top-level
``pfa_pickle`` key pointing at the serialised PFA dict.
"""

from __future__ import annotations

import argparse
import json
import pickle  # noqa: S403 - PFA pickles come from trusted preprocessing
import sys
from pathlib import Path

from .config import load_probabilistic_config
from .runner import run_probabilistic


def _pfa_path_from_config(config_path: Path) -> Path:
    """Return the config-relative, canonical PFA pickle path."""
    cfg_dict = json.loads(config_path.read_text(encoding="utf-8"))
    pickle_path = cfg_dict.get("pfa_pickle")
    if not pickle_path:
        raise ValueError(
            "config has no 'pfa_pickle' key; add a top-level "
            "'pfa_pickle': '<path/to/pfa.pkl>' entry pointing at the "
            "serialised output of the geoPFA pre-processing pipeline",
        )
    path = Path(pickle_path)
    if not path.is_absolute():
        path = config_path.resolve().parent / path
    return path.resolve()


def _load_pfa_from_config(config_path: Path) -> dict:
    """Load the PFA dict referenced by the config file's ``pfa_pickle`` key.

    Raises
    ------
    ValueError
        If ``pfa_pickle`` is absent from the config.
    FileNotFoundError
        If the pickle file does not exist.
    """
    p = _pfa_path_from_config(config_path)
    if not p.exists():
        raise FileNotFoundError(f"pfa_pickle not found: {p}")
    with p.open("rb") as fh:
        return pickle.load(fh)  # noqa: S301 - PFA pickles are trusted


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="geopfa-prob",
        description="Run the geoPFA probabilistic method from a config file.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser(
        "run",
        help="Execute the probabilistic method on a pre-processed PFA dict.",
    )
    run.add_argument(
        "--config",
        type=Path,
        required=True,
        help=(
            "Path to a PFA-config JSON file containing a 'probabilistic' "
            "block and a top-level 'pfa_pickle' key."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code (0 on success)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "run":  # pragma: no cover - argparse handles this
        parser.print_help()
        return 2

    if not args.config.exists():
        print(f"error: config file not found: {args.config}", file=sys.stderr)
        return 1

    try:
        cfg = load_probabilistic_config(args.config)
        pfa_path = _pfa_path_from_config(args.config)
        pfa = _load_pfa_from_config(args.config)
        result = run_probabilistic(
            pfa,
            cfg,
            input_artifacts={"config": args.config, "pfa_pickle": pfa_path},
        )
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if result.skipped:
        print("probabilistic run skipped (config.enabled = false)")
    else:
        print(
            f"fitted {len(result.components)} component(s); "
            f"outputs written to {cfg.output_dir}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
