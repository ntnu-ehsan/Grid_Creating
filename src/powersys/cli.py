"""Command line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .config import ConfigError, load_config
from .logging_utils import setup_logging
from .manifest import OutputConflictError, guard_output_dir

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
DEFAULT_CASE = "nesp"


def available_cases() -> list:
    """Case names, taken from the config files that are actually present."""
    if not CONFIG_DIR.is_dir():
        return []
    return sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))


def build_parser() -> argparse.ArgumentParser:
    cases = available_cases()
    parser = argparse.ArgumentParser(
        prog="powersys",
        description="Power system data preparation for the iDesignRES case studies, "
                    "producing IAMC-formatted output, an HTML report and an interactive map.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--case", default=DEFAULT_CASE,
                        help=f"Case study to run - one of: {', '.join(cases) or 'none found'}. "
                             f"Resolves to config/<case>.yaml")
    parser.add_argument("--config", "-c", type=Path, default=None,
                        help="Path to a YAML configuration file (overrides --case)")
    parser.add_argument("--list-cases", action="store_true",
                        help="List the available case studies and exit")

    override = parser.add_argument_group("config overrides")
    override.add_argument("--level", choices=["nodal", "national", "nuts2", "nuts3"],
                          help="Spatial resolution of the output")
    override.add_argument("--model", help="IAMC model name")
    override.add_argument("--scenario", help="IAMC scenario name")
    override.add_argument("--name", help="Run name (output folder)")
    override.add_argument("--countries", help="Comma-separated ISO2 codes, e.g. NO,SE")
    override.add_argument("--years", help="Comma-separated study years, e.g. 2020,2030")
    override.add_argument("--no-timeseries", action="store_true",
                          help="Write a single capacity snapshot instead of a time series")
    override.add_argument("--no-map", action="store_true", help="Skip the interactive map")
    override.add_argument("--no-report", action="store_true", help="Skip the HTML report")
    override.add_argument("--log-level", default=None,
                          choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--strict", action="store_true",
                        help="Exit with a non-zero status if any validation check fails")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output folder even if it holds a different run")
    return parser


def _overrides(args: argparse.Namespace) -> dict:
    values = {
        "spatial.level": args.level,
        "iamc.model": args.model,
        "iamc.scenario": args.scenario,
        "run.name": args.name,
        "logging.level": args.log_level,
    }
    if args.countries:
        values["countries"] = [c.strip().upper() for c in args.countries.split(",") if c.strip()]
    if args.years:
        values["capacity_timeseries.years"] = [int(y) for y in args.years.split(",") if y.strip()]
    if args.no_timeseries:
        values["capacity_timeseries.enabled"] = False
    if args.no_map:
        values["outputs.map_html"] = False
    if args.no_report:
        values["outputs.report_html"] = False
    return values


def resolve_config(args: argparse.Namespace) -> Path:
    """--config wins; otherwise --case picks config/<case>.yaml."""
    if args.config is not None:
        return args.config

    path = CONFIG_DIR / f"{args.case}.yaml"
    if not path.exists():
        cases = available_cases()
        raise ConfigError(
            f"Unknown case '{args.case}'. Available: {', '.join(cases) or 'none'}. "
            f"Use --config to point at a file elsewhere."
        )
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_cases:
        for case in available_cases():
            print(case)
        return 0

    try:
        config_path = resolve_config(args)
        cfg = load_config(config_path, _overrides(args))
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    # Before setup_logging: it opens pipeline.log in "w" mode, which would
    # truncate the previous run's log before anyone could be warned about it.
    try:
        guard_output_dir(cfg, force=args.force)
    except OutputConflictError as exc:
        print(f"Refusing to overwrite: {exc}", file=sys.stderr)
        return 5

    log = setup_logging(cfg.logging_level, cfg.run.dir / "pipeline.log")

    from .pipeline import run  # imported late so config errors surface first

    try:
        result = run(cfg)
    except FileNotFoundError as exc:
        log.error("Missing input: %s", exc)
        return 3
    except Exception as exc:  # pragma: no cover
        log.exception("Pipeline failed: %s", exc)
        return 1

    counts = result.validation.counts
    print()
    print(f"  IAMC rows     : {len(result.iamc_table):,}")
    print(f"  Regions       : {result.aggregated.regions.query('not is_external').shape[0]:,}")
    print(f"  Checks        : {counts['pass']} passed, {counts['warn']} warnings, "
          f"{counts['fail']} failed")
    for key, path in result.outputs.items():
        if key in ("xlsx", "csv", "report", "map"):
            print(f"  {key:<14}: {path}")

    if args.strict and result.failed:
        return 4
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
