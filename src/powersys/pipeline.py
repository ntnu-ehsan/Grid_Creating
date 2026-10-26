"""Pipeline orchestration - the order in which the stages run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from . import figures as figures_mod
from . import iamc as iamc_mod
from . import maps as maps_mod
from . import report as report_mod
from .aggregate import Aggregated, aggregate_all
from .config import Config
from .demand import build_demand, summarise_demand
from .generation import expand_capacity_years, load_generation
from .grid import Grid, load_grid, summarise_grid
from .logging_utils import get_logger
from .manifest import write_manifest
from .regions import resolve_regions
from .validate import Report, run_checks

log = get_logger("pipeline")

MAPPING_DIR = Path(__file__).resolve().parents[2] / "config" / "mappings"


@dataclass
class Result:
    config: Config
    grid: Grid
    buses: pd.DataFrame
    generation: pd.DataFrame
    demand: pd.DataFrame
    aggregated: Aggregated
    iamc_table: pd.DataFrame
    validation: Report
    outputs: Dict[str, Path]

    @property
    def failed(self) -> bool:
        return self.validation.failed


def run(cfg: Config, mapping_dir: Optional[Path] = None) -> Result:
    mapping_dir = mapping_dir or MAPPING_DIR
    out_dir = cfg.run.dir
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: Dict[str, Path] = {}

    log.info("=" * 68)
    log.info("Power system data preparation | level=%s | countries=%s",
             cfg.spatial.level, ", ".join(cfg.countries))
    log.info("Output folder: %s", cfg.run.dir)
    log.info("=" * 68)

    # 1 - grid ------------------------------------------------------------
    log.info("[1/7] Loading the transmission grid")
    grid = load_grid(cfg)
    grid_summary = summarise_grid(grid)

    # 2 - generation ------------------------------------------------------
    log.info("[2/7] Loading generation units")
    generation = load_generation(cfg, grid, mapping_dir)
    if cfg.capacity_timeseries.enabled:
        generation = expand_capacity_years(generation, cfg.capacity_timeseries.years)
        log.info("Expanded capacity to years: %s",
                 ", ".join(str(y) for y in cfg.capacity_timeseries.years))

    # 3 - demand ----------------------------------------------------------
    log.info("[3/7] Building the demand distribution")
    demand = build_demand(cfg, grid)

    # 4 - regions ---------------------------------------------------------
    log.info("[4/7] Resolving spatial regions")
    buses, nuts_gdf = resolve_regions(cfg, grid)

    # 5 - aggregation -----------------------------------------------------
    log.info("[5/7] Aggregating to '%s'", cfg.spatial.level)
    aggregated = aggregate_all(cfg, buses, generation, demand,
                               grid.lines, grid.links, cfg.years)

    # 6 - IAMC ------------------------------------------------------------
    log.info("[6/7] Converting to IAMC and assembling the workbook")
    mapping = iamc_mod.load_iamc_mapping(mapping_dir)
    gen_iamc, unmapped = iamc_mod.generation_to_iamc(cfg, aggregated.generation,
                                                     mapping, cfg.years)
    demand_iamc = iamc_mod.demand_to_iamc(cfg, aggregated.demand)
    network_iamc = iamc_mod.network_to_iamc(cfg, aggregated.lines, aggregated.links, mapping)

    iamc_table = iamc_mod.assemble(cfg, {
        "generation": gen_iamc,
        "demand": demand_iamc,
        "network": network_iamc,
    })
    outputs.update(iamc_mod.write_iamc(cfg, iamc_table, out_dir / "iamc"))

    if cfg.outputs.csv:
        outputs.update(_write_tables(cfg, out_dir / "data", buses, generation,
                                     demand, aggregated, grid_summary))

    # 7 - validation and presentation -------------------------------------
    log.info("[7/7] Running checks and building the report")
    validation = run_checks(cfg, buses, generation, demand, aggregated,
                            iamc_table, unmapped)

    map_path = None
    if cfg.outputs.map_html:
        map_path = maps_mod.build_map(cfg, aggregated, buses, nuts_gdf,
                                      out_dir / "map.html")
        if map_path:
            outputs["map"] = map_path

    if cfg.outputs.report_html:
        figure_dir = out_dir / "figures" if cfg.outputs.figures else None
        rendered = figures_mod.build_all(aggregated.generation, aggregated.demand,
                                         cfg.years, figure_dir)
        report_path = report_mod.build_report(
            cfg, validation, aggregated, generation, demand, grid_summary,
            iamc_table, rendered, out_dir / "report.html",
            map_path=map_path, written={k: v for k, v in outputs.items() if k in ("xlsx", "csv")},
        )
        outputs["report"] = report_path

    # Written last: a crashed run must not claim the folder as its own.
    write_manifest(cfg, {"checks": validation.counts,
                         "iamc_rows": int(len(iamc_table))})

    log.info("Done. Outputs in %s", out_dir.resolve())
    return Result(cfg, grid, buses, generation, demand, aggregated,
                  iamc_table, validation, outputs)


def _write_tables(cfg: Config, data_dir: Path, buses: pd.DataFrame,
                  generation: pd.DataFrame, demand: pd.DataFrame,
                  aggregated: Aggregated, grid_summary: pd.DataFrame) -> Dict[str, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "buses": buses,
        "generation_nodal": generation,
        "demand_nodal": demand,
        "generation_aggregated": aggregated.generation,
        "demand_aggregated": aggregated.demand,
        "lines_aggregated": aggregated.lines,
        "links_aggregated": aggregated.links,
        "regions": aggregated.regions,
        "grid_summary": grid_summary,
        "demand_summary": summarise_demand(demand),
    }
    written: Dict[str, Path] = {}
    for name, frame in tables.items():
        path = data_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path
    log.info("Wrote %d CSV tables -> %s", len(written), data_dir)
    return written
