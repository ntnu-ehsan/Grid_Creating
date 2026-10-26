"""Conversion to IAMC format and assembly of the single delivery workbook.

Everything is written in long format:

    model | scenario | region | variable | unit | year | value

Generation capacity, demand shares and network capacities all share that
schema, so the final workbook is a plain concatenation - no manual merging
or reshaping in Excel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

from .config import Config
from .logging_utils import get_logger

log = get_logger("iamc")

IAMC_COLUMNS = ["model", "scenario", "region", "variable", "unit", "year", "value"]


def load_iamc_mapping(mapping_dir: Path) -> Dict:
    with open(mapping_dir / "iamc_variables.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def map_generation_variable(fueltype: str, technology: str, mapping: Dict) -> Tuple[str, bool]:
    """Return (IAMC variable, mapped?) for a fuel/technology pair.

    Falls back from the exact pair to fuel-only, then technology-only, and
    finally to the configured fallback - mirroring how the source datasets
    leave technology blank for conventional plant.
    """
    variables = mapping.get("variables", {})
    prefix = mapping.get("prefix", "Capacity|Electricity|")
    fallback = mapping.get("fallback", "Other")

    fuel = (fueltype or "").strip()
    tech = (technology or "").strip()

    for key in (f"{fuel}|{tech}", f"{fuel}|", f"|{tech}"):
        if key in variables:
            return prefix + variables[key], True
    return prefix + fallback, False


def generation_to_iamc(cfg: Config, generation: pd.DataFrame, mapping: Dict,
                       years: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Long-format capacity rows plus a table of unmapped fuel/technology pairs."""
    frame = generation.copy()
    mapped = frame.apply(
        lambda row: map_generation_variable(row["fueltype"], row["technology"], mapping),
        axis=1, result_type="expand",
    )
    frame["variable"] = mapped[0]
    frame["is_mapped"] = mapped[1]

    unmapped = (frame[~frame["is_mapped"]]
                .groupby(["fueltype", "technology"], as_index=False)["capacity_mw"].sum()
                .sort_values("capacity_mw", ascending=False))
    if not unmapped.empty:
        log.warning("%d fuel/technology combinations (%.0f MW) fell back to 'Other'",
                    len(unmapped), unmapped["capacity_mw"].sum())

    records: List[pd.DataFrame] = []
    if cfg.capacity_timeseries.enabled:
        for year in years:
            column = str(year)
            if column not in frame.columns:
                log.warning("Year column '%s' missing from the generation table - skipped", column)
                continue
            records.append(pd.DataFrame({
                "region": frame["region"],
                "variable": frame["variable"],
                "unit": "MW",
                "year": year,
                "value": frame[column],
            }))
    else:
        records.append(pd.DataFrame({
            "region": frame["region"],
            "variable": frame["variable"],
            "unit": "MW",
            "year": cfg.iamc.reference_year,
            "value": frame["capacity_mw"],
        }))

    table = _finalise(cfg, pd.concat(records, ignore_index=True))
    log.info("Generation -> %d IAMC rows over %d year(s)", len(table), table["year"].nunique())
    return table, unmapped


def demand_to_iamc(cfg: Config, demand: pd.DataFrame) -> pd.DataFrame:
    """Load distribution as a percentage share per region."""
    table = pd.DataFrame({
        "region": demand["region"],
        "variable": cfg.iamc.demand_variable,
        "unit": cfg.iamc.demand_unit,
        "year": cfg.iamc.reference_year,
        "value": demand["load_share"] * 100.0,
    })
    table = _finalise(cfg, table)
    log.info("Demand -> %d IAMC rows", len(table))
    return table


def network_to_iamc(cfg: Config, lines: pd.DataFrame, links: pd.DataFrame,
                    mapping: Dict) -> pd.DataFrame:
    """AC capacity + reactance and DC capacity, one region pair per row."""
    spec = mapping.get("network", {})
    sep = cfg.iamc.region_separator
    year = cfg.iamc.reference_year
    frames: List[pd.DataFrame] = []

    if not lines.empty:
        region = lines["region0"].astype(str) + sep + lines["region1"].astype(str)
        ac_cap = spec.get("ac_capacity", {})
        frames.append(pd.DataFrame({
            "region": region,
            "variable": ac_cap.get("variable", "Network|Electricity|Capacity"),
            "unit": ac_cap.get("unit", "MW"),
            "year": year,
            "value": lines["s_nom"],
        }))
        ac_x = spec.get("ac_reactance", {})
        frames.append(pd.DataFrame({
            "region": region,
            "variable": ac_x.get("variable", "Network|Electricity|Reactance"),
            "unit": ac_x.get("unit", "Ohm"),
            "year": year,
            "value": lines["x"],
        }))

    if not links.empty:
        dc_cap = spec.get("dc_capacity", {})
        frames.append(pd.DataFrame({
            "region": links["region0"].astype(str) + sep + links["region1"].astype(str),
            "variable": dc_cap.get("variable", "Network|Electricity|DC|Capacity"),
            "unit": dc_cap.get("unit", "MW"),
            "year": year,
            "value": links["p_nom"],
        }))

    if not frames:
        return pd.DataFrame(columns=IAMC_COLUMNS)

    table = _finalise(cfg, pd.concat(frames, ignore_index=True))
    log.info("Network -> %d IAMC rows", len(table))
    return table


def assemble(cfg: Config, blocks: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate the IAMC blocks into the single delivery table."""
    present = {name: frame for name, frame in blocks.items() if frame is not None and not frame.empty}
    if not present:
        raise ValueError("Nothing to assemble - every IAMC block is empty.")

    combined = pd.concat(present.values(), ignore_index=True)
    combined = combined[IAMC_COLUMNS]
    combined = combined.sort_values(["variable", "region", "year"]).reset_index(drop=True)
    log.info("Assembled IAMC table: %d rows from %s",
             len(combined), ", ".join(f"{k} ({len(v)})" for k, v in present.items()))
    return combined


def write_iamc(cfg: Config, table: pd.DataFrame, out_dir: Path) -> Dict[str, Path]:
    """Write the assembled table as xlsx and/or csv."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, Path] = {}
    stem = f"{cfg.run.name}_IAMC"

    if cfg.outputs.iamc_xlsx:
        path = out_dir / f"{stem}.xlsx"
        table.to_excel(path, index=False, sheet_name="data")
        written["xlsx"] = path
        log.info("Wrote %s (%d rows)", path, len(table))

    if cfg.outputs.iamc_csv:
        path = out_dir / f"{stem}.csv"
        table.to_csv(path, index=False)
        written["csv"] = path
        log.info("Wrote %s (%d rows)", path, len(table))

    return written


def _finalise(cfg: Config, frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the run metadata, drop empties and sum duplicate keys."""
    frame = frame.copy()
    frame["model"] = cfg.iamc.model
    frame["scenario"] = cfg.iamc.scenario
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value", "region"])
    frame = frame[frame["region"].astype(str).str.len() > 0]

    keys = ["model", "scenario", "region", "variable", "unit", "year"]
    before = len(frame)
    frame = frame.groupby(keys, as_index=False, dropna=False)["value"].sum()
    if len(frame) < before:
        log.info("  merged %d duplicate IAMC keys by summing", before - len(frame))
    return frame[IAMC_COLUMNS]
