"""Aggregation of nodal data to the chosen spatial level."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .logging_utils import get_logger

log = get_logger("aggregate")


@dataclass
class Aggregated:
    generation: pd.DataFrame     # region, fueltype, technology, capacity_mw, <year>...
    demand: pd.DataFrame         # region, country, load_share
    lines: pd.DataFrame          # region0, region1, s_nom, x, circuits
    links: pd.DataFrame          # region0, region1, p_nom
    regions: pd.DataFrame        # region, country, buses, x, y


def aggregate_all(cfg: Config, buses: pd.DataFrame, generation: pd.DataFrame,
                  demand: pd.DataFrame, lines: pd.DataFrame, links: pd.DataFrame,
                  years: List[int]) -> Aggregated:
    lookup = buses.set_index("bus_id")["region"].to_dict()
    # Country of a region, taken from the buses it contains rather than from
    # the first two characters of its id - the two disagree whenever a bus is
    # snapped across a border.
    region_country = (buses.dropna(subset=["region"])
                      .groupby("region")["country"]
                      .agg(lambda s: s.value_counts().idxmax())
                      .to_dict())

    return Aggregated(
        generation=aggregate_generation(generation, lookup, region_country, years),
        demand=aggregate_demand(demand, lookup),
        lines=aggregate_lines(lines, lookup),
        links=aggregate_links(links, lookup),
        regions=build_region_table(buses),
    )


# ---------------------------------------------------------------------------


def aggregate_generation(generation: pd.DataFrame, lookup: dict, region_country: dict,
                         years: List[int]) -> pd.DataFrame:
    gen = generation.copy()
    gen["region"] = gen["bus_id"].map(lookup)

    orphans = gen["region"].isna()
    if orphans.any():
        lost = gen.loc[orphans, "capacity_mw"].sum()
        log.warning("%d generation units (%.0f MW) have no region and are dropped",
                    int(orphans.sum()), lost)
        gen = gen[~orphans]

    value_cols = ["capacity_mw"] + [str(y) for y in years if str(y) in gen.columns]
    aggregated = (gen.groupby(["region", "fueltype", "technology"], dropna=False, as_index=False)[value_cols]
                  .sum())
    aggregated["country"] = aggregated["region"].map(
        lambda r: region_country.get(r, str(r)[:2])
    )
    log.info("Generation aggregated to %d region/fuel/technology rows (%.0f MW)",
             len(aggregated), aggregated["capacity_mw"].sum())
    return aggregated


def aggregate_demand(demand: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    dem = demand.copy()
    dem["region"] = dem["bus_id"].map(lookup)

    orphans = dem["region"].isna()
    if orphans.any():
        log.warning("%d demand rows (%.4f share) have no region and are dropped",
                    int(orphans.sum()), dem.loc[orphans, "load_share"].sum())
        dem = dem[~orphans]

    aggregated = (dem.groupby(["country", "region"], as_index=False)["load_share"].sum()
                  .sort_values(["country", "region"]).reset_index(drop=True))
    log.info("Demand aggregated to %d regions", len(aggregated))
    return aggregated


def aggregate_lines(lines: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """Aggregate AC lines onto region pairs.

    Capacities of parallel corridors add up; their reactances combine as
    parallel impedances (1/x_eq = sum 1/x_i), which is the electrically
    correct equivalent for a bundle of parallel circuits.
    """
    frame = lines.copy()
    frame["region0"] = frame["bus0"].map(lookup)
    frame["region1"] = frame["bus1"].map(lookup)
    frame = _drop_unmapped(frame, "AC lines")
    frame = frame[frame["region0"] != frame["region1"]]
    if frame.empty:
        log.warning("No AC lines remain after aggregation")
        return pd.DataFrame(columns=["region0", "region1", "s_nom", "x", "circuits", "length"])

    frame = _normalise_pairs(frame)
    frame["x"] = pd.to_numeric(frame.get("x"), errors="coerce")
    frame["s_nom"] = pd.to_numeric(frame.get("s_nom"), errors="coerce")
    frame["susceptance"] = np.where(frame["x"] > 0, 1.0 / frame["x"], np.nan)

    grouped = frame.groupby(["region0", "region1"], as_index=False).agg(
        s_nom=("s_nom", "sum"),
        susceptance=("susceptance", "sum"),
        circuits=("s_nom", "size"),
        length=("length", "sum") if "length" in frame.columns else ("s_nom", "size"),
    )
    grouped["x"] = np.where(grouped["susceptance"] > 0, 1.0 / grouped["susceptance"], np.nan)
    grouped = grouped.drop(columns=["susceptance"])
    log.info("AC lines aggregated to %d region pairs (%.0f MW total)",
             len(grouped), grouped["s_nom"].sum())
    return grouped


def aggregate_links(links: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    frame = links.copy()
    frame["region0"] = frame["bus0"].map(lookup)
    frame["region1"] = frame["bus1"].map(lookup)
    frame = _drop_unmapped(frame, "DC links")
    frame = frame[frame["region0"] != frame["region1"]]
    if frame.empty:
        log.warning("No DC links remain after aggregation")
        return pd.DataFrame(columns=["region0", "region1", "p_nom", "circuits"])

    frame = _normalise_pairs(frame)
    frame["p_nom"] = pd.to_numeric(frame.get("p_nom"), errors="coerce")
    grouped = frame.groupby(["region0", "region1"], as_index=False).agg(
        p_nom=("p_nom", "sum"),
        circuits=("p_nom", "size"),
    )
    log.info("DC links aggregated to %d region pairs (%.0f MW total)",
             len(grouped), grouped["p_nom"].sum())
    return grouped


def build_region_table(buses: pd.DataFrame) -> pd.DataFrame:
    """One row per region with a representative coordinate for mapping."""
    table = (buses.groupby("region", as_index=False)
             .agg(country=("country", "first"),
                  buses=("bus_id", "nunique"),
                  x=("x", "mean"),
                  y=("y", "mean"),
                  is_external=("is_neighbour", "all")))
    return table.sort_values("region").reset_index(drop=True)


# ---------------------------------------------------------------------------


def _drop_unmapped(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    unmapped = frame["region0"].isna() | frame["region1"].isna()
    if unmapped.any():
        log.warning("%d %s reference a bus with no region and are dropped",
                    int(unmapped.sum()), label)
    return frame[~unmapped].copy()


def _normalise_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    """Order each pair alphabetically so A-B and B-A aggregate together."""
    swap = frame["region0"].astype(str) > frame["region1"].astype(str)
    frame.loc[swap, ["region0", "region1"]] = frame.loc[swap, ["region1", "region0"]].values
    return frame
