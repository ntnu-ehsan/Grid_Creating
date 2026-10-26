"""Transmission grid: buses, AC lines and DC links."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from .config import Config
from .logging_utils import get_logger

log = get_logger("grid")

BUS_COLUMNS = ["bus_id", "voltage", "dc", "country", "x", "y"]


@dataclass
class Grid:
    buses: pd.DataFrame          # study-area buses plus the neighbour buses kept for interconnectors
    lines: pd.DataFrame          # AC
    links: pd.DataFrame          # DC

    @property
    def internal_buses(self) -> pd.DataFrame:
        return self.buses[~self.buses["is_neighbour"]]


def _read_csv(path, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(path, **kwargs)
    # PyPSA exports carry a long tail of empty "Unnamed: n" columns.
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def load_grid(cfg: Config) -> Grid:
    """Read the grid, restrict it to the study countries and keep the
    neighbouring buses that cross-border lines connect to."""
    buses_all = _read_csv(cfg.grid.buses)
    if cfg.grid.neighbouring_buses is not None:
        extra = _read_csv(cfg.grid.neighbouring_buses)
        log.info("Read %d neighbouring buses from a separate file", len(extra))
        buses_all = pd.concat([buses_all, extra], ignore_index=True)

    missing = [c for c in ("bus_id", "country", "x", "y") if c not in buses_all.columns]
    if missing:
        raise ValueError(f"Bus file {cfg.grid.buses} is missing columns: {missing}")

    buses_all["bus_id"] = buses_all["bus_id"].astype(str)
    keep_cols = [c for c in BUS_COLUMNS if c in buses_all.columns]
    buses_all = buses_all[keep_cols].drop_duplicates(subset="bus_id", keep="first")

    inside = buses_all[buses_all["country"].isin(cfg.countries)].copy()
    log.info("Buses in study area (%s): %d", ", ".join(cfg.countries), len(inside))

    if cfg.grid.format == "derived":
        lines, links = _load_derived_connections(cfg)
    else:
        lines = _read_csv(cfg.grid.lines)
        links = _read_csv(cfg.grid.links)

    for frame in (lines, links):
        frame["bus0"] = frame["bus0"].astype(str)
        frame["bus1"] = frame["bus1"].astype(str)

    inside_ids = set(inside["bus_id"])
    lines = _filter_connections(lines, inside_ids, cfg.grid.keep_cross_border)
    links = _filter_connections(links, inside_ids, cfg.grid.keep_cross_border)
    log.info("Kept %d AC lines and %d DC links", len(lines), len(links))

    neighbour_ids = (set(lines["bus0"]) | set(lines["bus1"])
                     | set(links["bus0"]) | set(links["bus1"])) - inside_ids
    neighbours = buses_all[buses_all["bus_id"].isin(neighbour_ids)].copy()
    if neighbour_ids - set(neighbours["bus_id"]):
        unknown = sorted(neighbour_ids - set(neighbours["bus_id"]))
        log.warning("%d connection endpoints are not present in the bus file (e.g. %s)",
                    len(unknown), ", ".join(unknown[:5]))

    inside["is_neighbour"] = False
    neighbours["is_neighbour"] = True
    buses = pd.concat([inside, neighbours], ignore_index=True)
    log.info("Added %d neighbouring buses for cross-border connections", len(neighbours))

    return Grid(buses=buses, lines=lines, links=links)


def _load_derived_connections(cfg: Config) -> tuple:
    """Split a single connection file into AC and DC and derive the ratings.

    Some sources (the Spanish grid among them) give the physical description
    of a corridor rather than its electrical parameters:

        AC capacity   S = sqrt(3) * V[kV] * Imax[kA] * circuits      -> MVA
        AC reactance  x = x_per_length[ohm/km] * length[km] / circuits
        DC capacity   P = V[kV] * Imax[kA] * circuits                -> MW
    """
    frame = _read_csv(cfg.grid.lines)

    # Some files store one row per *circuit* rather than per line: the circuits
    # of one corridor share a line_id stem and differ only by a trailing letter
    # (LTGES0001a / LTGES0001b), while the per-length parameters and the
    # `circuits` count already describe the whole corridor. Collapsing them
    # first stops the capacity being counted once per circuit.
    if cfg.grid.collapse_circuit_suffix and cfg.grid.line_id_column in frame.columns:
        before = len(frame)
        frame = frame.copy()
        frame[cfg.grid.line_id_column] = frame[cfg.grid.line_id_column].astype(str).str.replace(
            cfg.grid.collapse_circuit_suffix, "", regex=True)
        frame = frame.drop_duplicates(subset=cfg.grid.line_id_column).reset_index(drop=True)
        if len(frame) < before:
            log.info("Collapsed %d circuit rows into %d physical lines", before, len(frame))

    dc_col = cfg.grid.dc_column
    if dc_col not in frame.columns:
        raise ValueError(
            f"grid.format='derived' expects a '{dc_col}' column in {cfg.grid.lines} "
            f"to tell AC and DC apart. Columns present: {list(frame.columns)}"
        )

    for column in ("voltage", cfg.grid.current_column, "circuits"):
        if column not in frame.columns:
            raise ValueError(f"grid.format='derived' needs a '{column}' column in "
                             f"{cfg.grid.lines}")

    voltage = pd.to_numeric(frame["voltage"], errors="coerce")
    current = pd.to_numeric(frame[cfg.grid.current_column], errors="coerce")
    circuits = pd.to_numeric(frame["circuits"], errors="coerce").fillna(1).replace(0, 1)

    is_dc = frame[dc_col].astype(str).str.strip().str.lower().isin(cfg.grid.dc_true_values)

    lines = frame[~is_dc].copy()
    lines["s_nom"] = np.sqrt(3) * voltage[~is_dc] * current[~is_dc] * circuits[~is_dc]
    if cfg.grid.reactance_per_length_column in frame.columns and "length" in frame.columns:
        x_per_km = pd.to_numeric(frame[cfg.grid.reactance_per_length_column], errors="coerce")
        length = pd.to_numeric(frame["length"], errors="coerce")
        lines["x"] = x_per_km[~is_dc] * length[~is_dc] / circuits[~is_dc]
    else:
        log.warning("No reactance data available - AC reactance will be empty")
        lines["x"] = np.nan

    links = frame[is_dc].copy()
    links["p_nom"] = voltage[is_dc] * current[is_dc] * circuits[is_dc]

    log.info("Derived ratings for %d AC lines (%.0f MVA) and %d DC links (%.0f MW)",
             len(lines), lines["s_nom"].sum(), len(links), links["p_nom"].sum())
    return lines, links


def _filter_connections(frame: pd.DataFrame, inside_ids: set, keep_cross_border: bool) -> pd.DataFrame:
    in0 = frame["bus0"].isin(inside_ids)
    in1 = frame["bus1"].isin(inside_ids)
    mask = (in0 | in1) if keep_cross_border else (in0 & in1)
    return frame[mask].copy()


def summarise_grid(grid: Grid) -> pd.DataFrame:
    """One row per country: bus count, AC capacity, DC capacity."""
    rows: List[dict] = []
    buses = grid.internal_buses
    bus_country = grid.buses.set_index("bus_id")["country"].to_dict()

    for country, group in buses.groupby("country"):
        ids = set(group["bus_id"])
        ac = grid.lines[grid.lines["bus0"].isin(ids) | grid.lines["bus1"].isin(ids)]
        dc = grid.links[grid.links["bus0"].isin(ids) | grid.links["bus1"].isin(ids)]
        cross = dc[~(dc["bus0"].isin(ids) & dc["bus1"].isin(ids))]
        rows.append({
            "country": country,
            "buses": len(group),
            "ac_lines": len(ac),
            "ac_capacity_mw": float(ac["s_nom"].sum()) if "s_nom" in ac else float("nan"),
            "dc_links": len(dc),
            "dc_capacity_mw": float(dc["p_nom"].sum()) if "p_nom" in dc else float("nan"),
            "interconnectors": len(cross),
        })

    del bus_country
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)
