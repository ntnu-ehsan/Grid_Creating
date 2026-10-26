"""Generation units: loading, label normalisation, bus assignment and the
capacity-per-year expansion used by the time-series pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

from .config import Config
from .grid import Grid
from .logging_utils import get_logger
from .spatial import nearest_bus

log = get_logger("generation")

# Full country names appearing in the PyPSA powerplant table.
_NAME_TO_ISO = {
    "sweden": "SE", "denmark": "DK", "finland": "FI", "norway": "NO",
    "germany": "DE", "netherlands": "NL", "belgium": "BE", "france": "FR",
    "united kingdom": "GB", "great britain": "GB", "spain": "ES",
    "portugal": "PT", "italy": "IT", "ireland": "IE", "poland": "PL",
    "estonia": "EE", "latvia": "LV", "lithuania": "LT", "switzerland": "CH",
    "austria": "AT", "czechia": "CZ", "czech republic": "CZ",
}


def load_labels(mapping_dir: Path) -> Dict[str, Dict[str, str]]:
    with open(mapping_dir / "technology_labels.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _normalise(series: pd.Series, lookup: Dict[str, str]) -> pd.Series:
    key = series.fillna("").astype(str).str.strip().str.lower()
    lookup = {str(k).strip().lower(): v for k, v in (lookup or {}).items()}
    mapped = key.map(lookup)
    # Anything unknown keeps its original spelling, title-cased, so that it
    # shows up clearly in the unmapped-variable check later on.
    fallback = series.fillna("").astype(str).str.strip().str.title()
    return mapped.fillna(fallback)


def load_generation(cfg: Config, grid: Grid, mapping_dir: Path) -> pd.DataFrame:
    """Read generation units, restrict to the study countries, normalise the
    fuel/technology labels and attach each unit to a bus."""
    raw = pd.read_csv(cfg.generation.file)
    raw.columns = [c.strip() for c in raw.columns]

    country_col = _pick(raw, ["Country", "country"])
    capacity_col = _pick(raw, ["Capacity", "capacity", "capacity_mw", "p_nom"])
    fuel_col = _pick(raw, ["Fueltype", "fueltype", "source", "Source",
                           "primary_fuel", "PrimaryFuel"])
    tech_col = _pick(raw, ["Technology", "technology"])
    lat_col = _pick(raw, ["lat", "Lat", "y", "latitude"])
    lon_col = _pick(raw, ["lon", "Lon", "x", "longitude"])

    if country_col is not None:
        country = _to_iso(raw[country_col])
    elif cfg.generation.default_country:
        country = pd.Series(cfg.generation.default_country, index=raw.index)
        log.info("Generation file has no country column - using generation.default_country=%s",
                 cfg.generation.default_country)
    else:
        raise ValueError(
            f"{cfg.generation.file} has no country column. Set generation.default_country "
            f"in the config for single-country cases."
        )

    gen = pd.DataFrame({
        "unit_id": raw[_pick(raw, ["id", "unit_id", "Name", "name"])].astype(str),
        "name": raw[_pick(raw, ["Name", "name", "id", "unit_id"])].astype(str),
        "country": country,
        "capacity_mw": pd.to_numeric(raw[capacity_col], errors="coerce"),
        "fueltype": raw[fuel_col] if fuel_col else "",
        "technology": raw[tech_col] if tech_col else "",
        "lat": pd.to_numeric(raw[lat_col], errors="coerce") if lat_col else np.nan,
        "lon": pd.to_numeric(raw[lon_col], errors="coerce") if lon_col else np.nan,
        "date_in": pd.to_numeric(raw.get("DateIn"), errors="coerce"),
        "date_out": pd.to_numeric(raw.get("DateOut"), errors="coerce"),
    })
    if cfg.generation.format == "nodal" and cfg.generation.bus_column in raw.columns:
        gen["bus_id"] = raw[cfg.generation.bus_column].astype(str)

    before = len(gen)
    gen = gen[gen["country"].isin(cfg.countries)].copy()
    log.info("Generation units in study area: %d (of %d in the source file)", len(gen), before)

    gen = gen[gen["capacity_mw"].fillna(0) > 0]

    if cfg.generation.retire_before is not None:
        retired = gen["date_out"].lt(cfg.generation.retire_before)
        if retired.any():
            log.info("Dropping %d units decommissioned before %s",
                     int(retired.sum()), cfg.generation.retire_before)
            gen = gen[~retired]

    labels = load_labels(mapping_dir)
    gen["fueltype"] = _normalise(gen["fueltype"], labels.get("fueltype", {}))
    gen["technology"] = _normalise(gen["technology"], labels.get("technology", {}))

    gen = _attach_buses(cfg, gen, grid)

    log.info("Total installed capacity: %.0f MW across %d units",
             gen["capacity_mw"].sum(), len(gen))
    return gen.reset_index(drop=True)


def _attach_buses(cfg: Config, gen: pd.DataFrame, grid: Grid) -> pd.DataFrame:
    if cfg.generation.assign_to_bus == "column":
        if "bus_id" not in gen.columns:
            raise ValueError(
                "generation.assign_to_bus='column' but the source file has no "
                f"'{cfg.generation.bus_column}' column."
            )
        known = set(grid.buses["bus_id"])
        unknown = ~gen["bus_id"].isin(known)
        if unknown.any():
            log.warning("%d units reference a bus id that does not exist; "
                        "falling back to nearest-bus for those", int(unknown.sum()))
            fixed, dist = nearest_bus(gen[unknown], grid.internal_buses,
                                      lon_col="lon", lat_col="lat", group_col="country")
            gen.loc[unknown, "bus_id"] = fixed
            gen.loc[unknown, "bus_distance_km"] = dist
        return gen

    missing_coords = gen["lat"].isna() | gen["lon"].isna()
    if missing_coords.any():
        log.warning("%d units have no coordinates and cannot be assigned to a bus",
                    int(missing_coords.sum()))

    assigned, distance = nearest_bus(
        gen[~missing_coords], grid.internal_buses,
        lon_col="lon", lat_col="lat", group_col="country",
    )
    gen["bus_id"] = pd.NA
    gen["bus_distance_km"] = np.nan
    gen.loc[assigned.index, "bus_id"] = assigned
    gen.loc[distance.index, "bus_distance_km"] = distance

    far = gen["bus_distance_km"] > 100
    if far.any():
        log.warning("%d units are further than 100 km from their assigned bus "
                    "(max %.0f km)", int(far.sum()), gen["bus_distance_km"].max())
    return gen


def expand_capacity_years(gen: pd.DataFrame, years: List[int]) -> pd.DataFrame:
    """Add one column per study year holding the capacity online in that year.

    A unit contributes to year Y when it was commissioned no later than Y and
    is not yet decommissioned. Missing dates are treated as "always on",
    matching how the source datasets record long-standing plant.
    """
    out = gen.copy()
    date_in = out["date_in"]
    date_out = out["date_out"]
    for year in years:
        online = (date_in.isna() | (date_in <= year)) & (date_out.isna() | (date_out > year))
        out[str(year)] = out["capacity_mw"].where(online, 0.0)
    return out


def _pick(frame: pd.DataFrame, candidates) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _to_iso(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    iso = text.str.lower().map(_NAME_TO_ISO)
    return iso.fillna(text.str.upper().where(text.str.len() == 2, text))


def summarise_generation(gen: pd.DataFrame) -> pd.DataFrame:
    """Installed capacity by country and fuel, for the report."""
    table = (gen.groupby(["country", "fueltype"], dropna=False)["capacity_mw"]
             .sum().reset_index())
    return table.sort_values(["country", "capacity_mw"], ascending=[True, False])
