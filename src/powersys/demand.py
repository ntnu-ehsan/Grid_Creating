"""Demand distribution.

The Atlas raster product does not cover Norway or the UK, so a single method
cannot serve every country. Each country is therefore routed to one of four
methods through ``demand.method`` in the config:

  atlas               1 km electricity demand pixels, assigned to the nearest bus
  population          1 km population as a proxy for demand
  population+industry population proxy plus explicit industrial and offshore
                      oil sites, weighted by their share of national load
  precomputed         a ready-made bus-level share table

Whatever the method, the result has the same shape - one row per bus with a
``load_share`` that sums to 1.0 within each country.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import Config
from .grid import Grid
from .logging_utils import get_logger
from .spatial import nearest_bus

log = get_logger("demand")


def build_demand(cfg: Config, grid: Grid) -> pd.DataFrame:
    """Bus-level load shares for every study country."""
    frames: List[pd.DataFrame] = []
    precomputed_cache: pd.DataFrame | None = None

    for country in cfg.countries:
        method = cfg.demand.method_for(country)
        buses = grid.internal_buses[grid.internal_buses["country"] == country]
        if buses.empty:
            log.warning("No buses for %s - skipping its demand", country)
            continue

        log.info("Demand for %s using method '%s'", country, method)
        if method == "atlas":
            shares = _from_pixels(cfg, country, buses, cfg.demand.atlas,
                                  path_key="country", source="atlas")
        elif method == "population":
            shares = _from_pixels(cfg, country, buses, cfg.demand.population,
                                  path_key="code", source="population")
        elif method == "population+industry":
            shares = _population_plus_industry(cfg, country, buses)
        elif method == "precomputed":
            if precomputed_cache is None:
                precomputed_cache = _read_precomputed(cfg)
            shares = precomputed_cache[precomputed_cache["bus_id"].isin(set(buses["bus_id"]))].copy()
            if shares.empty:
                raise ValueError(
                    f"No precomputed demand rows matched the {country} buses. "
                    "Check demand.precomputed.files."
                )
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"Unknown demand method '{method}'")

        shares["country"] = country
        shares["method"] = method
        frames.append(_normalise_shares(shares, country))

    demand = pd.concat(frames, ignore_index=True)
    demand = demand[["country", "bus_id", "load_share", "method"]]
    log.info("Demand assigned to %d buses across %d countries",
             len(demand), demand["country"].nunique())
    return demand


# ---------------------------------------------------------------------------
# methods
# ---------------------------------------------------------------------------


def _from_pixels(cfg: Config, country: str, buses: pd.DataFrame,
                 settings: Dict, path_key: str, source: str) -> pd.DataFrame:
    """Assign gridded pixels (demand or population) to the nearest bus."""
    template = settings.get("path_template")
    if not template:
        raise ValueError(f"demand.{source}.path_template is not set in the config.")

    token = cfg.country_name(country) if path_key == "country" else country
    path = Path(template.format(**{path_key: token, "country": cfg.country_name(country),
                                   "code": country}))
    if not path.exists():
        raise FileNotFoundError(
            f"{source} input for {country} not found: {path}\n"
            f"(the Atlas product has no coverage for Norway or the UK - use the "
            f"'population' or 'population+industry' method for those countries)"
        )

    cached = _cache_path(cfg, country, source, path)
    if cached is not None and cfg.demand.reuse_cache and cached.exists():
        log.info("  reusing cached pixel assignment %s", cached.name)
        return pd.read_csv(cached, dtype={"bus_id": str})

    lon_col = settings.get("lon_column", "X")
    lat_col = settings.get("lat_column", "Y")
    value_col = settings.get("value_column", "Z")

    pixels = pd.read_csv(path, usecols=lambda c: c in (lon_col, lat_col, value_col))
    pixels = pixels.rename(columns={lon_col: "lon", lat_col: "lat", value_col: "value"})
    pixels["value"] = pd.to_numeric(pixels["value"], errors="coerce")
    pixels = pixels[pixels["value"] > 0].dropna(subset=["lon", "lat", "value"])
    log.info("  %s pixels: %d", source, len(pixels))

    assigned, _ = nearest_bus(pixels, buses, lon_col="lon", lat_col="lat")
    pixels["bus_id"] = assigned
    shares = (pixels.groupby("bus_id", dropna=True)["value"].sum()
              .rename("load_share").reset_index())

    if cached is not None:
        cached.parent.mkdir(parents=True, exist_ok=True)
        shares.to_csv(cached, index=False)
        log.info("  cached pixel assignment -> %s", cached)
    return shares


def _population_plus_industry(cfg: Config, country: str, buses: pd.DataFrame) -> pd.DataFrame:
    """Population proxy with explicit industrial and offshore sites on top.

    The site consumption is known in MW, but the population layer is only a
    share. The two are combined through the sites' share of national load:
    industry and offshore keep their own spatial pattern, and the remaining
    share is spread by population.
    """
    settings = (cfg.demand.industry or {}).get(country)
    if not settings:
        raise ValueError(f"demand.industry.{country} block is missing.")

    population = _from_pixels(cfg, country, buses, cfg.demand.population,
                              path_key="code", source="population")
    population["load_share"] = population["load_share"] / population["load_share"].sum()

    consumption_col = settings.get("consumption_column", "consumption_mw")
    total_col = settings.get("total_column", "mean_load_mw")

    national_total = pd.read_csv(settings["national_total"])[total_col].sum()
    if not national_total > 0:
        raise ValueError(f"National load total for {country} is zero or negative.")

    site_frames = []
    for key, label in (("sites", "industry"), ("offshore", "offshore")):
        path = settings.get(key)
        if not path:
            continue
        sites = pd.read_csv(path)
        sites = sites.rename(columns={"lon": "lon", "lat": "lat"})
        sites["consumption"] = pd.to_numeric(sites[consumption_col], errors="coerce")
        sites = sites.dropna(subset=["lon", "lat", "consumption"])
        sites["category"] = label
        site_frames.append(sites)
        log.info("  %s sites: %d (%.0f MW)", label, len(sites), sites["consumption"].sum())

    if not site_frames:
        log.warning("  no industrial sites configured for %s - falling back to population only",
                    country)
        return population

    sites = pd.concat(site_frames, ignore_index=True)
    assigned, _ = nearest_bus(sites, buses, lon_col="lon", lat_col="lat")
    sites["bus_id"] = assigned

    site_share = sites["consumption"].sum() / national_total
    if site_share >= 1:
        raise ValueError(
            f"Industrial + offshore load for {country} ({sites['consumption'].sum():.0f} MW) "
            f"exceeds the national total ({national_total:.0f} MW)."
        )
    log.info("  sites cover %.1f%% of national load, population spreads the remaining %.1f%%",
             site_share * 100, (1 - site_share) * 100)

    population["load_share"] *= (1 - site_share)
    site_shares = (sites.groupby("bus_id")["consumption"].sum() / national_total
                   ).rename("load_share").reset_index()

    combined = (pd.concat([population, site_shares], ignore_index=True)
                .groupby("bus_id", as_index=False)["load_share"].sum())
    return combined


def _read_precomputed(cfg: Config) -> pd.DataFrame:
    files = (cfg.demand.precomputed or {}).get("files", [])
    if not files:
        raise ValueError("demand.precomputed.files is empty but a country asks for it.")

    frames = []
    for entry in files:
        path = Path(entry["path"])
        if not path.exists():
            log.warning("Precomputed demand file not found, skipping: %s", path)
            continue
        frame = pd.read_csv(path)
        frame = frame.rename(columns={entry.get("bus_column", "bus_id"): "bus_id",
                                      entry.get("share_column", "load_share"): "load_share"})
        frame["bus_id"] = frame["bus_id"].astype(str)
        frame["load_share"] = pd.to_numeric(frame["load_share"], errors="coerce").fillna(0.0)
        frames.append(frame[["bus_id", "load_share"]])
        log.info("  read %d precomputed bus shares from %s", len(frame), path.name)

    if not frames:
        raise FileNotFoundError("None of the demand.precomputed.files exist.")
    combined = pd.concat(frames, ignore_index=True)
    # Later files win over earlier ones for the same bus.
    return combined.drop_duplicates(subset="bus_id", keep="last")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _normalise_shares(shares: pd.DataFrame, country: str) -> pd.DataFrame:
    total = shares["load_share"].sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Demand shares for {country} sum to {total}; cannot normalise.")
    if abs(total - 1.0) > 1e-6:
        log.info("  normalising %s shares (raw sum %.4f -> 1.0)", country, total)
    shares = shares.copy()
    shares["load_share"] = shares["load_share"] / total
    return shares


def _cache_path(cfg: Config, country: str, source: str, input_path: Path) -> Path | None:
    if not cfg.demand.cache_dir:
        return None
    try:
        stamp = f"{input_path}:{input_path.stat().st_mtime_ns}"
    except OSError:
        return None
    digest = hashlib.sha1(stamp.encode("utf-8")).hexdigest()[:10]
    return Path(cfg.demand.cache_dir) / f"{source}_{country}_{digest}.csv"


def summarise_demand(demand: pd.DataFrame) -> pd.DataFrame:
    """One row per country: method used, buses served, share total."""
    return (demand.groupby(["country", "method"], as_index=False)
            .agg(buses=("bus_id", "nunique"),
                 share_sum=("load_share", "sum"),
                 max_bus_share=("load_share", "max"))
            .sort_values("country").reset_index(drop=True))
