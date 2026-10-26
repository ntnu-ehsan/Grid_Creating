"""Resolution of the spatial level.

Every level ends up as a single ``region`` column on the bus table, so the
aggregation code downstream never needs to know whether it is working with
bus ids, country codes or NUTS ids.

  nodal     region = bus id
  national  region = ISO2 country code
  nuts2/3   region = NUTS id from a spatial join, with offshore buses snapped
            to their nearest resolved neighbour

Buses outside the study area (kept only because an interconnector lands on
them) get their country code plus the configured external suffix, e.g. DE_EXT.
"""

from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

from .config import Config
from .grid import Grid
from .logging_utils import get_logger
from .spatial import assign_nuts, load_nuts

log = get_logger("regions")


def resolve_regions(cfg: Config, grid: Grid) -> Tuple[pd.DataFrame, Optional["object"]]:
    """Add ``region`` to the bus table. Returns (buses, nuts geometry or None)."""
    level = cfg.spatial.level
    buses = grid.buses.copy()
    nuts_gdf = None

    if level == "nodal":
        buses["region"] = buses["bus_id"]
        buses["region_source"] = "bus"

    elif level == "national":
        buses["region"] = buses["country"]
        buses["region_source"] = "country"

    else:
        nuts_gdf = load_nuts(cfg.spatial.nuts_sources, cfg.spatial.nuts_level)
        buses = assign_nuts(buses, nuts_gdf,
                            snap_outside=cfg.spatial.snap_outside_buses,
                            fix_country_mismatch=cfg.spatial.fix_country_mismatch)

    # Buses outside the study area get one of two conventions, chosen in the
    # config: a country tag ("NO01 > DE_EXT"), which keeps foreign regions out
    # of the model, or their own region id ("ES113 > PT111"), which is more
    # informative when the neighbour's geography is meaningful.
    external = buses["is_neighbour"]
    if external.any():
        if cfg.grid.external_region == "region":
            # Fall back to the country tag where the neighbour has no region
            # (countries outside the NUTS geometry, or the nodal/national levels).
            unresolved = external & buses["region"].isna()
            buses.loc[unresolved, "region"] = (
                buses.loc[unresolved, "country"].fillna("XX") + cfg.grid.external_suffix
            )
            log.info("Kept the real region id for %d external buses (%d fell back to a "
                     "country tag)", int((external & ~unresolved).sum()), int(unresolved.sum()))
        else:
            buses.loc[external, "region"] = (
                buses.loc[external, "country"].fillna("XX") + cfg.grid.external_suffix
            )
        buses.loc[external, "region_source"] = "external"

    unassigned = buses["region"].isna()
    if unassigned.any():
        log.warning("%d buses could not be assigned to a region and will be dropped",
                    int(unassigned.sum()))
        buses = buses[~unassigned]

    n_regions = buses.loc[~buses["is_neighbour"], "region"].nunique()
    log.info("Spatial level '%s' resolved to %d regions inside the study area",
             level, n_regions)
    return buses, nuts_gdf


def region_country(region: pd.Series, external_suffix: str = "_EXT") -> pd.Series:
    """Country code implied by a region id (works for all levels)."""
    text = region.astype(str)
    return text.str.replace(external_suffix, "", regex=False).str[:2]


def region_lookup(buses: pd.DataFrame) -> dict:
    return buses.set_index("bus_id")["region"].to_dict()
