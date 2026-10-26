"""Spatial helpers: nearest-bus assignment and NUTS region lookup.

Both operations were repeated (and slowly re-derived) in every case-study
notebook. They live here once, vectorised, so that every country and every
spatial level goes through identical code.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .logging_utils import get_logger

log = get_logger("spatial")

EARTH_RADIUS_KM = 6371.0


def _to_cartesian(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Project lon/lat degrees onto a unit sphere in 3-D.

    Using chord distance on the sphere avoids the distortion a plain
    lat/lon Euclidean metric produces at Nordic latitudes, where one degree
    of longitude is roughly a third of one degree of latitude.
    """
    lon_r = np.radians(np.asarray(lon, dtype=float))
    lat_r = np.radians(np.asarray(lat, dtype=float))
    cos_lat = np.cos(lat_r)
    return np.column_stack((cos_lat * np.cos(lon_r), cos_lat * np.sin(lon_r), np.sin(lat_r)))


def nearest_bus(
    points: pd.DataFrame,
    buses: pd.DataFrame,
    lon_col: str,
    lat_col: str,
    bus_lon_col: str = "x",
    bus_lat_col: str = "y",
    bus_id_col: str = "bus_id",
    group_col: Optional[str] = None,
) -> Tuple[pd.Series, pd.Series]:
    """Assign every point to its closest bus.

    When ``group_col`` is given (typically the country code) points are only
    matched against buses of the same group, which stops a Danish demand
    pixel from being served by a Swedish bus across the strait.

    Returns a tuple of (assigned bus id, great-circle distance in km),
    both indexed like ``points``.
    """
    assigned = pd.Series(pd.NA, index=points.index, dtype="object")
    distance = pd.Series(np.nan, index=points.index, dtype="float")

    if group_col and group_col in points.columns and group_col in buses.columns:
        groups = sorted(set(points[group_col].dropna().unique()))
    else:
        groups = [None]

    for group in groups:
        if group is None:
            pts, bss = points, buses
        else:
            pts = points[points[group_col] == group]
            bss = buses[buses[group_col] == group]
            if bss.empty:
                log.warning("No buses available for group '%s' - %d points left unassigned",
                            group, len(pts))
                continue
        if pts.empty:
            continue

        tree = cKDTree(_to_cartesian(bss[bus_lon_col].values, bss[bus_lat_col].values))
        chord, idx = tree.query(_to_cartesian(pts[lon_col].values, pts[lat_col].values), k=1)

        assigned.loc[pts.index] = bss[bus_id_col].values[idx]
        # chord length on the unit sphere -> great-circle distance in km
        distance.loc[pts.index] = 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))

    return assigned, distance


def nearest_neighbour_within(
    targets: pd.DataFrame,
    reference: pd.DataFrame,
    value_col: str,
    lon_col: str = "x",
    lat_col: str = "y",
) -> pd.Series:
    """Copy ``value_col`` from the geographically closest reference row.

    Used to give offshore or coastline buses the region of the nearest bus
    that a spatial join did resolve.
    """
    if targets.empty or reference.empty:
        return pd.Series(pd.NA, index=targets.index, dtype="object")

    tree = cKDTree(_to_cartesian(reference[lon_col].values, reference[lat_col].values))
    _, idx = tree.query(_to_cartesian(targets[lon_col].values, targets[lat_col].values), k=1)
    return pd.Series(reference[value_col].values[idx], index=targets.index)


def load_nuts(sources, level: int):
    """Read NUTS geometry for one level, merging several releases if needed.

    ``sources`` is a list of config ``NutsSource`` entries. Earlier entries win
    on duplicate region ids, so a case can take most regions from the current
    release and pull the United Kingdom - dropped after 2021 - from an older
    one.
    """
    import geopandas as gpd
    import pandas as pd

    if not isinstance(sources, (list, tuple)):
        sources = [sources]

    frames = []
    for source in sources:
        path = getattr(source, "path", source)
        countries = getattr(source, "countries", None)

        gdf = gpd.read_file(path)
        if "LEVL_CODE" not in gdf.columns:
            raise ValueError(f"{path} has no LEVL_CODE column - is it a NUTS file?")
        gdf = gdf[gdf["LEVL_CODE"] == level].copy()

        if countries:
            wanted = {c.upper() for c in countries}
            # GB and UK name the same place in different datasets.
            wanted |= {"UK"} if "GB" in wanted else set()
            wanted |= {"GB"} if "UK" in wanted else set()
            gdf = gdf[gdf["CNTR_CODE"].astype(str).str.upper().isin(wanted)]

        if gdf.empty:
            log.warning("No NUTS level %d regions matched in %s", level, path)
            continue

        log.info("Loaded %d NUTS%d regions from %s%s", len(gdf), level, path,
                 f" (countries: {', '.join(sorted(countries))})" if countries else "")
        frames.append(gdf)

    if not frames:
        raise ValueError(f"No NUTS level {level} geometries found in any configured source")

    merged = frames[0] if len(frames) == 1 else gpd.GeoDataFrame(
        pd.concat([f.to_crs(frames[0].crs) for f in frames], ignore_index=True),
        crs=frames[0].crs,
    )
    merged = merged.drop_duplicates(subset=["NUTS_ID"], keep="first")
    if len(frames) > 1:
        log.info("Merged NUTS sources into %d regions", len(merged))
    return merged


def _region_centroids(nuts_gdf) -> pd.DataFrame:
    """NUTS ids with a representative lon/lat, for nearest-region lookups."""
    wgs = nuts_gdf.to_crs(epsg=4326)
    points = wgs.geometry.representative_point()
    return pd.DataFrame({
        "region": wgs["NUTS_ID"].values,
        "country": wgs["CNTR_CODE"].astype(str).str.upper().replace({"UK": "GB"}).values,
        "x": points.x.values,
        "y": points.y.values,
    })


def assign_nuts(buses: pd.DataFrame, nuts_gdf, snap_outside: bool = True,
                fix_country_mismatch: bool = True) -> pd.DataFrame:
    """Spatially join buses to NUTS polygons.

    Adds ``region`` (the NUTS id) and ``region_source``, which records how the
    assignment was made: "within" for a direct hit, "snapped" for offshore
    buses placed by their nearest neighbour, "country_corrected" for buses that
    landed in a neighbouring country's polygon, "unassigned" for the rest.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    frame = buses.copy()
    geometry = [Point(xy) for xy in zip(frame["x"], frame["y"])]
    gdf = gpd.GeoDataFrame(frame, geometry=geometry, crs="EPSG:4326").to_crs(nuts_gdf.crs)

    joined = gpd.sjoin(gdf, nuts_gdf[["NUTS_ID", "CNTR_CODE", "geometry"]],
                       how="left", predicate="within")
    # A bus sitting exactly on a shared border can match two polygons.
    joined = joined[~joined.index.duplicated(keep="first")]

    frame["region"] = joined["NUTS_ID"].values
    frame["region_source"] = np.where(frame["region"].notna(), "within", "unassigned")

    # Buses outside the study area are labelled by country later on, so they
    # are left out of the snapping and correction steps.
    in_study = ~frame["is_neighbour"] if "is_neighbour" in frame.columns else pd.Series(
        True, index=frame.index)

    n_outside = int((frame["region"].isna() & in_study).sum())
    if n_outside and snap_outside:
        resolved = frame[frame["region"].notna()]
        outside = frame[frame["region"].isna() & in_study]

        # Snap within the bus's own country. Without this, an offshore Danish
        # bus can inherit a German region simply because a German bus happens
        # to be the closest one with a resolved region.
        snapped = pd.Series(pd.NA, index=outside.index, dtype="object")
        for country, group in outside.groupby("country", dropna=False):
            candidates = resolved[resolved["country"] == country]
            if candidates.empty:
                log.warning("No resolved bus in %s to snap %d offshore bus(es) to",
                            country, len(group))
                continue
            snapped.loc[group.index] = nearest_neighbour_within(group, candidates, "region")

        frame.loc[outside.index, "region"] = snapped
        frame.loc[outside.index, "region_source"] = np.where(
            snapped.notna(), "snapped", "unassigned"
        )
        log.info("Snapped %d buses outside every NUTS polygon to the nearest bus "
                 "in the same country", int(snapped.notna().sum()))
    elif n_outside:
        log.warning("%d buses fall outside every NUTS polygon and stay unassigned", n_outside)

    if fix_country_mismatch:
        frame = _fix_country_mismatch(frame, nuts_gdf, in_study)

    return frame


def _fix_country_mismatch(frame: pd.DataFrame, nuts_gdf, in_study: pd.Series) -> pd.DataFrame:
    """Move buses that landed in a neighbouring country's region.

    A bus a few kilometres off - common along the Denmark/Germany,
    Sweden/Norway and Spain/Portugal borders, and amplified by the 1:60M
    generalisation of the polygons - otherwise exports its capacity into a
    region the model does not own.

    The bus is moved to the **nearest polygon of its own country**, measured to
    the region boundary rather than to a centroid or to another bus: a border
    bus is by definition adjacent to the right region, even when the nearest
    modelled bus of that country is far inland.
    """
    import geopandas as gpd
    from shapely.geometry import Point

    country_of_region = (nuts_gdf.set_index("NUTS_ID")["CNTR_CODE"]
                         .astype(str).str.upper().replace({"UK": "GB"}).to_dict())

    bus_country = frame["country"].astype(str).str.upper().replace({"UK": "GB"})
    assigned_country = frame["region"].map(country_of_region)
    mismatch = (frame["region"].notna() & assigned_country.notna()
                & (assigned_country != bus_country) & in_study)

    if not mismatch.any():
        return frame

    regions = nuts_gdf[["NUTS_ID", "CNTR_CODE", "geometry"]].copy()
    regions["CNTR_CODE"] = regions["CNTR_CODE"].astype(str).str.upper().replace({"UK": "GB"})

    for country, group in frame[mismatch].groupby("country"):
        code = str(country).upper().replace("UK", "GB")
        candidates = regions[regions["CNTR_CODE"] == code]
        if candidates.empty:
            log.warning("%d %s bus(es) sit in a foreign region and no %s region exists "
                        "to move them to", len(group), country, country)
            continue

        points = gpd.GeoDataFrame(
            group[["bus_id"]].copy(),
            geometry=[Point(xy) for xy in zip(group["x"], group["y"])],
            crs="EPSG:4326",
        ).to_crs(candidates.crs)

        nearest = gpd.sjoin_nearest(points, candidates, how="left", distance_col="_distance")
        nearest = nearest[~nearest.index.duplicated(keep="first")]

        frame.loc[group.index, "region"] = nearest["NUTS_ID"].values
        frame.loc[group.index, "region_source"] = "country_corrected"
        frame.loc[group.index, "region_fix_km"] = nearest["_distance"].values / 1000.0
        log.info("Moved %d %s bus(es) out of a neighbouring country's region, to the "
                 "nearest %s region (e.g. %s -> %s, %.1f km)", len(group), country, country,
                 group["region"].iloc[0], nearest["NUTS_ID"].iloc[0],
                 nearest["_distance"].iloc[0] / 1000.0)

    return frame
