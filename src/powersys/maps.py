"""Interactive Folium map of the assembled dataset.

Layers (each can be switched off in the map's own control):
  regions      NUTS polygons shaded by their share of national demand
  generation   one circle per region, sized by installed capacity
  ac lines     aggregated AC corridors, width by capacity
  dc links     aggregated DC links, width by capacity
  buses        the underlying transmission nodes
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import Config
from .logging_utils import get_logger

log = get_logger("maps")

# Sequential single-hue ramp for demand magnitude.
DEMAND_RAMP = ["#eaf2fc", "#bcd8f7", "#7db3ee", "#4a92e2", "#2a78d6", "#1b5192", "#123a68"]
AC_COLOUR = "#2a78d6"
DC_COLOUR = "#eb6834"
BUS_COLOUR = "#52514e"
GEN_COLOUR = "#1baf7a"


def _ramp_colour(value: float, vmax: float) -> str:
    if not np.isfinite(value) or vmax <= 0:
        return DEMAND_RAMP[0]
    idx = int(round((value / vmax) * (len(DEMAND_RAMP) - 1)))
    return DEMAND_RAMP[min(max(idx, 0), len(DEMAND_RAMP) - 1)]


def build_map(cfg: Config, aggregated, buses: pd.DataFrame, nuts_gdf,
              out_path: Path) -> Optional[Path]:
    """Write the interactive map and return its path."""
    try:
        import folium
    except ImportError:  # pragma: no cover
        log.warning("folium is not installed - skipping the map")
        return None

    regions = aggregated.regions
    internal = regions[~regions["is_external"]]
    if internal.empty:
        log.warning("No internal regions to map")
        return None

    centre = [float(internal["y"].mean()), float(internal["x"].mean())]
    fmap = folium.Map(location=centre, zoom_start=4, tiles="cartodbpositron",
                      control_scale=True)

    demand = aggregated.demand.set_index("region")["load_share"].to_dict()
    capacity = (aggregated.generation.groupby("region")["capacity_mw"].sum().to_dict())
    fuel_mix = _fuel_mix(aggregated.generation)
    coords = _region_coords(regions, nuts_gdf, cfg)

    if nuts_gdf is not None:
        _add_region_polygons(folium, fmap, nuts_gdf, internal, demand, capacity)

    _add_generation(folium, fmap, internal, coords, capacity, fuel_mix)
    _add_connections(folium, fmap, aggregated, coords, cfg)
    _add_buses(folium, fmap, buses)
    _add_legend(fmap, demand)

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(out_path))
    log.info("Wrote interactive map -> %s", out_path)
    return out_path


# ---------------------------------------------------------------------------


def _region_coords(regions: pd.DataFrame, nuts_gdf, cfg: Config) -> Dict[str, tuple]:
    """Representative point per region: NUTS centroid where available,
    otherwise the mean position of the region's buses."""
    coords = {row.region: (row.y, row.x) for row in regions.itertuples()}

    if nuts_gdf is not None:
        try:
            centroids = nuts_gdf.to_crs(epsg=4326)
            centroids = centroids.set_index("NUTS_ID").geometry.representative_point()
            for region_id, point in centroids.items():
                if region_id in coords:
                    coords[region_id] = (point.y, point.x)
        except Exception as exc:  # pragma: no cover
            log.warning("Could not derive NUTS centroids (%s) - using bus positions", exc)
    return coords


def _fuel_mix(generation: pd.DataFrame) -> Dict[str, str]:
    """Small HTML table of the fuel breakdown, used in the popups."""
    mixes: Dict[str, str] = {}
    for region, group in generation.groupby("region"):
        rows = (group.groupby("fueltype")["capacity_mw"].sum()
                .sort_values(ascending=False).head(8))
        body = "".join(
            f"<tr><td style='padding:1px 8px 1px 0'>{fuel}</td>"
            f"<td style='text-align:right'>{value:,.0f}</td></tr>"
            for fuel, value in rows.items()
        )
        mixes[region] = (
            "<table style='font-size:12px;border-collapse:collapse'>"
            f"{body}</table>"
        )
    return mixes


def _add_region_polygons(folium, fmap, nuts_gdf, regions, demand, capacity) -> None:
    known = set(regions["region"])
    subset = nuts_gdf[nuts_gdf["NUTS_ID"].isin(known)].to_crs(epsg=4326)
    if subset.empty:
        return

    vmax = max(demand.values()) if demand else 0.0
    layer = folium.FeatureGroup(name="Regions - demand share", show=True)

    for row in subset.itertuples():
        region_id = row.NUTS_ID
        share = demand.get(region_id, 0.0)
        tooltip = (f"<b>{region_id}</b><br>"
                   f"Demand share: {share * 100:.2f}%<br>"
                   f"Capacity: {capacity.get(region_id, 0):,.0f} MW")
        folium.GeoJson(
            row.geometry.__geo_interface__,
            style_function=lambda _f, colour=_ramp_colour(share, vmax): {
                "fillColor": colour, "color": "#ffffff",
                "weight": 1.0, "fillOpacity": 0.82,
            },
            highlight_function=lambda _f: {"weight": 2.5, "color": "#0b0b0b"},
            tooltip=folium.Tooltip(tooltip),
        ).add_to(layer)

    layer.add_to(fmap)


def _add_generation(folium, fmap, regions, coords, capacity, fuel_mix) -> None:
    layer = folium.FeatureGroup(name="Generation capacity", show=True)
    values = [v for v in capacity.values() if np.isfinite(v) and v > 0]
    if not values:
        layer.add_to(fmap)
        return
    vmax = max(values)

    for row in regions.itertuples():
        total = capacity.get(row.region, 0.0)
        if total <= 0 or row.region not in coords:
            continue
        # Area-proportional radius so a circle twice the area reads as twice the MW.
        radius = 4 + 22 * np.sqrt(total / vmax)
        popup = (f"<div style='font-family:system-ui;font-size:13px'>"
                 f"<b>{row.region}</b><br>"
                 f"Installed capacity: <b>{total:,.0f} MW</b><br>"
                 f"{fuel_mix.get(row.region, '')}</div>")
        folium.CircleMarker(
            location=coords[row.region], radius=radius,
            color="#ffffff", weight=1.5, fill=True,
            fill_color=GEN_COLOUR, fill_opacity=0.75,
            tooltip=f"{row.region}: {total:,.0f} MW",
            popup=folium.Popup(popup, max_width=320),
        ).add_to(layer)

    layer.add_to(fmap)


def _add_connections(folium, fmap, aggregated, coords, cfg: Config) -> None:
    for frame, name, colour, value_col, unit in (
        (aggregated.lines, "AC corridors", AC_COLOUR, "s_nom", "MW"),
        (aggregated.links, "DC links", DC_COLOUR, "p_nom", "MW"),
    ):
        if frame is None or frame.empty:
            continue
        layer = folium.FeatureGroup(name=name, show=True)
        vmax = frame[value_col].max() or 1.0

        for row in frame.itertuples():
            start, end = coords.get(row.region0), coords.get(row.region1)
            if start is None or end is None:
                continue
            value = getattr(row, value_col)
            weight = 1.2 + 5.0 * float(value) / float(vmax)
            tooltip = (f"<b>{row.region0} {cfg.iamc.region_separator} {row.region1}</b><br>"
                       f"Capacity: {value:,.0f} {unit}")
            if "x" in frame.columns and hasattr(row, "x") and np.isfinite(getattr(row, "x", np.nan)):
                tooltip += f"<br>Reactance: {row.x:.3f} Ohm"
            folium.PolyLine([start, end], color=colour, weight=weight,
                            opacity=0.85, tooltip=folium.Tooltip(tooltip)).add_to(layer)

        layer.add_to(fmap)


def _add_buses(folium, fmap, buses: pd.DataFrame) -> None:
    layer = folium.FeatureGroup(name="Transmission buses", show=False)
    for row in buses.itertuples():
        if not np.isfinite(row.x) or not np.isfinite(row.y):
            continue
        external = bool(getattr(row, "is_neighbour", False))
        folium.CircleMarker(
            location=[row.y, row.x], radius=2.5,
            color=BUS_COLOUR if not external else "#a9a89f",
            weight=1, fill=True, fill_opacity=0.9,
            tooltip=(f"{row.bus_id} ({row.country})"
                     f"{' - outside study area' if external else ''}"
                     f"<br>Region: {getattr(row, 'region', 'n/a')}"),
        ).add_to(layer)
    layer.add_to(fmap)


def _add_legend(fmap, demand: Dict[str, float]) -> None:
    vmax = (max(demand.values()) * 100) if demand else 0.0
    swatches = "".join(
        f"<span style='display:inline-block;width:22px;height:10px;background:{c}'></span>"
        for c in DEMAND_RAMP
    )
    html = f"""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                background:#fcfcfbee;border:1px solid #e6e5e0;border-radius:8px;
                padding:10px 12px;font-family:system-ui,sans-serif;font-size:12px;
                color:#0b0b0b;box-shadow:0 2px 8px rgba(0,0,0,.08)">
      <div style="font-weight:600;margin-bottom:6px">Demand share of national load</div>
      <div>{swatches}</div>
      <div style="display:flex;justify-content:space-between;color:#52514e;margin-top:2px">
        <span>0%</span><span>{vmax:.1f}%</span>
      </div>
      <div style="margin-top:8px;font-weight:600">Network</div>
      <div style="color:#52514e">
        <span style="display:inline-block;width:18px;height:3px;background:{AC_COLOUR};
                     vertical-align:middle"></span> AC corridor &nbsp;
        <span style="display:inline-block;width:18px;height:3px;background:{DC_COLOUR};
                     vertical-align:middle"></span> DC link
      </div>
      <div style="margin-top:6px;color:#52514e">
        <span style="display:inline-block;width:11px;height:11px;border-radius:50%;
                     background:{GEN_COLOUR};opacity:.75;vertical-align:middle"></span>
        circle area = installed capacity
      </div>
    </div>
    """
    fmap.get_root().html.add_child(__import__("folium").Element(html))
