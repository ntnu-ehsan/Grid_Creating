"""Self-contained HTML summary report.

Everything (figures included, as base64 PNGs) is inlined into a single file so
the report can be mailed or dropped in a shared folder and still render. Every
figure is paired with the table behind it, so identity never rests on colour
alone.
"""

from __future__ import annotations

import datetime as dt
import html
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import Config
from .logging_utils import get_logger
from .validate import FAIL, PASS, WARN, Report

log = get_logger("report")

STATUS_STYLE = {
    PASS: ("#1baf7a", "pass"),
    WARN: ("#eda100", "warning"),
    FAIL: ("#e34948", "fail"),
}

CSS = """
:root {
  --surface: #fcfcfb; --panel: #ffffff; --border: #e6e5e0;
  --ink: #0b0b0b; --ink-2: #52514e; --ink-3: #76756e;
  --accent: #2a78d6;
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 0 0 64px;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  background: var(--surface); color: var(--ink); line-height: 1.55;
}
.wrap { max-width: 1080px; margin: 0 auto; padding: 0 24px; }
header { border-bottom: 1px solid var(--border); padding: 40px 0 24px; margin-bottom: 32px; }
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
h2 { font-size: 17px; margin: 40px 0 14px; letter-spacing: -0.005em; }
h3 { font-size: 13px; margin: 24px 0 8px; color: var(--ink-2); font-weight: 600; }
.sub { color: var(--ink-3); font-size: 13px; margin: 0; }
.tiles { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 24px 0 8px; }
.tile { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px; }
.tile .label { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--ink-3); }
.tile .value { font-size: 24px; font-weight: 650; margin-top: 4px; letter-spacing: -0.02em; }
.tile .note { font-size: 12px; color: var(--ink-3); margin-top: 2px; }
figure { margin: 0 0 8px; }
figure img { max-width: 100%; height: auto; display: block; }
figcaption { font-size: 12px; color: var(--ink-3); margin-top: 6px; }
.tablewrap { overflow-x: auto; border: 1px solid var(--border); border-radius: 10px; background: var(--panel); }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 8px 12px; border-bottom: 1px solid var(--border); white-space: nowrap; }
th { font-weight: 600; color: var(--ink-2); font-size: 11px; text-transform: uppercase; letter-spacing: .05em; }
tbody tr:last-child td { border-bottom: none; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
details { margin-top: 8px; }
summary { cursor: pointer; font-size: 13px; color: var(--ink-2); }
.check { display: flex; gap: 12px; align-items: flex-start; padding: 11px 14px; border: 1px solid var(--border);
         border-radius: 10px; background: var(--panel); margin-bottom: 8px; }
.badge { flex: 0 0 auto; font-size: 11px; font-weight: 650; text-transform: uppercase; letter-spacing: .05em;
         padding: 2px 8px; border-radius: 999px; color: #fff; }
.check .name { font-weight: 600; font-size: 13.5px; }
.check .detail { color: var(--ink-2); font-size: 13px; }
code { background: #f2f1ec; padding: 1px 5px; border-radius: 4px; font-size: 12.5px; }
.kv { display: grid; grid-template-columns: 200px 1fr; gap: 4px 16px; font-size: 13px; }
.kv dt { color: var(--ink-3); }
.kv dd { margin: 0; }
.maplink { display: inline-block; background: var(--accent); color: #fff; text-decoration: none;
           padding: 9px 16px; border-radius: 8px; font-size: 13.5px; font-weight: 600; }
footer { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--border);
         color: var(--ink-3); font-size: 12px; }
"""


def _fmt(value) -> str:
    if isinstance(value, float):
        if value != value:  # NaN
            return "-"
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) >= 1:
            return f"{value:,.2f}"
        return f"{value:.4f}"
    return html.escape(str(value))


def _table(frame: pd.DataFrame, limit: Optional[int] = None) -> str:
    if frame is None or frame.empty:
        return "<p class='sub'>No rows.</p>"
    shown = frame.head(limit) if limit else frame
    numeric = {c for c in shown.columns if pd.api.types.is_numeric_dtype(shown[c])}

    head = "".join(
        f"<th class='{'num' if c in numeric else ''}'>{html.escape(str(c))}</th>"
        for c in shown.columns
    )
    rows = "".join(
        "<tr>" + "".join(
            f"<td class='{'num' if c in numeric else ''}'>{_fmt(row[c])}</td>"
            for c in shown.columns
        ) + "</tr>"
        for _, row in shown.iterrows()
    )
    more = ""
    if limit and len(frame) > limit:
        more = f"<p class='sub'>Showing {limit} of {len(frame):,} rows.</p>"
    return f"<div class='tablewrap'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>{more}"


def _tile(label: str, value: str, note: str = "") -> str:
    note_html = f"<div class='note'>{html.escape(note)}</div>" if note else ""
    return (f"<div class='tile'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{html.escape(value)}</div>{note_html}</div>")


def _figure(figures: Dict[str, str], key: str, caption: str) -> str:
    if key not in figures:
        return ""
    return (f"<figure><img src='{figures[key]}' alt='{html.escape(caption)}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>")


def _checks_section(validation: Report) -> str:
    blocks = []
    for check in validation.checks:
        colour, label = STATUS_STYLE[check.status]
        detail_table = ""
        if check.data is not None and not check.data.empty:
            detail_table = (f"<details><summary>Details ({len(check.data)} rows)</summary>"
                            f"{_table(check.data, limit=25)}</details>")
        blocks.append(
            f"<div class='check'>"
            f"<span class='badge' style='background:{colour}'>{label}</span>"
            f"<div><div class='name'>{html.escape(check.name)}</div>"
            f"<div class='detail'>{html.escape(check.detail)}</div>{detail_table}</div></div>"
        )
    return "".join(blocks)


def build_report(cfg: Config, validation: Report, aggregated, generation: pd.DataFrame,
                 demand: pd.DataFrame, grid_summary: pd.DataFrame,
                 iamc_table: pd.DataFrame, figures: Dict[str, str],
                 out_path: Path, map_path: Optional[Path] = None,
                 written: Optional[Dict[str, Path]] = None) -> Path:
    counts = validation.counts
    years = cfg.years
    internal_regions = aggregated.regions[~aggregated.regions["is_external"]]

    variable_counts = (iamc_table.groupby("variable", as_index=False)
                       .agg(rows=("value", "size"), total=("value", "sum"))
                       .sort_values("rows", ascending=False))

    demand_methods = (demand.groupby(["country", "method"], as_index=False)
                      .agg(buses=("bus_id", "nunique"), share=("load_share", "sum")))

    capacity_by_country = (generation.groupby("country", as_index=False)["capacity_mw"]
                           .sum().sort_values("capacity_mw", ascending=False))

    map_block = ""
    if map_path is not None:
        map_block = (f"<p><a class='maplink' href='{html.escape(map_path.name)}'>"
                     f"Open the interactive map &rarr;</a></p>")

    files_block = ""
    if written:
        items = "".join(f"<dt>{html.escape(k.upper())}</dt><dd><code>{html.escape(v.name)}</code></dd>"
                        for k, v in written.items())
        files_block = f"<dl class='kv'>{items}</dl>"

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(cfg.run.name)} - power system data preparation</title>
<style>{CSS}</style></head>
<body><div class="wrap">

<header>
  <h1>{html.escape(cfg.run.name)}</h1>
  <p class="sub">
    {html.escape(cfg.iamc.model)} / {html.escape(cfg.iamc.scenario)} &middot;
    spatial level <b>{html.escape(cfg.spatial.level)}</b> &middot;
    countries {html.escape(", ".join(cfg.countries))} &middot;
    generated {dt.datetime.now():%Y-%m-%d %H:%M}
  </p>
</header>

<div class="tiles">
  {_tile("Regions", f"{len(internal_regions):,}", f"level: {cfg.spatial.level}")}
  {_tile("Buses", f"{int(grid_summary['buses'].sum()):,}", "inside the study area")}
  {_tile("Installed capacity", f"{generation['capacity_mw'].sum():,.0f} MW",
         f"{len(generation):,} units")}
  {_tile("Study years", f"{len(years)}",
         "capacity time series" if cfg.capacity_timeseries.enabled else "single snapshot")}
  {_tile("IAMC rows", f"{len(iamc_table):,}",
         f"{iamc_table['variable'].nunique()} variables")}
  {_tile("Checks", f"{counts[PASS]}/{len(validation.checks)}",
         f"{counts[WARN]} warnings, {counts[FAIL]} failures")}
</div>

{map_block}

<h2>Validation</h2>
{_checks_section(validation)}

<h2>Generation</h2>
{_figure(figures, "capacity_by_country", "Installed capacity by country and fuel")}
{_table(capacity_by_country)}

{_figure(figures, "capacity_trajectory",
         "Capacity online per fuel across the study years, from commissioning and decommissioning dates")}

<h3>Capacity by region</h3>
{_figure(figures, "capacity_by_region", "Largest regions by installed capacity")}
<details><summary>Full aggregated generation table</summary>
{_table(aggregated.generation, limit=60)}</details>

<h2>Demand</h2>
{_figure(figures, "demand_by_region", "Largest regions by share of national electricity demand")}
<h3>Method per country</h3>
{_table(demand_methods)}
<details><summary>Full demand distribution</summary>
{_table(aggregated.demand, limit=60)}</details>

<h2>Network</h2>
<h3>Grid summary by country</h3>
{_table(grid_summary)}
<h3>Aggregated AC corridors</h3>
{_table(aggregated.lines, limit=30)}
<h3>Aggregated DC links</h3>
{_table(aggregated.links, limit=30)}

<h2>IAMC output</h2>
{files_block}
{_table(variable_counts)}
<details><summary>First rows of the assembled IAMC table</summary>
{_table(iamc_table, limit=40)}</details>

<h2>Run configuration</h2>
<dl class="kv">
  <dt>Config file</dt><dd><code>{html.escape(str(cfg.source_path))}</code></dd>
  <dt>Model / scenario</dt><dd>{html.escape(cfg.iamc.model)} / {html.escape(cfg.iamc.scenario)}</dd>
  <dt>Spatial level</dt><dd>{html.escape(cfg.spatial.level)}</dd>
  <dt>Years</dt><dd>{", ".join(str(y) for y in years)}</dd>
  <dt>Grid buses</dt><dd><code>{html.escape(str(cfg.grid.buses))}</code></dd>
  <dt>Generation</dt><dd><code>{html.escape(str(cfg.generation.file))}</code></dd>
  <dt>Demand methods</dt><dd>{html.escape(", ".join(f"{c}: {cfg.demand.method_for(c)}" for c in cfg.countries))}</dd>
</dl>

<footer>
  Generated by the power system data preparation pipeline.
  Figures are inlined; every chart is backed by the table beside it.
</footer>

</div></body></html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc, encoding="utf-8")
    log.info("Wrote HTML report -> %s", out_path)
    return out_path
