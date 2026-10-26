"""Automated checks.

The legacy notebooks verified their results by printing tables and reading
them. Here every check returns a pass/warn/fail record, the records are
written to the HTML report, and the run exits non-zero if anything fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from .config import Config
from .logging_utils import get_logger

log = get_logger("validate")

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    data: Optional[pd.DataFrame] = None


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str, data: pd.DataFrame | None = None) -> None:
        self.checks.append(Check(name, status, detail, data))
        level = {PASS: log.info, WARN: log.warning, FAIL: log.error}[status]
        level("[%s] %s - %s", status.upper(), name, detail)

    @property
    def failed(self) -> bool:
        return any(c.status == FAIL for c in self.checks)

    @property
    def counts(self) -> dict:
        return {s: sum(1 for c in self.checks if c.status == s) for s in (PASS, WARN, FAIL)}


def run_checks(cfg: Config, buses: pd.DataFrame, generation: pd.DataFrame,
               demand: pd.DataFrame, aggregated, iamc_table: pd.DataFrame,
               unmapped: pd.DataFrame) -> Report:
    report = Report()

    _check_demand_shares(report, aggregated.demand)
    _check_region_coverage(report, cfg, buses)
    _check_region_countries(report, buses, aggregated)
    _check_generation_assignment(report, generation)
    _check_unmapped_variables(report, unmapped)
    _check_line_endpoints(report, aggregated)
    _check_iamc_table(report, iamc_table)
    _check_capacity_conservation(report, generation, aggregated.generation)
    _check_entsoe(report, cfg, generation)

    counts = report.counts
    log.info("Validation finished: %d passed, %d warnings, %d failures",
             counts[PASS], counts[WARN], counts[FAIL])
    return report


# ---------------------------------------------------------------------------


def _check_demand_shares(report: Report, demand: pd.DataFrame) -> None:
    sums = demand.groupby("country", as_index=False)["load_share"].sum()
    off = sums[(sums["load_share"] - 1.0).abs() > 1e-6]
    if off.empty:
        report.add("Demand shares sum to 1 per country", PASS,
                   f"All {len(sums)} countries within 1e-6 of 1.0")
    else:
        detail = ", ".join(f"{r.country}={r.load_share:.4f}" for r in off.itertuples())
        report.add("Demand shares sum to 1 per country", FAIL,
                   f"{len(off)} country total(s) off: {detail}", off)


def _check_region_coverage(report: Report, cfg: Config, buses: pd.DataFrame) -> None:
    internal = buses[~buses["is_neighbour"]]
    missing = internal["region"].isna().sum()
    sources = (internal["region_source"].value_counts().to_dict()
               if "region_source" in internal else {})
    snapped = sources.get("snapped", 0)
    corrected = sources.get("country_corrected", 0)

    if missing:
        report.add("Every study bus has a region", FAIL,
                   f"{missing} buses without a region")
        return

    notes = []
    if snapped:
        notes.append(f"{snapped} snapped from outside every polygon (offshore or coastline)")
    if corrected:
        note = f"{corrected} moved out of a neighbouring country's region"
        # A border fix should be a few kilometres. A long move means the bus is
        # far from its own country - usually an offshore node - and is worth a
        # second look rather than silent acceptance.
        if "region_fix_km" in internal.columns:
            far = internal["region_fix_km"].dropna()
            if not far.empty and far.max() > 50:
                note += (f", of which {int((far > 50).sum())} by more than 50 km "
                         f"(max {far.max():.0f} km)")
        notes.append(note)

    if notes:
        columns = [c for c in ["bus_id", "country", "region", "region_source",
                               "region_fix_km", "x", "y"] if c in internal.columns]
        report.add("Every study bus has a region", WARN,
                   f"All {len(internal)} buses assigned: " + "; ".join(notes),
                   internal.loc[internal["region_source"] != "within", columns])
    else:
        report.add("Every study bus has a region", PASS,
                   f"All {len(internal)} buses assigned directly")


def _check_region_countries(report: Report, buses: pd.DataFrame, aggregated) -> None:
    """No region may end up owned by a country outside the study area."""
    regions = aggregated.regions
    internal = regions[~regions["is_external"]]
    foreign = internal[~internal["country"].isin(set(buses.loc[~buses["is_neighbour"], "country"]))]

    if foreign.empty:
        report.add("Regions belong to study countries", PASS,
                   f"All {len(internal)} regions map to a study country")
    else:
        report.add("Regions belong to study countries", FAIL,
                   f"{len(foreign)} region(s) belong to a country outside the study area: "
                   f"{', '.join(foreign['region'].astype(str))}", foreign)


def _check_generation_assignment(report: Report, generation: pd.DataFrame) -> None:
    unassigned = generation["bus_id"].isna().sum()
    if unassigned:
        lost = generation.loc[generation["bus_id"].isna(), "capacity_mw"].sum()
        report.add("Generation units assigned to buses", FAIL,
                   f"{unassigned} units ({lost:.0f} MW) have no bus")
        return

    if "bus_distance_km" not in generation.columns:
        # The units came with a bus id, so there is no assignment distance.
        report.add("Generation units assigned to buses", PASS,
                   f"All {len(generation)} units carry a bus id from the source file")
        return

    far = generation[generation["bus_distance_km"] > 100]
    if not far.empty:
        report.add("Generation units assigned to buses", WARN,
                   f"All units assigned, but {len(far)} sit more than 100 km from their bus "
                   f"(max {far['bus_distance_km'].max():.0f} km)",
                   far[["name", "country", "capacity_mw", "bus_id", "bus_distance_km"]].head(20))
    else:
        report.add("Generation units assigned to buses", PASS,
                   f"All {len(generation)} units assigned")


def _check_unmapped_variables(report: Report, unmapped: pd.DataFrame) -> None:
    if unmapped is None or unmapped.empty:
        report.add("All fuel/technology pairs map to an IAMC variable", PASS,
                   "No combination fell back to 'Other'")
    else:
        report.add("All fuel/technology pairs map to an IAMC variable", WARN,
                   f"{len(unmapped)} combination(s), {unmapped['capacity_mw'].sum():.0f} MW, "
                   f"fell back to 'Other'", unmapped)


def _check_line_endpoints(report: Report, aggregated) -> None:
    pairs = pd.concat([
        aggregated.lines[["region0", "region1"]] if not aggregated.lines.empty else pd.DataFrame(),
        aggregated.links[["region0", "region1"]] if not aggregated.links.empty else pd.DataFrame(),
    ], ignore_index=True)
    if pairs.empty:
        report.add("Network endpoints resolve to regions", WARN, "No connections in the output")
        return

    known = set(aggregated.regions["region"])
    endpoints = set(pairs["region0"]) | set(pairs["region1"])
    unknown = sorted(e for e in endpoints if e not in known)
    if unknown:
        report.add("Network endpoints resolve to regions", FAIL,
                   f"{len(unknown)} endpoint(s) unknown: {', '.join(unknown[:6])}")
    else:
        report.add("Network endpoints resolve to regions", PASS,
                   f"All {len(endpoints)} endpoints are known regions")


def _check_iamc_table(report: Report, table: pd.DataFrame) -> None:
    keys = ["model", "scenario", "region", "variable", "unit", "year"]
    dups = table[table.duplicated(subset=keys, keep=False)]
    nulls = int(table["value"].isna().sum())

    if not dups.empty:
        report.add("IAMC keys are unique", FAIL,
                   f"{len(dups)} duplicate rows on {keys}", dups.head(20))
    elif nulls:
        report.add("IAMC keys are unique", WARN, f"No duplicates, but {nulls} null values")
    else:
        report.add("IAMC keys are unique", PASS,
                   f"{len(table)} rows, no duplicates, no null values")


def _check_capacity_conservation(report: Report, nodal: pd.DataFrame,
                                 aggregated: pd.DataFrame) -> None:
    before = float(nodal["capacity_mw"].sum())
    after = float(aggregated["capacity_mw"].sum())
    diff = abs(before - after)
    rel = diff / before if before else 0.0

    if rel < 1e-9:
        report.add("Capacity conserved through aggregation", PASS,
                   f"{after:,.0f} MW in and out")
    elif rel < 0.01:
        report.add("Capacity conserved through aggregation", WARN,
                   f"{diff:,.0f} MW lost ({rel:.3%}) between nodal and aggregated tables")
    else:
        report.add("Capacity conserved through aggregation", FAIL,
                   f"{diff:,.0f} MW lost ({rel:.1%}) between nodal and aggregated tables")


def _check_entsoe(report: Report, cfg: Config, generation: pd.DataFrame) -> None:
    """Compare the national capacity totals against the reference file."""
    path = cfg.generation.validation_file
    if not path or not path.exists():
        report.add("Capacity compared with reference statistics", WARN,
                   "No validation file configured or found - comparison skipped")
        return

    try:
        reference = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        report.add("Capacity compared with reference statistics", WARN,
                   f"Could not read {path.name}: {exc}")
        return

    cols = {c.lower(): c for c in reference.columns}
    country_col = next((cols[c] for c in ("country", "area", "region", "zone") if c in cols), None)
    value_col = next((cols[c] for c in ("capacity", "capacity_mw", "value", "installed_capacity")
                      if c in cols), None)
    if not country_col or not value_col:
        report.add("Capacity compared with reference statistics", WARN,
                   f"{path.name} has no recognisable country/capacity columns "
                   f"(found {list(reference.columns)})")
        return

    ref = reference.copy()
    ref["country"] = _country_code(ref[country_col])
    # Statistics files mark unavailable entries with placeholders such as "n/e".
    ref[value_col] = pd.to_numeric(ref[value_col], errors="coerce")
    ref = ref.dropna(subset=["country", value_col])
    if ref.empty:
        report.add("Capacity compared with reference statistics", WARN,
                   f"{path.name} holds no numeric capacity values to compare against")
        return

    # Compare like with like: if the reference is stamped with a year the model
    # also covers, use that year's capacity rather than the raw total.
    ours_col = "capacity_mw"
    year_note = "installed capacity"
    if "year" in cols:
        years = pd.to_numeric(ref[cols["year"]], errors="coerce").dropna().unique()
        candidates = [int(y) for y in years if str(int(y)) in generation.columns]
        if candidates:
            year = max(candidates)
            ref = ref[pd.to_numeric(ref[cols["year"]], errors="coerce") == year]
            ours_col = str(year)
            year_note = f"capacity online in {year}"

    ref_totals = ref.groupby("country", as_index=False)[value_col].sum()
    ours = generation.groupby("country", as_index=False)[ours_col].sum()

    merged = ours.merge(ref_totals, on="country", how="inner")
    if merged.empty:
        report.add("Capacity compared with reference statistics", WARN,
                   "No overlapping countries between the model and the reference file")
        return

    merged["deviation"] = (merged[ours_col] - merged[value_col]) / merged[value_col]
    missing = sorted(set(generation["country"]) - set(merged["country"]))
    off = merged[merged["deviation"].abs() > cfg.generation.validation_tolerance]
    detail_cols = ["country", ours_col, value_col, "deviation"]
    suffix = f" ({year_note}; no reference for {', '.join(missing)})" if missing else f" ({year_note})"
    if off.empty:
        report.add("Capacity compared with reference statistics", PASS,
                   f"All {len(merged)} countries within "
                   f"{cfg.generation.validation_tolerance:.0%} of the reference{suffix}",
                   merged[detail_cols])
    else:
        detail = ", ".join(f"{r.country} {r.deviation:+.0%}" for r in off.itertuples())
        report.add("Capacity compared with reference statistics", WARN,
                   f"{len(off)} country total(s) deviate more than "
                   f"{cfg.generation.validation_tolerance:.0%}: {detail}{suffix}",
                   merged[detail_cols])


def _country_code(series: pd.Series) -> pd.Series:
    """ISO2 code from labels such as 'Sweden (SE)', 'SE' or 'Sweden'."""
    text = series.astype(str).str.strip()
    in_brackets = text.str.extract(r"\(([A-Za-z]{2})\)", expand=False)
    two_letter = text.where(text.str.fullmatch(r"[A-Za-z]{2}"))
    from .generation import _NAME_TO_ISO
    by_name = text.str.lower().map(_NAME_TO_ISO)
    return in_brackets.fillna(two_letter).fillna(by_name).str.upper()
