"""Configuration loading and validation.

The whole pipeline is driven by a single YAML file. This module turns that
file into a small set of dataclasses so that the rest of the code never has
to reach into raw dictionaries, and so that a typo in the config produces a
clear error before any data is read.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

VALID_LEVELS = ("nodal", "national", "nuts2", "nuts3")
VALID_DEMAND_METHODS = ("atlas", "population", "population+industry", "precomputed")


class ConfigError(ValueError):
    """Raised when the configuration file is inconsistent or incomplete."""


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


@dataclass
class RunCfg:
    name: str = "run"
    output_dir: Path = Path("outputs")
    overwrite: bool = True

    @property
    def dir(self) -> Path:
        return self.output_dir / self.name


@dataclass
class IamcCfg:
    model: str = "PowerSystem"
    scenario: str = "Baseline"
    reference_year: int = 2024
    region_separator: str = ">"
    demand_variable: str = "Final Energy|Electricity [Share]"
    demand_unit: str = "%"


@dataclass
class NutsSource:
    """One NUTS geometry file, optionally restricted to some countries.

    The Eurostat 2024 release dropped the United Kingdom, so a case covering
    GB has to take those regions from the 2021 release. Listing several
    sources keeps that explicit instead of hidden in the code.
    """
    path: Path
    countries: Optional[List[str]] = None   # None = every country in the file


@dataclass
class SpatialCfg:
    level: str = "nuts2"
    nuts_sources: List[NutsSource] = field(default_factory=list)
    snap_outside_buses: bool = True
    fix_country_mismatch: bool = True

    @property
    def nuts_level(self) -> Optional[int]:
        """NUTS level number, or None when the level is not NUTS based."""
        return {"nuts2": 2, "nuts3": 3}.get(self.level)

    @property
    def needs_geometry(self) -> bool:
        return self.nuts_level is not None


@dataclass
class CapacityTimeseriesCfg:
    enabled: bool = True
    years: List[int] = field(default_factory=lambda: [2020, 2025, 2030])


@dataclass
class GridCfg:
    buses: Optional[Path] = None
    lines: Optional[Path] = None
    links: Optional[Path] = None
    neighbouring_buses: Optional[Path] = None
    keep_cross_border: bool = True
    external_suffix: str = "_EXT"
    # How a bus outside the study area is named:
    #   "suffix" -> <country><external_suffix>, e.g. DE_EXT
    #   "region" -> its own region id where one exists, e.g. the NUTS id PT111
    external_region: str = "suffix"
    # "pypsa"   - separate line/link files that already carry s_nom, x and p_nom
    # "derived" - one file holding AC and DC, with the electrical parameters
    #             computed from voltage, current rating, circuits and length
    format: str = "pypsa"
    dc_column: str = "dc"
    dc_true_values: List[str] = field(default_factory=lambda: ["t", "true", "1", "yes"])
    current_column: str = "Imax"
    reactance_per_length_column: str = "x_per_length"
    line_id_column: str = "line_id"
    # Regex stripped from the line id to collapse per-circuit rows into one
    # physical line. Empty or null disables the collapse.
    collapse_circuit_suffix: Optional[str] = None


@dataclass
class GenerationCfg:
    file: Optional[Path] = None
    format: str = "pypsa"
    retire_before: Optional[int] = 2020
    assign_to_bus: str = "nearest"
    bus_column: str = "bus_id"
    # Country to use when the generation file has no country column at all
    # (single-country cases such as Spain).
    default_country: Optional[str] = None
    validation_file: Optional[Path] = None
    validation_tolerance: float = 0.25


@dataclass
class DemandCfg:
    method: Dict[str, str] = field(default_factory=lambda: {"default": "atlas"})
    atlas: Dict[str, Any] = field(default_factory=dict)
    population: Dict[str, Any] = field(default_factory=dict)
    industry: Dict[str, Any] = field(default_factory=dict)
    precomputed: Dict[str, Any] = field(default_factory=dict)
    cache_dir: Path = Path(".cache")
    reuse_cache: bool = True

    def method_for(self, country: str) -> str:
        return self.method.get(country, self.method.get("default", "atlas"))


@dataclass
class OutputsCfg:
    csv: bool = True
    iamc_xlsx: bool = True
    iamc_csv: bool = True
    report_html: bool = True
    map_html: bool = True
    figures: bool = True


@dataclass
class Config:
    run: RunCfg
    iamc: IamcCfg
    spatial: SpatialCfg
    countries: List[str]
    capacity_timeseries: CapacityTimeseriesCfg
    grid: GridCfg
    generation: GenerationCfg
    demand: DemandCfg
    outputs: OutputsCfg
    country_names: Dict[str, str]
    logging_level: str = "INFO"
    source_path: Optional[Path] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    # -- derived ---------------------------------------------------------
    @property
    def years(self) -> List[int]:
        """Years written to the IAMC output."""
        if self.capacity_timeseries.enabled:
            return list(self.capacity_timeseries.years)
        return [self.iamc.reference_year]

    def country_name(self, code: str) -> str:
        return self.country_names.get(code, code)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _as_path(value: Any) -> Optional[Path]:
    return Path(value) if value else None


def _iso(value: Any) -> str:
    """Country code as a string.

    YAML 1.1 reads an unquoted ``NO`` as boolean false, which would silently
    drop Norway from the study. Booleans are therefore mapped back to the
    country codes they came from.
    """
    if isinstance(value, bool):
        return "NO" if value is False else "YES"
    return str(value).strip().upper()


def _retarget_run_name(name: str, level: str) -> str:
    """Rename the output folder to match a level that was overridden.

    ``run.name`` is a free-text folder name, so a config that says
    ``nesp_nuts2`` keeps that name whatever ``--level`` is passed - and with
    ``overwrite: true`` a nodal run would land on top of the NUTS2 results.
    Any trailing level suffix is replaced by the level actually being run,
    giving ``nesp_nuts2`` -> ``nesp_nodal``. Names that carry no level suffix
    simply gain one.
    """
    base = name
    for known in VALID_LEVELS:
        if base.lower().endswith(f"_{known}"):
            base = base[: -len(known) - 1]
            break
    return f"{base}_{level}"


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> Config:
    """Read the YAML config, apply CLI overrides and validate the result."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    file_level = str(raw.get("spatial", {}).get("level", "nuts2")).lower()

    for key, value in overrides.items():
        _apply_override(raw, key, value)

    # A level override with no explicit run name would otherwise write into the
    # folder named for the level in the config file, silently overwriting it.
    level_override = str(overrides.get("spatial.level", file_level)).lower()
    if level_override != file_level and "run.name" not in overrides:
        raw.setdefault("run", {})
        raw["run"]["name"] = _retarget_run_name(
            str(raw["run"].get("name", "run")), level_override
        )

    run_raw = raw.get("run", {})
    cfg = Config(
        run=RunCfg(
            name=run_raw.get("name", "run"),
            output_dir=Path(run_raw.get("output_dir", "outputs")),
            overwrite=bool(run_raw.get("overwrite", True)),
        ),
        iamc=IamcCfg(**{k: v for k, v in raw.get("iamc", {}).items() if k in IamcCfg.__annotations__}),
        spatial=SpatialCfg(
            level=str(raw.get("spatial", {}).get("level", "nuts2")).lower(),
            nuts_sources=_nuts_sources(raw.get("spatial", {})),
            snap_outside_buses=bool(raw.get("spatial", {}).get("snap_outside_buses", True)),
            fix_country_mismatch=bool(raw.get("spatial", {}).get("fix_country_mismatch", True)),
        ),
        countries=[_iso(c) for c in raw.get("countries", [])],
        capacity_timeseries=CapacityTimeseriesCfg(
            enabled=bool(raw.get("capacity_timeseries", {}).get("enabled", True)),
            years=[int(y) for y in raw.get("capacity_timeseries", {}).get("years", [2020, 2025, 2030])],
        ),
        grid=GridCfg(
            buses=_as_path(raw.get("grid", {}).get("buses")),
            lines=_as_path(raw.get("grid", {}).get("lines")),
            links=_as_path(raw.get("grid", {}).get("links")),
            neighbouring_buses=_as_path(raw.get("grid", {}).get("neighbouring_buses")),
            keep_cross_border=bool(raw.get("grid", {}).get("keep_cross_border", True)),
            external_suffix=raw.get("grid", {}).get("external_suffix", "_EXT"),
            external_region=str(raw.get("grid", {}).get("external_region", "suffix")).lower(),
            format=str(raw.get("grid", {}).get("format", "pypsa")).lower(),
            dc_column=raw.get("grid", {}).get("dc_column", "dc"),
            dc_true_values=[str(v).lower() for v in raw.get("grid", {}).get(
                "dc_true_values", ["t", "true", "1", "yes"])],
            current_column=raw.get("grid", {}).get("current_column", "Imax"),
            reactance_per_length_column=raw.get("grid", {}).get(
                "reactance_per_length_column", "x_per_length"),
            line_id_column=raw.get("grid", {}).get("line_id_column", "line_id"),
            collapse_circuit_suffix=raw.get("grid", {}).get("collapse_circuit_suffix") or None,
        ),
        generation=GenerationCfg(
            file=_as_path(raw.get("generation", {}).get("file")),
            format=str(raw.get("generation", {}).get("format", "pypsa")).lower(),
            retire_before=raw.get("generation", {}).get("retire_before", 2020),
            assign_to_bus=str(raw.get("generation", {}).get("assign_to_bus", "nearest")).lower(),
            bus_column=raw.get("generation", {}).get("bus_column", "bus_id"),
            default_country=(_iso(raw["generation"]["default_country"])
                             if raw.get("generation", {}).get("default_country") is not None
                             else None),
            validation_file=_as_path(raw.get("generation", {}).get("validation_file")),
            validation_tolerance=float(raw.get("generation", {}).get("validation_tolerance", 0.25)),
        ),
        demand=DemandCfg(
            method={("default" if k == "default" else _iso(k)): str(v).lower()
                    for k, v in raw.get("demand", {}).get("method", {}).items()},
            atlas=raw.get("demand", {}).get("atlas", {}) or {},
            population=raw.get("demand", {}).get("population", {}) or {},
            industry={_iso(k): v for k, v in (raw.get("demand", {}).get("industry") or {}).items()},
            precomputed=raw.get("demand", {}).get("precomputed", {}) or {},
            cache_dir=Path(raw.get("demand", {}).get("cache_dir", ".cache")),
            reuse_cache=bool(raw.get("demand", {}).get("reuse_cache", True)),
        ),
        outputs=OutputsCfg(**{k: bool(v) for k, v in (raw.get("outputs") or {}).items()
                              if k in OutputsCfg.__annotations__}),
        country_names={_iso(k): v for k, v in (raw.get("country_names") or {}).items()},
        logging_level=str(raw.get("logging", {}).get("level", "INFO")).upper(),
        source_path=path,
        raw=raw,
    )

    _validate(cfg)
    return cfg


def _nuts_sources(spatial_raw: Dict[str, Any]) -> List[NutsSource]:
    """Accept either a single path or a list of {path, countries} entries."""
    raw = spatial_raw.get("nuts_geojson")
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        return [NutsSource(path=Path(raw))]

    sources: List[NutsSource] = []
    for entry in raw:
        if isinstance(entry, (str, Path)):
            sources.append(NutsSource(path=Path(entry)))
            continue
        countries = entry.get("countries")
        sources.append(NutsSource(
            path=Path(entry["path"]),
            countries=[_iso(c) for c in countries] if countries else None,
        ))
    return sources


def _apply_override(raw: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a nested key such as ``spatial.level`` in the raw config dict."""
    parts = dotted_key.split(".")
    node = raw
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _validate(cfg: Config) -> None:
    if cfg.spatial.level not in VALID_LEVELS:
        raise ConfigError(
            f"spatial.level must be one of {VALID_LEVELS}, got '{cfg.spatial.level}'"
        )
    if not cfg.countries:
        raise ConfigError("No countries listed in the config.")

    if cfg.spatial.needs_geometry:
        if not cfg.spatial.nuts_sources:
            raise ConfigError(f"spatial.level='{cfg.spatial.level}' requires spatial.nuts_geojson")
        for source in cfg.spatial.nuts_sources:
            if not source.path.exists():
                raise ConfigError(f"NUTS geometry file not found: {source.path}")

    if cfg.grid.format not in ("pypsa", "derived"):
        raise ConfigError(f"grid.format must be 'pypsa' or 'derived', got '{cfg.grid.format}'")

    if cfg.grid.external_region not in ("suffix", "region"):
        raise ConfigError("grid.external_region must be 'suffix' or 'region', "
                          f"got '{cfg.grid.external_region}'")

    required = [("grid.buses", cfg.grid.buses),
                ("grid.lines", cfg.grid.lines),
                ("generation.file", cfg.generation.file)]
    # A "derived" grid keeps AC and DC in the line file, so grid.links is optional.
    if cfg.grid.format == "pypsa":
        required.append(("grid.links", cfg.grid.links))

    for name, path in required:
        if path is None:
            raise ConfigError(f"{name} is not set in the config.")
        if not path.exists():
            raise ConfigError(f"{name} points to a missing file: {path}")

    for name, path in (("grid.links", cfg.grid.links),
                       ("grid.neighbouring_buses", cfg.grid.neighbouring_buses)):
        if path is not None and not path.exists():
            raise ConfigError(f"{name} points to a missing file: {path}")

    for country in cfg.countries:
        method = cfg.demand.method_for(country)
        if method not in VALID_DEMAND_METHODS:
            raise ConfigError(
                f"demand.method for {country} must be one of {VALID_DEMAND_METHODS}, got '{method}'"
            )
        if method == "population+industry" and country not in (cfg.demand.industry or {}):
            raise ConfigError(
                f"demand.method for {country} is 'population+industry' but no "
                f"demand.industry.{country} block was provided."
            )

    if cfg.capacity_timeseries.enabled and not cfg.capacity_timeseries.years:
        raise ConfigError("capacity_timeseries.enabled is true but no years were given.")
