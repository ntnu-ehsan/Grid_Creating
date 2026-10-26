"""Unit tests for the parts of the pipeline that carry the real logic.

These run on small in-memory frames, so the suite works without access to the
M: drive or the PyPSA exports.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from powersys.aggregate import aggregate_lines, aggregate_links, aggregate_generation  # noqa: E402
from powersys.config import ConfigError, load_config  # noqa: E402
from powersys.demand import _normalise_shares  # noqa: E402
from powersys.generation import expand_capacity_years  # noqa: E402
from powersys.iamc import load_iamc_mapping, map_generation_variable  # noqa: E402
from powersys.manifest import (OutputConflictError, guard_output_dir,  # noqa: E402
                               read_manifest, write_manifest)
from powersys.spatial import nearest_bus  # noqa: E402

MAPPING_DIR = ROOT / "config" / "mappings"


# --------------------------------------------------------------------------
# capacity time series
# --------------------------------------------------------------------------


def test_capacity_years_respect_commissioning_and_retirement():
    gen = pd.DataFrame({
        "capacity_mw": [100.0, 200.0, 300.0],
        "date_in": [2015, 2028, np.nan],     # existing, future, unknown
        "date_out": [2027, np.nan, np.nan],  # retires, never, never
    })
    out = expand_capacity_years(gen, [2020, 2030])

    # unit 1 is online in 2020 but retired by 2030
    assert out.loc[0, "2020"] == 100.0
    assert out.loc[0, "2030"] == 0.0
    # unit 2 is not commissioned until 2028
    assert out.loc[1, "2020"] == 0.0
    assert out.loc[1, "2030"] == 200.0
    # a unit with no dates counts as always online
    assert out.loc[2, "2020"] == out.loc[2, "2030"] == 300.0


def test_retirement_exactly_on_the_study_year_is_offline():
    gen = pd.DataFrame({"capacity_mw": [50.0], "date_in": [2000], "date_out": [2030]})
    out = expand_capacity_years(gen, [2030])
    assert out.loc[0, "2030"] == 0.0


# --------------------------------------------------------------------------
# network aggregation
# --------------------------------------------------------------------------


def test_parallel_lines_sum_capacity_and_combine_reactance():
    lines = pd.DataFrame({
        "bus0": ["a", "a"], "bus1": ["b", "b"],
        "s_nom": [1000.0, 1000.0], "x": [4.0, 4.0], "length": [10.0, 10.0],
    })
    out = aggregate_lines(lines, {"a": "R1", "b": "R2"})

    assert len(out) == 1
    assert out.loc[0, "s_nom"] == 2000.0
    # two 4 ohm circuits in parallel -> 2 ohm
    assert out.loc[0, "x"] == pytest.approx(2.0)
    assert out.loc[0, "circuits"] == 2


def test_reversed_region_pairs_aggregate_together():
    lines = pd.DataFrame({
        "bus0": ["a", "c"], "bus1": ["c", "a"],
        "s_nom": [500.0, 700.0], "x": [1.0, 1.0], "length": [1.0, 1.0],
    })
    out = aggregate_lines(lines, {"a": "R1", "c": "R2"})

    assert len(out) == 1, "A>B and B>A must not produce two rows"
    assert out.loc[0, "s_nom"] == 1200.0
    assert (out.loc[0, "region0"], out.loc[0, "region1"]) == ("R1", "R2")


def test_lines_inside_one_region_are_dropped():
    lines = pd.DataFrame({
        "bus0": ["a", "a"], "bus1": ["b", "c"],
        "s_nom": [100.0, 200.0], "x": [1.0, 1.0], "length": [1.0, 1.0],
    })
    # a and b share a region, c is elsewhere
    out = aggregate_lines(lines, {"a": "R1", "b": "R1", "c": "R2"})
    assert len(out) == 1
    assert out.loc[0, "s_nom"] == 200.0


def test_dc_links_aggregate_by_pair():
    links = pd.DataFrame({"bus0": ["a", "a"], "bus1": ["b", "b"], "p_nom": [700.0, 700.0]})
    out = aggregate_links(links, {"a": "R1", "b": "R2"})
    assert out.loc[0, "p_nom"] == 1400.0


def test_generation_aggregation_conserves_capacity():
    gen = pd.DataFrame({
        "bus_id": ["a", "b", "c"],
        "fueltype": ["Hydro", "Hydro", "Wind"],
        "technology": ["Reservoir", "Reservoir", "Onshore"],
        "capacity_mw": [10.0, 20.0, 30.0],
    })
    out = aggregate_generation(gen, {"a": "R1", "b": "R1", "c": "R2"},
                               {"R1": "NO", "R2": "SE"}, [])
    assert out["capacity_mw"].sum() == 60.0
    assert out.loc[out["region"] == "R1", "capacity_mw"].iloc[0] == 30.0


def test_generation_country_comes_from_the_region_map_not_the_id():
    """A bus snapped across a border must not be counted in the wrong country."""
    gen = pd.DataFrame({
        "bus_id": ["a"], "fueltype": ["Wind"], "technology": ["Onshore"],
        "capacity_mw": [5.0],
    })
    out = aggregate_generation(gen, {"a": "DEF0"}, {"DEF0": "DK"}, [])
    assert out.loc[0, "country"] == "DK"


# --------------------------------------------------------------------------
# IAMC mapping
# --------------------------------------------------------------------------


def test_iamc_mapping_prefers_the_exact_pair():
    mapping = load_iamc_mapping(MAPPING_DIR)
    variable, mapped = map_generation_variable("Natural Gas", "CCGT", mapping)
    assert mapped
    assert variable == "Capacity|Electricity|Gas|CCGT|w/o CCS"


def test_iamc_mapping_falls_back_to_fuel_then_technology():
    mapping = load_iamc_mapping(MAPPING_DIR)

    # unknown technology, known fuel
    variable, mapped = map_generation_variable("Nuclear", "Something Odd", mapping)
    assert (variable, mapped) == ("Capacity|Electricity|Nuclear", True)

    # unknown fuel, known technology
    variable, mapped = map_generation_variable("Other", "CCGT", mapping)
    assert (variable, mapped) == ("Capacity|Electricity|Gas|CCGT|w/o CCS", True)

    # neither known -> flagged as unmapped so the check can report it
    variable, mapped = map_generation_variable("Mystery", "", mapping)
    assert (variable, mapped) == ("Capacity|Electricity|Other", False)


def test_fuel_level_mapping_wins_over_technology_level():
    mapping = load_iamc_mapping(MAPPING_DIR)
    variable, _ = map_generation_variable("Solid Biomass", "Steam Turbine", mapping)
    assert variable == "Capacity|Electricity|Biomass|w/o CCS"


# --------------------------------------------------------------------------
# demand
# --------------------------------------------------------------------------


def test_shares_are_normalised_to_one():
    shares = pd.DataFrame({"bus_id": ["a", "b"], "load_share": [3.0, 1.0]})
    out = _normalise_shares(shares, "NO")
    assert out["load_share"].sum() == pytest.approx(1.0)
    assert out.loc[0, "load_share"] == pytest.approx(0.75)


def test_empty_shares_raise_rather_than_produce_nan():
    shares = pd.DataFrame({"bus_id": ["a"], "load_share": [0.0]})
    with pytest.raises(ValueError):
        _normalise_shares(shares, "NO")


# --------------------------------------------------------------------------
# spatial
# --------------------------------------------------------------------------


def test_nearest_bus_stays_inside_its_country_group():
    # The Danish point is physically closer to the Swedish bus, but grouping
    # by country must keep it on the Danish one.
    points = pd.DataFrame({"lon": [12.7], "lat": [55.7], "country": ["DK"]})
    buses = pd.DataFrame({
        "bus_id": ["se1", "dk1"],
        "x": [12.75, 12.0], "y": [55.72, 55.4],
        "country": ["SE", "DK"],
    })
    assigned, distance = nearest_bus(points, buses, "lon", "lat", group_col="country")
    assert assigned.iloc[0] == "dk1"
    assert distance.iloc[0] > 0


def test_nearest_bus_uses_great_circle_distance():
    points = pd.DataFrame({"lon": [10.0], "lat": [60.0]})
    buses = pd.DataFrame({"bus_id": ["b"], "x": [11.0], "y": [60.0]})
    _, distance = nearest_bus(points, buses, "lon", "lat")
    # one degree of longitude at 60 N is about 55 km, not 111 km
    assert 50 < distance.iloc[0] < 60


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


def _minimal_config(tmp_path: Path, **overrides) -> Path:
    for name in ("buses.csv", "lines.csv", "links.csv", "generation.csv"):
        (tmp_path / name).write_text("x\n", encoding="utf-8")

    body = {
        "run": {"name": "test"},
        "spatial": {"level": "national"},
        "countries": ["NO", "SE"],
        "grid": {"buses": str(tmp_path / "buses.csv"),
                 "lines": str(tmp_path / "lines.csv"),
                 "links": str(tmp_path / "links.csv")},
        "generation": {"file": str(tmp_path / "generation.csv")},
        "demand": {"method": {"default": "precomputed"}},
    }
    body.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_unquoted_no_is_read_as_norway_not_false(tmp_path):
    """YAML 1.1 turns an unquoted NO into boolean false."""
    path = tmp_path / "config.yaml"
    base = _minimal_config(tmp_path)
    body = yaml.safe_load(base.read_text(encoding="utf-8"))
    raw = base.read_text(encoding="utf-8").replace("- 'NO'", "- NO").replace('- "NO"', "- NO")
    path.write_text(raw, encoding="utf-8")

    cfg = load_config(path)
    assert "NO" in cfg.countries
    assert False not in cfg.countries
    del body


def test_invalid_level_is_rejected(tmp_path):
    path = _minimal_config(tmp_path, spatial={"level": "county"})
    with pytest.raises(ConfigError, match="spatial.level"):
        load_config(path)


def test_missing_input_file_is_reported_before_the_run(tmp_path):
    path = _minimal_config(tmp_path)
    (tmp_path / "buses.csv").unlink()
    with pytest.raises(ConfigError, match="grid.buses"):
        load_config(path)


def test_cli_overrides_reach_the_config(tmp_path):
    path = _minimal_config(tmp_path)
    cfg = load_config(path, {"spatial.level": "nodal", "iamc.model": "Other",
                             "capacity_timeseries.enabled": False})
    assert cfg.spatial.level == "nodal"
    assert cfg.iamc.model == "Other"
    assert cfg.years == [cfg.iamc.reference_year]


def test_level_override_retargets_the_output_folder(tmp_path):
    """A --level override must not write into the folder named for another level."""
    path = _minimal_config(tmp_path, run={"name": "nesp_nuts2"},
                           spatial={"level": "nuts2", "nuts_geojson": str(tmp_path / "n.geojson")})
    (tmp_path / "n.geojson").write_text("{}", encoding="utf-8")

    assert load_config(path).run.name == "nesp_nuts2"
    assert load_config(path, {"spatial.level": "nodal"}).run.name == "nesp_nodal"


def test_explicit_run_name_wins_over_the_level(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "nesp_nuts2"})
    cfg = load_config(path, {"spatial.level": "nodal", "run.name": "my_run"})
    assert cfg.run.name == "my_run"


def test_run_name_without_a_level_suffix_gains_one(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "nesp"})
    assert load_config(path, {"spatial.level": "nodal"}).run.name == "nesp_nodal"


def test_redundant_level_override_leaves_the_folder_alone(tmp_path):
    """--level national on an already-national config must not rename anything."""
    path = _minimal_config(tmp_path, run={"name": "whatever"})
    assert load_config(path, {"spatial.level": "national"}).run.name == "whatever"


def _manifest(out_dir: Path, level: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "run.json").write_text(
        json.dumps({"level": level, "written": "2026-08-01T15:14:19"}), encoding="utf-8")


def test_guard_blocks_a_different_level_in_the_same_folder(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "nesp_nuts2",
                                          "output_dir": str(tmp_path / "outputs")})
    cfg = load_config(path, {"spatial.level": "nodal", "run.name": "nesp_nuts2"})
    _manifest(cfg.run.dir, "nuts2")

    with pytest.raises(OutputConflictError, match="nuts2"):
        guard_output_dir(cfg)
    guard_output_dir(cfg, force=True)  # --force gets through


def test_guard_allows_repeating_the_same_level(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "run", "output_dir": str(tmp_path / "outputs")})
    cfg = load_config(path)
    _manifest(cfg.run.dir, cfg.spatial.level)
    guard_output_dir(cfg)


def test_guard_allows_a_folder_with_no_manifest(tmp_path):
    """Folders written before manifests existed must not become unusable."""
    path = _minimal_config(tmp_path, run={"name": "run", "output_dir": str(tmp_path / "outputs")})
    cfg = load_config(path)
    cfg.run.dir.mkdir(parents=True)
    (cfg.run.dir / "report.html").write_text("old", encoding="utf-8")
    guard_output_dir(cfg)


def test_guard_ignores_an_unreadable_manifest(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "run", "output_dir": str(tmp_path / "outputs")})
    cfg = load_config(path)
    cfg.run.dir.mkdir(parents=True)
    (cfg.run.dir / "run.json").write_text("{not json", encoding="utf-8")
    guard_output_dir(cfg)


def test_overwrite_false_protects_a_populated_folder(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "run", "overwrite": False,
                                          "output_dir": str(tmp_path / "outputs")})
    cfg = load_config(path)
    guard_output_dir(cfg)  # folder does not exist yet

    cfg.run.dir.mkdir(parents=True)
    guard_output_dir(cfg)  # exists but empty

    (cfg.run.dir / "report.html").write_text("old", encoding="utf-8")
    with pytest.raises(OutputConflictError, match="overwrite"):
        guard_output_dir(cfg)


def test_manifest_round_trips(tmp_path):
    path = _minimal_config(tmp_path, run={"name": "run", "output_dir": str(tmp_path / "outputs")})
    cfg = load_config(path)
    write_manifest(cfg, {"iamc_rows": 895})

    loaded = read_manifest(cfg.run.dir)
    assert loaded["level"] == cfg.spatial.level
    assert loaded["countries"] == cfg.countries
    assert loaded["iamc_rows"] == 895
    assert read_manifest(tmp_path / "nowhere") is None


def test_population_industry_without_a_block_is_rejected(tmp_path):
    path = _minimal_config(tmp_path, demand={"method": {"default": "population+industry"}})
    with pytest.raises(ConfigError, match="population\\+industry"):
        load_config(path)


def test_years_follow_the_timeseries_switch(tmp_path):
    path = _minimal_config(tmp_path,
                           capacity_timeseries={"enabled": True, "years": [2030, 2040]})
    cfg = load_config(path)
    assert cfg.years == [2030, 2040]


def test_derived_grid_does_not_require_a_links_file(tmp_path):
    """AC and DC share one file, so grid.links may be omitted."""
    path = _minimal_config(tmp_path, grid={
        "format": "derived",
        "buses": str(tmp_path / "buses.csv"),
        "lines": str(tmp_path / "lines.csv"),
        "links": None,
    })
    cfg = load_config(path)
    assert cfg.grid.format == "derived"
    assert cfg.grid.links is None


def test_unknown_grid_format_is_rejected(tmp_path):
    path = _minimal_config(tmp_path, grid={
        "format": "something_else",
        "buses": str(tmp_path / "buses.csv"),
        "lines": str(tmp_path / "lines.csv"),
    })
    with pytest.raises(ConfigError, match="grid.format"):
        load_config(path)


def test_several_nuts_sources_are_parsed_with_their_country_filters(tmp_path):
    geo = tmp_path / "nuts.geojson"
    geo.write_text("{}", encoding="utf-8")
    path = _minimal_config(tmp_path, spatial={
        "level": "nuts2",
        "nuts_geojson": [
            {"path": str(geo)},
            {"path": str(geo), "countries": ["GB"]},
        ],
    })
    cfg = load_config(path)
    assert len(cfg.spatial.nuts_sources) == 2
    assert cfg.spatial.nuts_sources[0].countries is None
    assert cfg.spatial.nuts_sources[1].countries == ["GB"]


def test_a_single_nuts_path_still_works(tmp_path):
    geo = tmp_path / "nuts.geojson"
    geo.write_text("{}", encoding="utf-8")
    path = _minimal_config(tmp_path, spatial={"level": "nuts3", "nuts_geojson": str(geo)})
    cfg = load_config(path)
    assert len(cfg.spatial.nuts_sources) == 1
    assert cfg.spatial.nuts_level == 3


def test_external_region_convention_is_validated(tmp_path):
    path = _minimal_config(tmp_path, grid={
        "buses": str(tmp_path / "buses.csv"),
        "lines": str(tmp_path / "lines.csv"),
        "links": str(tmp_path / "links.csv"),
        "external_region": "whatever",
    })
    with pytest.raises(ConfigError, match="external_region"):
        load_config(path)


# --------------------------------------------------------------------------
# case configs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("case", ["nesp", "spain", "northsea"])
def test_shipped_case_configs_are_valid(case):
    """Each case config must load and validate as written."""
    path = ROOT / "config" / f"{case}.yaml"
    if not path.exists():
        pytest.skip(f"{case}.yaml not present")
    cfg = load_config(path)
    assert cfg.countries
    assert cfg.spatial.level in ("nodal", "national", "nuts2", "nuts3")


def test_norway_survives_the_yaml_boolean_trap_in_the_shipped_configs():
    for case in ("nesp", "northsea"):
        path = ROOT / "config" / f"{case}.yaml"
        if not path.exists():
            continue
        cfg = load_config(path)
        assert "NO" in cfg.countries, f"{case}: Norway was lost"
