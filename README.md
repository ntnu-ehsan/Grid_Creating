# Power system data preparation - North Sea

Builds the North Sea power-system input dataset (NO, GB, FR, DE, BE, NL, DK)
and writes it in IAMC format, together with an HTML report and an interactive
map.

Everything is driven by one configuration file. There are no notebooks: the
same code path produces nodal, national, NUTS2 or NUTS3 output, and assembles
the delivery workbook itself.

The pipeline is case-agnostic - it also runs the Nordic (NESP) and Spanish
case studies - but this copy ships the North Sea case and its input data only.
See "Adding another case" at the end if you want to point it at other data.

---

## Quick start

```bash
python run.py --case northsea --level nodal      # one region per transmission bus
python run.py --case northsea --level national   # one region per country
python run.py --case northsea --level nuts3      # writes to outputs/northsea_nuts3/
python run.py --case northsea --model "North Sea" --scenario "Baseline v2"
python run.py --case northsea --countries NO,GB  # a subset of the case
python run.py --case northsea --years 2030,2040  # different study years
python run.py --case northsea --no-timeseries    # single capacity snapshot
python run.py --case northsea --strict           # non-zero exit if a check fails
```
ash
# From the repository root, in an environment with the packages in
# requirements.txt (geopandas, folium, matplotlib, openpyxl, pyyaml, scipy):
python -m pip install -r requirements.txt

python run.py --case northsea       # the whole pipeline, about 20 seconds

python run.py --list-cases          # what is available
python run.py --config my.yaml      # a config file kept elsewhere
```

All input data is bundled in `data/`, so there is nothing to download and no
path to edit. See `data/README.md` for what each file is.

Output lands in `outputs/<run name>/`:

```
outputs/northsea_nuts2/
├── report.html                    <- open this first
├── map.html                       <- interactive map, linked from the report
├── iamc/
│   ├── northsea_nuts2_IAMC.xlsx   <- the deliverable, ready to submit
│   └── northsea_nuts2_IAMC.csv
├── data/                          <- tidy CSV for every stage
│   ├── buses.csv  generation_nodal.csv  demand_nodal.csv
│   ├── generation_aggregated.csv  demand_aggregated.csv
│   ├── lines_aggregated.csv  links_aggregated.csv  regions.csv
│   └── grid_summary.csv  demand_summary.csv
├── figures/                       <- the report's charts as PNG
├── run.json                       <- what this run was: level, countries, years
└── pipeline.log
```

### Common variations

```bash
python run.py --case northsea --level nodal       # one region per transmission bus
python run.py --case northsea --level national    # one region per country
python run.py --case northsea --level nuts3       # writes to outputs/northsea_nuts3/
python run.py --case northsea --model "North Sea" --scenario "Baseline v2"
python run.py --case northsea --countries NO,GB   # a subset of the case
python run.py --case northsea --years 2030,2040   # different study years
python run.py --case northsea --no-timeseries     # single capacity snapshot
python run.py --case northsea --strict            # non-zero exit if a check fails
python run.py --case northsea --level nodal --name my_run  # pick the folder yourself
```

Command-line flags override the config file; the file itself is never modified.

### Not overwriting the wrong folder

`run.name` in the config decides the output folder, and `run.overwrite` defaults
to true, so two runs at different levels pointed at one folder would leave the
second silently on top of the first. Two things prevent that.

**The run name follows the level.** A `--level` override with no `--name`
replaces any trailing level suffix with the level actually being run, so
`--level nodal` writes to `outputs/northsea_nodal/` rather than
`outputs/northsea_nuts2/`. An explicit `--name` always wins, and a run with no level
override keeps the configured name unchanged.

**Each run leaves a `run.json`** recording its level, countries, years and
check counts. A run that would overwrite a folder whose manifest holds a
different level stops with exit code 5 and writes nothing:

```
Refusing to overwrite: outputs\northsea_nuts2 holds a 'nuts2' run
(written 2026-08-02T13:05:07), but this run is 'nodal'.
```

Pass `--force` to overwrite anyway, or `--name` to write elsewhere. Setting
`run.overwrite: false` extends the same protection to a rerun at the *same*
level. A folder with no manifest — written before this existed, or left by a
crashed run — is not protected, since there is nothing to compare against.

### Adding another case

Copy `config/northsea.yaml` to `config/<name>.yaml`, point the paths at the new
data and adjust `countries`. `python run.py --case <name>` picks it up with no
code change. The blocks most likely to need attention are `grid.format`
(whether the line ratings are given or have to be derived) and `demand.method`.

---

## What the pipeline does

```
grid ──> generation ──> demand ──> regions ──> aggregation ──> IAMC ──> report + map
                                                                  │
                                                              validation
```

1. **Grid** — reads the PyPSA buses, AC lines and DC links, keeps the study
   countries, and retains the neighbouring buses that interconnectors land on
   so cross-border capacity is not lost.
2. **Generation** — reads the powerplant table, drops units decommissioned
   before the cut-off, normalises the fuel and technology labels, and assigns
   each unit to its nearest bus **within its own country**. With
   `capacity_timeseries.enabled`, a capacity column is derived per study year
   from each unit's commissioning and decommissioning dates.
3. **Demand** — per-country method (see below), producing a bus-level share
   that sums to 1.0 within each country.
4. **Regions** — resolves the configured spatial level onto a single `region`
   column. Offshore buses are snapped to the nearest bus in the same country;
   border buses that land in a neighbouring country's polygon are moved back.
5. **Aggregation** — sums capacity per region, sums demand shares, and
   aggregates the network onto region pairs (capacities add, reactances
   combine in parallel). Connections inside one region are dropped.
6. **IAMC** — maps each fuel/technology pair to an IAMC variable, converts
   demand and network data, and concatenates all three into one long-format
   table: `model, scenario, region, variable, unit, year, value`.
7. **Validation, report and map** — every check is recorded pass/warn/fail and
   rendered into `report.html`; the map shows regions, capacity and the network.

---

## Input formats

Case studies do not share one input format, so two grid readers exist. The
North Sea data uses `pypsa`; the other is documented because you may need it
for new data.

| `grid.format` | Expects |
|---|---|
| `pypsa` | separate line and link files that already carry `s_nom`, `x` and `p_nom` — **what the North Sea data uses** |
| `derived` | one file holding AC and DC, told apart by a `dc` column, with ratings computed from voltage, current and circuits |

The derived form computes

```
AC capacity   S = sqrt(3) * V[kV] * Imax[kA] * circuits    -> MVA
AC reactance  x = x_per_length[ohm/km] * length[km] / circuits
DC capacity   P = V[kV] * Imax[kA] * circuits              -> MW
```

Where a file stores one row per **circuit** rather than per line (the Spanish
`LTGES0001a` / `LTGES0001b` pattern), `collapse_circuit_suffix` folds them into
one physical line first — without it the capacity is counted once per circuit.

Generation can be read two ways: `assign_to_bus: nearest` matches each unit to
its closest bus within its own country, while `assign_to_bus: column` uses a
`bus_id` the source already provides - which is what the North Sea data does.

**NUTS geometry** accepts several sources, because Eurostat dropped the United
Kingdom after the 2021 release:

```yaml
spatial:
  nuts_geojson:
    - path: ".../NUTS_RG_60M_2024_3035.geojson"
    - path: ".../NUTS_RG_60M_2021_3035.geojson"
      countries: ["GB"]
```

**Interconnector endpoints** outside the study area follow whichever convention
the case wants, set by `grid.external_region`: `suffix` gives `NO09>DE_EXT`,
`region` gives the neighbour's real region id, `ES113>PT111`.

---

## Demand: one method per country

The JRC Atlas raster product has **no coverage for Norway or the UK**, so a
single method cannot serve the whole case study. `demand.method` routes each
country separately:

| Method | What it uses | Used for |
|---|---|---|
| `atlas` | Atlas 1 km electricity-demand pixels, assigned to the nearest bus | BE, DE, DK, FR, NL |
| `population` | WorldPop 1 km population as a proxy for demand | GB, where Atlas has no coverage |
| `population+industry` | population proxy **plus** explicit industrial and offshore-oil sites, weighted by their share of national load | NO |
| `precomputed` | a finished bus-level share table | **the active method here** |

The North Sea dataset arrives with a finished bus-level share table, so
`precomputed` is the active method and the pipeline uses it out of the box.
That table was itself built from Atlas pixels (BE, DE, DK, FR, NL), population
(GB) and population plus industry (NO); those three methods stay configured in
`config/northsea.yaml` so the distribution can be rebuilt, but the raw rasters
they read are **not bundled** (about 210 MB) — ask Ehsan for them if you need
to re-derive demand rather than use the finished table.

Whatever the method, shares are renormalised to exactly 1.0 per country, which
incidentally repairs the input table, where country totals ranged from 0.90
to 1.07.

```yaml
demand:
  method:
    default: atlas
    "NO": population+industry     # quote NO - unquoted it is boolean false in YAML
```

For `population+industry`, the industrial and offshore sites keep their own
spatial pattern and their combined share of national load; population spreads
the remainder. The pipeline reports that split in the log and the report — for
Norway the sites currently cover about 28% of national load.

Pixel-to-bus assignment is cached in `.cache/`, keyed on the input file's
modification time. Delete the folder or set `reuse_cache: false` to recompute.

---

## Spatial levels

The case runs at every level. Region counts from the bundled data:

| `spatial.level` | Region id | Regions |
|---|---|---|
| `nodal` | transmission bus id | 2,999 |
| `national` | ISO2 country code | 7 |
| `nuts2` | NUTS level 2 id | 133 |
| `nuts3` | NUTS level 3 id | 542 |

Connections are written as `A>B` (separator configurable). Buses outside the
study area become `<country>_EXT`, so an interconnector reads `NO09>DE_EXT`.

---

## Configuration reference

Each file in `config/` is commented throughout. The main blocks are:

| Block | Controls |
|---|---|
| `run` | run name and output folder |
| `iamc` | **model and scenario name**, reference year, region separator, demand variable |
| `spatial` | level, NUTS geometry, offshore snapping, border correction |
| `countries` | which countries are in the study |
| `capacity_timeseries` | whether to write a capacity time series, and for which years |
| `grid` | bus/line/link files, cross-border handling, external suffix |
| `generation` | source file, retirement cut-off, bus assignment, validation reference |
| `demand` | per-country method and the inputs each method needs |
| `outputs` | which artefacts to write |

Two mapping files sit alongside it:

- `config/mappings/technology_labels.yaml` — normalises raw fuel and
  technology spellings.
- `config/mappings/iamc_variables.yaml` — maps a normalised
  `(fuel, technology)` pair to an IAMC variable, with fuel-only and
  technology-only fallbacks. Anything unmatched becomes
  `Capacity|Electricity|Other` **and is reported** as an unmapped combination.

> **YAML gotcha:** `NO` without quotes parses as boolean `false`. The config
> quotes it, and the loader converts a stray boolean back to `"NO"`, but it is
> worth knowing when editing.

---

## Validation

Checks run on every execution and appear at the top of the report:

- demand shares sum to 1.0 per country
- every study bus has a region (reports snapped and border-corrected buses)
- no region belongs to a country outside the study area
- every generation unit is assigned to a bus, flagging units far from one
- every fuel/technology pair maps to an IAMC variable
- all network endpoints resolve to known regions
- IAMC keys are unique and no value is null
- installed capacity is conserved through aggregation
- national capacity totals against the reference statistics file

`--strict` makes a failure exit non-zero, for use in a scheduled run.

### Reconciliation with the previous notebooks

The pipeline replaces a set of Jupyter notebooks. To show the rewrite changed
no numbers, the Spanish case (built by the same code, not shipped here) was
checked against its notebook: 0 of 1,236 buses assigned differently, an
identical set of aggregated line pairs, and a difference of 0.000 in line
capacity, reactance and installed generation.

The North Sea results differ slightly from the older `lines_nuts2.csv`
produced by the notebooks — 260 aggregated line pairs against 259, and
transmission capacity about 0.4% higher. The reason is that 28 border buses
are moved out of a neighbouring country's NUTS region, a correction the
earlier pipeline did not make. Check this against your expectations before
using the output.

---

## Development

```bash
python -m pytest tests -q
```

The tests cover the parts that carry the logic — capacity year expansion,
parallel-line aggregation, the IAMC mapping fallback chain, share
normalisation, country-grouped nearest-bus assignment and config validation.
They use small in-memory frames, so they run without the large input files.

Package layout:

```
src/powersys/
├── config.py     configuration dataclasses and validation
├── grid.py       buses, lines, links
├── generation.py units, label normalisation, capacity per year
├── demand.py     the four demand methods
├── spatial.py    nearest-bus and NUTS assignment
├── regions.py    resolves the spatial level to one region column
├── aggregate.py  aggregation to the chosen level
├── iamc.py       IAMC conversion and workbook assembly
├── validate.py   the checks
├── manifest.py   run.json, and the guard against overwriting another run
├── figures.py    report charts
├── maps.py       interactive map
├── report.py     HTML report
├── pipeline.py   stage orchestration
└── cli.py        command line interface
```

---

## Data sources

| Input | File | Source |
|---|---|---|
| Buses, lines, links | `data/northsea/grid/` | the curated North Sea network, all bus-matched |
| Generation units | `data/northsea/generation/generation.csv` | 16,446 units carrying `bus_id`, no commissioning dates |
| Demand distribution | `data/northsea/demand/precomputed/load_distribution.csv` | bus-level shares, see the demand section above |
| Industrial sites (NO) | `data/northsea/demand/industry/` | Norwegian industrial and offshore-oil load, scaled on PowerVision national load |
| NUTS geometry | `data/spatial/` | Eurostat NUTS 2024, 1:60M, EPSG:3035, plus the 2021 release for UK regions |

Every path in `config/northsea.yaml` points inside this repository, so nothing
depends on a shared drive. `data/README.md` lists the files and says which raw
inputs were left out.

**This copy is for internal use within the project — please do not
redistribute the data further without checking with Ehsan first.**
