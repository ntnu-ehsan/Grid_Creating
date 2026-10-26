# Bundled input data

Everything `config/northsea.yaml` reads is in this folder, so the pipeline runs
straight after cloning — there is no shared drive to mount and no path to edit.
About 5 MB in total, plain CSV and GeoJSON, no Git LFS.

```
data/
├── northsea/
│   ├── grid/
│   │   ├── buses.csv                 2,999 transmission buses (bus_id, voltage, dc, x, y, country)
│   │   ├── lines.csv                 AC lines with s_nom, x, length
│   │   ├── links.csv                 DC links with p_nom
│   │   └── neighbouring_buses.csv    buses outside the study area that interconnectors land on
│   ├── generation/
│   │   └── generation.csv            16,446 units, each already carrying bus_id
│   └── demand/
│       ├── precomputed/
│       │   └── load_distribution.csv bus-level share of national demand - the active input
│       └── industry/                 Norwegian industrial and offshore-oil sites
└── spatial/
    ├── NUTS_RG_60M_2024_3035.geojson Eurostat NUTS 2024, 1:60M, EPSG:3035
    └── NUTS_RG_60M_2021_3035.geojson the 2021 release, used only for UK regions
```

## Why two NUTS files

Eurostat dropped the United Kingdom after the 2021 release, so GB regions are
taken from the older file. `config/northsea.yaml` lists both sources and
restricts the second one to `countries: ["GB"]`.

## What is deliberately not here

`config/northsea.yaml` also configures three raw demand methods — `atlas`,
`population` and `population+industry` — that can rebuild the load
distribution from 1 km rasters instead of reading the finished table. Those
rasters are about 210 MB (JRC Atlas electricity demand for BE, DE, DK, FR, NL
and WorldPop population for GB and NO) and are **not included**, because the
default run does not touch them.

If you switch `demand.method` to one of those, the run stops with a clear
"file not found" naming the missing raster. Ask Ehsan for the files if you
need to re-derive demand rather than use the shipped table.

## Provenance

The grid, generation and demand tables come from the curated North Sea
dataset used in the iDesignRES project, where generation units and load were
already matched to bus ids. The load distribution behind
`load_distribution.csv` was built from Atlas pixels (BE, DE, DK, FR, NL),
WorldPop population (GB) and population plus industrial sites (NO).

## Attribution

These are openly published datasets, redistributed here so the case study
runs out of the box. When reusing them, credit the original providers:

- **JRC Atlas of the European Energy Sector** — 1 km electricity demand
  rasters, behind the Atlas-derived part of the load distribution
- **WorldPop** — 1 km gridded population, behind the GB and NO parts
- **Eurostat** — NUTS 2021 and 2024 administrative geometries
- **ENTSO-E** — transparency statistics used for capacity validation
- **PyPSA-Eur** — the transmission network the North Sea dataset builds on

Check each provider's terms for the exact citation they ask for.
