# Flood Monitor & Water Depth Pipeline

One pipeline from raw inputs to water-depth maps: **FABDEM DEM** (Google Earth
Engine) + **GFM flood extent** (EODC STAC) + **FLEXTH water depth/level**
([FLEXTH fork](https://github.com/kwundram2602/FLEXTH), based on
[hyunholee26/FLEXTH](https://github.com/hyunholee26/FLEXTH)) — driven by a
single `config.yaml` and a Streamlit dashboard.

## Quick start

```bash
uv sync

# one-time Google Earth Engine login (needed for the dem step only)
uv run flood-pipeline auth projects/austria_demo/config.yaml

# full pipeline: dem -> gfm -> flexth
uv run flood-pipeline run projects/austria_demo/config.yaml

# subsets
uv run flood-pipeline run projects/austria_demo/config.yaml --steps gfm,flexth

# browse available GFM scenes for the AOI (no auth)
uv run flood-pipeline scenes projects/austria_demo/config.yaml --start 2024-09-01 --end 2024-09-30

# dashboard (project picker, config editor, AOI drawing, scene browser, run monitor, results map)
uv run flood-pipeline dashboard
```

## Pipeline steps

| Step     | Source                                                | Output (defaults, inside the project folder)              |
|----------|-------------------------------------------------------|-----------------------------------------------------------|
| `dem`    | FABDEM mosaic via Google Earth Engine                 | `flood_data/fabdem.tif`                                    |
| `gfm`    | GFM `ensemble_flood_extent` via EODC STAC (public)    | `flood_data/gfm_flood_max.tif` (+ optional `..._sum.tif`)  |
| `flexth` | FLEXTH `pipeline` (resample -> prepare-dtm -> run)    | `preprocessed/{flood,dtm}.tif`, `wl_wd_out/WD_*/WL_*.tif`  |

Each run is a **project folder** holding its own `config.yaml` and all data;
every path in the config resolves relative to that folder:

```text
projects/
└── austria_demo/
    ├── config.yaml         # the project's config (tracked in git)
    ├── aoi.gpkg            # AOI vectors (tracked); drawn AOIs -> aoi_drawn.geojson
    ├── flood_data/         # dem + gfm downloads (gitignored)
    ├── preprocessed/       # flexth intermediates + generated flexth_config.yaml
    └── wl_wd_out/          # final WD_/WL_ rasters
```

New projects: use **Create new project** on the dashboard's Home page (copies
the current config into `<parent>/<name>/config.yaml` — any parent folder,
absolute paths included), or copy a config.yaml into a new folder by hand.
The dashboard discovers projects under `./projects/`; other locations can be
opened by path.

The dem step **skips itself** if its output already exists *and covers the
current AOI* (set `dem.overwrite: true` to force a download; a cached DEM
that doesn't cover the AOI is re-downloaded automatically). With `dem.delivery: drive` the DEM is exported
to Google Drive instead and the pipeline halts with instructions; after placing
the file in `flood_data/` a re-run continues with gfm/flexth.

FLEXTH's `WD_*.tif` is uint16 in **centimeters**, `WL_*.tif` float32 in meters;
both use `999` as the permanent-water sentinel and `0` as nodata.

## config.yaml schema

All paths are relative to the config file's directory (= the project folder);
absolute paths are used as-is. Comments in the file are lost when the
dashboard saves it — this section is the reference.

```yaml
project:
  name: austria_demo          # run label
  gee_project: ee-kwun        # your GEE cloud project (dem step)
  data_dir: flood_data        # dem/gfm outputs
  work_dir: preprocessed      # flexth intermediates + generated flexth_config.yaml
  output_dir: wl_wd_out       # final WD_/WL_ rasters

aoi:
  path: aoi_drawn.geojson     # any vector geopandas reads; the dashboard's AOI
                              # page saves drawings here

dem:
  enabled: true
  delivery: local             # local = direct download | drive = Drive export + halt
  scale: 30                   # metres
  crs: EPSG:4326
  out_name: fabdem.tif
  overwrite: false
  drive_folder: fabdem_exports   # delivery: drive only
  drive_prefix: fabdem

gfm:
  enabled: true
  stac_url: https://stac.eodc.eu/api/v1
  collection: GFM
  band: ensemble_flood_extent
  temporal_extent: ["2024-09-15", "2024-09-20"]
  resolution: 0.0003          # degrees (EPSG:4326)
  max_items: 20
  aggregation: max            # max | sum | both — the temporal max is always
                              # written (it feeds flexth); sum/both add the sum raster
  out_name: gfm_flood_max.tif

flexth:                       # passed through into work_dir/flexth_config.yaml;
  enabled: true               # io + merge sections are generated automatically
  resample:                   # GFM mask -> metric master grid (work_dir/flood.tif)
    enabled: true
    crs: EPSG:32633
    resolution: [30, 30]
    resample_alg: near
    compression: LZW
  prepare_dtm:                # DEM -> flood.tif grid (work_dir/dtm.tif)
    enabled: true
    method: rasterio_gdal
    continuous_input: true
    compression: ZSTD
  flood_processing:           # see FLEXTH docs for the params
    enabled: true
    output_map: WL_WD         # WD | WL | WL_WD
    wl_estimation_method: method_A
    params: { threshold_slope: 0.05, ... }
```

## Development

```bash
uv run pytest          # config round-trip/validation + FLEXTH contract tests
```

Layout: `src/flood_pipeline/` — `config.py` (unified schema),
`runner.py` (orchestration + `##[step:...]` log markers), `cli.py`,
`steps/{dem,gfm,flexth_step}.py`, `app/` (Streamlit dashboard).

## Flood-risk framework (project outline)

- **Hazard**: GFM flood extent + FABDEM DEM -> FLEXTH water depth estimate
- **Exposure**: World Settlement Footprint built-up [0..1], roads and railways
- **Vulnerability**: (open)
- **Ideas / ToDo**: fallbacks for DEM (e.g. EODC Austria DTM)
