# Flood Pipeline Workflow
## User Workflow

This diagram summarizes the main execution flow of the project, including optional and conditional branches.

```mermaid
flowchart TD
    A[Start] --> B[Load project config.yaml]
    B --> C[Resolve project paths and AOI]
    C --> D{Command}

    D -->|auth| E[Authenticate Google Earth Engine]
    E --> Z[End]

    D -->|scenes| F[Query EODC STAC for GFM scenes in date range]
    F --> Z

    D -->|run| G[Run selected steps in order]

    G --> H{dem enabled?}
    H -->|no| I
    H -->|yes| H1{Valid DEM already exists and covers AOI?}
    H1 -->|yes| I
    H1 -->|no| H2{dem.delivery}
    H2 -->|local| H3[Download FABDEM from GEE to flood_data/fabdem.tif]
    H2 -->|drive| H4[Export DEM to Google Drive and stop with instructions]
    H4 --> Z
    H3 --> I

    I --> J{gfm enabled?}
    J -->|no| K
    J -->|yes| J1[Fetch GFM flood extent scenes from STAC]
    J1 --> J2[Write per-scene raster and polygonized gpkg]
    J2 --> J3[Build temporal aggregate raster max or sum or both]
    J3 --> K

    K --> L{flexth enabled?}
    L -->|no| M
    L -->|yes| L1[Generate flexth_config.yaml]
    L1 --> L2[Resample flood mask to metric grid]
    L2 --> L3[Prepare DTM aligned to flood grid]
    L3 --> L4[Run FLEXTH and write WD and WL outputs]
    L4 --> M

    M --> N{population enabled?}
    N -->|no| O[Pipeline complete]
    N -->|yes| N1[Download WorldPop 2020 from GEE]
    N1 --> O

    O --> Z[End]
```

## Developer workflow (code-level)

This chart shows how the code executes the pipeline internally, from CLI entrypoint to step modules and artifacts.

```mermaid
flowchart TD
    A[cli.py: run_cmd] --> B[config.py: load_config]
    B --> C[runner.py: run_pipeline]
    C --> D[validate cfg]
    D --> E[normalize_steps]

    E --> F{for step in dem gfm flexth population}
    F --> G{step enabled in cfg?}
    G -->|no| F
    G -->|yes| H[_step_runner imports module lazily]
    H --> I[module.run cfg log]

    I --> J{Step name}

    J -->|dem| D1[steps/dem.py run]
    D1 --> D2[load AOI via geopandas]
    D2 --> D3{DEM exists and covers AOI?}
    D3 -->|yes| D4[return StepOutcome outputs]
    D3 -->|no| D5[gee.init_gee + build_fabdem_image]
    D5 --> D6{delivery mode}
    D6 -->|local| D7[geemap.download_ee_image]
    D6 -->|drive| D8[ee Export.image.toDrive]
    D7 --> D9[write flood_data/fabdem.tif]
    D8 --> D10[return StepOutcome halt true]

    J -->|gfm| G1[steps/gfm.py run]
    G1 --> G2[search_gfm_items via pystac_client]
    G2 --> G3[load_flood_cube via odc.stac.load]
    G3 --> G4[write temporal max raster]
    G4 --> G5[clear stale per-scene rasters]
    G5 --> G6[for each timestamped scene]
    G6 --> G7[write gfm_flood_stamp.tif]
    G7 --> G8[polygonize.write_flood_polygons]
    G8 --> G9[optional sum raster]

    J -->|flexth| F1[steps/flexth_step.py run]
    F1 --> F2[discover scene rasters in data_dir]
    F2 --> F3[for each scene stamp]
    F3 --> F4[build_flexth_config]
    F4 --> F5[write preprocessed stamp/flexth_config.yaml]
    F5 --> F6[subprocess python -m flexth.cli pipeline]
    F6 --> F7[collect WD and WL outputs]
    F7 --> F8[stamp filenames with scene id]

    J -->|population| P1[steps/population.py run]
    P1 --> P2[load AOI + covers_aoi check]
    P2 --> P3[gee.init_gee + build_population_image]
    P3 --> P4[geemap.download_ee_image]
    P4 --> P5[write flood_data/worldpop_2020.tif]

    D4 --> K[runner logs step done]
    D9 --> K
    D10 --> L[pipeline halted cleanly]
    G9 --> K
    F8 --> K
    P5 --> K
    K --> F
    F --> M[runner logs pipeline finished]
```
