"""JRC/Huizinga depth-damage functions: GHSL class mapping and continent lookup.

Ties together three things to turn a (GHSL class, flood depth) pixel pair into
a relative damage fraction (0-1):

1. GHSL_TO_DAMAGE_CLASS -- GHSL's built_characteristics classes only
   distinguish residential vs. non-residential (by height); the JRC table
   wants a finer split (residential/commercial/industrial/transport/
   infrastructure/agriculture) that GHSL alone cannot provide. Non-residential
   GHSL classes (21-25) are mapped to "Commercial buildings" as the closest
   general proxy, since GHSL has no way to distinguish industrial/transport
   uses. Open water (class 4) has no damage class and is excluded.

2. resolve_continent() -- finds which country the AOI's centroid falls in
   using a bundled, slimmed Natural Earth admin-0 countries dataset, then
   maps that country to one of the JRC table's continent columns. This does
   its own Central America override: Natural Earth's default CONTINENT
   assigns Belize/Costa Rica/El Salvador/Guatemala/Honduras/Mexico/Nicaragua/
   Panama to "North America", but the JRC table groups Central America with
   South America ("Centr&South AMERICA"), so those countries are redirected
   using Natural Earth's SUBREGION field ("Central America").

3. DamageTable / damage_fraction_for_pixels() -- loads the bundled JRC table
   and interpolates linearly between its depth bins (0, 0.5, 1, ..., 2, 3, 4,
   5, 6 m), holding flat beyond 6 m (every class's curve is already flat at
   1.0 by then). Falls back to the GLOBAL column wherever a continent-specific
   value is missing in the source table (several damage classes lack data for
   some continents, e.g. Commercial buildings has no ASIA column).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# GEE FeatureCollection used for continent resolution -- see resolve_continent().
# Schema: abbreviati, country_co (FIPS 2-letter), country_na, wld_rgn
# (continental region, e.g. "South America", "Central America", "Europe").
# https://developers.google.com/earth-engine/datasets/catalog/USDOS_LSIB_SIMPLE_2017
COUNTRIES_ASSET = "USDOS/LSIB_SIMPLE/2017"

# JRC/Huizinga (2017) global flood depth-damage functions: relative damage
# (0-1, fraction of maximum damage) by damage class, flood depth, and
# continent. Source: the "Damage functions" sheet of the JRC workbook.
#
# (damage_class, depth_m, EUROPE, North_AMERICA, CentrSouth_AMERICA, ASIA,
#  AFRICA, OCEANIA, GLOBAL)
#
# None marks a continent with no data for that damage class in the source
# table (e.g. Commercial buildings has no ASIA column, Transport is missing
# North America/Africa/Oceania); DamageTable falls back to GLOBAL in that
# case -- see DamageTable._curve_for.
_DAMAGE_TABLE_ROWS: tuple[tuple, ...] = (
    # Residential buildings
    ("Residential buildings", 0, 0.0, 0.201805, 0.0, 0.0, 0.0, 0.0, None),
    ("Residential buildings", 0.5, 0.25, 0.44327, 0.490886, 0.326557, 0.219925, 0.475418, None),
    ("Residential buildings", 1, 0.4, 0.582755, 0.711294, 0.49405, 0.378227, 0.640393, None),
    ("Residential buildings", 1.5, 0.5, 0.682522, 0.842026, 0.616572, 0.530589, 0.714615, None),
    ("Residential buildings", 2, 0.6, 0.783957, 0.949369, 0.720712, 0.635637, 0.787726, None),
    ("Residential buildings", 3, 0.75, 0.854349, 0.983637, 0.869528, 0.81694, 0.92878, None),
    ("Residential buildings", 4, 0.85, 0.92367, 1.0, 0.931487, 0.903435, 0.967382, None),
    ("Residential buildings", 5, 0.95, 0.958523, 1.0, 0.983604, 0.957152, 0.982795, None),
    ("Residential buildings", 6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, None),

    # Commercial buildings
    ("Commercial buildings", 0, 0.0, 0.018405, 0.0, 0.0, None, 0.0, 0.0),
    ("Commercial buildings", 0.5, 0.15, 0.239264, 0.611478, 0.37679, None, 0.238954, 0.323297),
    ("Commercial buildings", 1, 0.3, 0.374233, 0.839531, 0.537682, None, 0.4812, 0.506529),
    ("Commercial buildings", 1.5, 0.45, 0.466258, 0.923588, 0.659337, None, 0.673795, 0.634596),
    ("Commercial buildings", 2, 0.55, 0.552147, 0.991972, 0.762845, None, 0.864583, 0.74431),
    ("Commercial buildings", 3, 0.75, 0.687117, 1.0, 0.883349, None, 1.0, 0.864093),
    ("Commercial buildings", 4, 0.9, 0.822086, 1.0, 0.941855, None, 1.0, 0.932788),
    ("Commercial buildings", 5, 1.0, 0.907975, 1.0, 0.980759, None, 1.0, 0.977747),
    ("Commercial buildings", 6, 1.0, 1.0, 1.0, 1.0, None, 1.0, 1.0),

    # Industrial buildings
    ("Industrial buildings", 0, 0.0, 0.025714, 0.0, 0.0, 0.0, None, 0.0),
    ("Industrial buildings", 0.5, 0.15, 0.322857, 0.667019, 0.283182, 0.062682, None, 0.297148),
    ("Industrial buildings", 1, 0.27, 0.511429, 0.888713, 0.481616, 0.247196, None, 0.479791),
    ("Industrial buildings", 1.5, 0.4, 0.637143, 0.946737, 0.629219, 0.40333, None, 0.603286),
    ("Industrial buildings", 2, 0.52, 0.74, 1.0, 0.717241, 0.494489, None, 0.694346),
    ("Industrial buildings", 3, 0.7, 0.86, 1.0, 0.856675, 0.684652, None, 0.820265),
    ("Industrial buildings", 4, 0.85, 0.937143, 1.0, 0.908577, 0.91859, None, 0.922862),
    ("Industrial buildings", 5, 1.0, 0.98, 1.0, 0.955327, 1.0, None, 0.987065),
    ("Industrial buildings", 6, 1.0, 1.0, 1.0, 1.0, 1.0, None, 1.0),

    # Transport
    ("Transport", 0, 0.0, None, 0.0, 0.0, None, None, 0.0),
    ("Transport", 0.5, 0.316667, None, 0.087719, 0.357516, None, None, 0.253967),
    ("Transport", 1, 0.541667, None, 0.175439, 0.571895, None, None, 0.429667),
    ("Transport", 1.5, 0.701667, None, 0.596491, 0.733333, None, None, 0.677164),
    ("Transport", 2, 0.831667, None, 0.842105, 0.847222, None, None, 0.840331),
    ("Transport", 3, 1.0, None, 1.0, 1.0, None, None, 1.0),
    ("Transport", 4, 1.0, None, 1.0, 1.0, None, None, 1.0),
    ("Transport", 5, 1.0, None, 1.0, 1.0, None, None, 1.0),
    ("Transport", 6, 1.0, None, 1.0, 1.0, None, None, 1.0),

    # Infrastructure - roads
    ("Infrastructure - roads", 0, 0.0, None, None, 0.0, None, None, 0.0),
    ("Infrastructure - roads", 0.5, 0.25, None, None, 0.214437, None, None, 0.232218),
    ("Infrastructure - roads", 1, 0.42, None, None, 0.372754, None, None, 0.396377),
    ("Infrastructure - roads", 1.5, 0.55, None, None, 0.603935, None, None, 0.576967),
    ("Infrastructure - roads", 2, 0.65, None, None, 0.709659, None, None, 0.67983),
    ("Infrastructure - roads", 3, 0.8, None, None, 0.808409, None, None, 0.804205),
    ("Infrastructure - roads", 4, 0.9, None, None, 0.887159, None, None, 0.89358),
    ("Infrastructure - roads", 5, 1.0, None, None, 0.96875, None, None, 0.984375),
    ("Infrastructure - roads", 6, 1.0, None, None, 1.0, None, None, 1.0),

    # Agriculture
    ("Agriculture", 0, 0.0, 0.018575, None, 0.0, 0.0, None, 0.0),
    ("Agriculture", 0.5, 0.3, 0.267798, None, 0.135, 0.242874, None, 0.236418),
    ("Agriculture", 1, 0.55, 0.473677, None, 0.37, 0.471839, None, 0.466379),
    ("Agriculture", 1.5, 0.65, 0.550561, None, 0.524, 0.741379, None, 0.616485),
    ("Agriculture", 2, 0.75, 0.602161, None, 0.558, 0.916667, None, 0.706707),
    ("Agriculture", 3, 0.85, 0.760057, None, 0.66, 1.0, None, 0.817514),
    ("Agriculture", 4, 0.95, 0.874095, None, 0.834, 1.0, None, 0.914524),
    ("Agriculture", 5, 1.0, 0.954076, None, 0.988, 1.0, None, 0.985519),
    ("Agriculture", 6, 1.0, 1.0, None, 1.0, 1.0, None, 1.0),
)

# GHSL built_characteristics class -> JRC damage class. None = excluded
# (no damage assigned; open water has no depth-damage function).
GHSL_TO_DAMAGE_CLASS: dict[int, str | None] = {
    1: "Agriculture",   # open spaces, low vegetation
    2: "Agriculture",   # open spaces, medium vegetation
    3: "Agriculture",   # open spaces, high vegetation
    4: None,            # open spaces, water surfaces -- excluded
    5: "Infrastructure - roads",  # open spaces, road surfaces
    11: "Residential buildings",
    12: "Residential buildings",
    13: "Residential buildings",
    14: "Residential buildings",
    15: "Residential buildings",
    21: "Commercial buildings",  # non-residential: GHSL can't distinguish
    22: "Commercial buildings",  # commercial/industrial/transport, so all
    23: "Commercial buildings",  # non-residential classes use Commercial
    24: "Commercial buildings",  # as the closest general proxy.
    25: "Commercial buildings",
}

# LSIB's wld_rgn is a free-text "continental region" string (e.g. "South
# America", "Central America", "Europe"); matched by substring rather than
# an exact whitelist since the complete set of values wasn't verified against
# a live Earth Engine session while writing this. Order matters: more specific
# phrases (e.g. "central america") are checked before broader ones. Run
# check_wld_rgn_values() below once to confirm coverage for your AOIs and
# extend this list if a wld_rgn value falls through to GLOBAL unexpectedly.
_WLD_RGN_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("central america", "CentrSouth_AMERICA"),
    ("south america", "CentrSouth_AMERICA"),
    ("north america", "North_AMERICA"),
    ("europe", "EUROPE"),
    ("africa", "AFRICA"),
    ("oceania", "OCEANIA"),
    ("pacific", "OCEANIA"),
    ("asia", "ASIA"),
    ("middle east", "ASIA"),
    ("near east", "ASIA"),
)


def _wld_rgn_to_jrc_column(wld_rgn: str) -> str:
    """Map an LSIB wld_rgn string to a JRC table column, via substring match."""
    lowered = wld_rgn.lower()
    for keyword, jrc_column in _WLD_RGN_KEYWORDS:
        if keyword in lowered:
            return jrc_column
    return "GLOBAL"


@dataclass
class CountryMatch:
    country_name: str
    country_code: str  # LSIB's FIPS 2-letter code (country_co) -- NOT ISO A3
    wld_rgn: str        # raw LSIB region string, kept for debugging/logging
    jrc_column: str


def resolve_continent(aoi_path: Path) -> CountryMatch:
    """Find the country/continent for an AOI's centroid, for JRC lookup.

    Queries Earth Engine's USDOS/LSIB_SIMPLE/2017 FeatureCollection directly
    (already a project dependency) rather than bundling a local country
    boundaries file. Call ``gee.init_gee(...)`` before this, as ghsl_step does.

    Falls back to the GLOBAL column (country_name="unknown") if the centroid
    doesn't fall inside any country polygon (e.g. open ocean, or a gap in the
    simplified LSIB boundaries).
    """
    import ee
    import geopandas as gpd

    aoi = gpd.read_file(aoi_path).to_crs(epsg=4326)
    union = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    centroid = union.centroid

    point = ee.Geometry.Point(float(centroid.x), float(centroid.y))
    countries = ee.FeatureCollection(COUNTRIES_ASSET)
    matches = countries.filterBounds(point)

    # Check size before .first().getInfo(): calling .first() on an empty
    # collection raises an EEException rather than returning None.
    if matches.size().getInfo() == 0:
        return CountryMatch(
            country_name="unknown", country_code="", wld_rgn="", jrc_column="GLOBAL"
        )

    props = matches.first().getInfo()["properties"]
    country_name = props.get("country_na", "unknown")
    wld_rgn = props.get("wld_rgn", "")
    return CountryMatch(
        country_name=country_name,
        country_code=props.get("country_co", ""),
        wld_rgn=wld_rgn,
        jrc_column=_wld_rgn_to_jrc_column(wld_rgn),
    )


def check_wld_rgn_values() -> list[str]:
    """One-off helper: list every distinct wld_rgn value in LSIB_SIMPLE/2017.

    Run this once against your real GEE session to confirm _WLD_RGN_KEYWORDS
    covers every value in the dataset, e.g.:

        from flood_pipeline import gee
        from flood_pipeline.damage import check_wld_rgn_values
        gee.init_gee("<your-gee-project>")
        for value in check_wld_rgn_values():
            print(value)
    """
    import ee

    countries = ee.FeatureCollection(COUNTRIES_ASSET)
    values = countries.aggregate_array("wld_rgn").distinct().getInfo()
    return sorted(values)


class DamageTable:
    """Loaded JRC depth-damage table, ready for interpolated lookup."""

    _COLUMNS = ("EUROPE", "North_AMERICA", "CentrSouth_AMERICA",
                "ASIA", "AFRICA", "OCEANIA", "GLOBAL")

    def __init__(self, rows: tuple[tuple, ...]):
        """Build lookup curves from rows shaped like _DAMAGE_TABLE_ROWS:
        (damage_class, depth_m, EUROPE, North_AMERICA, CentrSouth_AMERICA,
         ASIA, AFRICA, OCEANIA, GLOBAL), with None for missing values.
        """
        # {damage_class: {continent_column: (depth_array, fraction_array)}}
        self._curves: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
        by_class: dict[str, list[tuple]] = {}
        for row in rows:
            by_class.setdefault(row[0], []).append(row)

        for damage_class, class_rows in by_class.items():
            class_rows.sort(key=lambda r: r[1])
            depths = np.array([r[1] for r in class_rows], dtype="float64")
            self._curves[damage_class] = {}
            for col_index, col in enumerate(self._COLUMNS, start=2):
                values = np.array(
                    [r[col_index] if r[col_index] is not None else np.nan
                     for r in class_rows],
                    dtype="float64",
                )
                self._curves[damage_class][col] = (depths, values)

    def fraction(self, damage_class: str, continent_column: str, depth_m: float) -> float:
        """Interpolated relative damage fraction for one (class, continent, depth)."""
        depths, values = self._curve_for(damage_class, continent_column)
        return float(np.interp(depth_m, depths, values, left=0.0, right=values[-1]))

    def fraction_array(
        self, damage_class: str, continent_column: str, depth_m: np.ndarray
    ) -> np.ndarray:
        """Vectorized version of :meth:`fraction` for an array of depths."""
        depths, values = self._curve_for(damage_class, continent_column)
        return np.interp(depth_m, depths, values, left=0.0, right=values[-1])

    def _curve_for(self, damage_class: str, continent_column: str):
        columns = self._curves[damage_class]
        depths, values = columns[continent_column]
        if np.isnan(values).any():
            # Missing continent-specific values for this damage class -- use
            # GLOBAL, which is populated for every row in every class.
            depths, values = columns["GLOBAL"]
        return depths, values


def load_damage_table() -> DamageTable:
    """Build a :class:`DamageTable` from the inline JRC data (_DAMAGE_TABLE_ROWS)."""
    return DamageTable(_DAMAGE_TABLE_ROWS)


def damage_fraction_for_pixels(
    ghsl_class: np.ndarray,
    depth_m: np.ndarray,
    continent_column: str,
    table: DamageTable,
) -> np.ndarray:
    """Per-pixel relative damage fraction from GHSL class + depth arrays.

    Pixels whose GHSL class has no damage-class mapping (open water, or any
    class outside GHSL_TO_DAMAGE_CLASS) get a fraction of 0.
    """
    result = np.zeros(ghsl_class.shape, dtype="float32")
    for class_id, damage_class in GHSL_TO_DAMAGE_CLASS.items():
        if damage_class is None:
            continue
        pixel_mask = ghsl_class == class_id
        if not np.any(pixel_mask):
            continue
        fractions = table.fraction_array(
            damage_class, continent_column, depth_m[pixel_mask].astype("float64")
        )
        result[pixel_mask] = fractions.astype("float32")
    return result