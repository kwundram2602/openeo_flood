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
   some continents, e.g. Commercial buildings has no AFRICA column).

   One damage class -- "Residential buildings" -- has NO GLOBAL column at all
   in the source table (every depth bin is None there), even though its six
   continent-specific columns are all fully populated. That breaks the usual
   fallback whenever ``resolve_continent()`` itself returns "GLOBAL" (an AOI
   whose centroid matches no LSIB country, or whose wld_rgn string isn't
   covered by _WLD_RGN_KEYWORDS): naively "falling back" to GLOBAL just
   reselects the same all-missing column. DamageTable._curve_for handles this
   one level deeper: if a class's own GLOBAL column is itself missing, it
   synthesizes a stand-in GLOBAL curve as the unweighted average of that
   class's available continent curves, and warns once. This is a reasonable
   emergency fallback, not the real JRC-published GLOBAL figure -- replace it
   with the actual value from the source Huizinga workbook if you have access
   to it.

4. MaxDamageTable / load_max_damage_table() -- a *separate* per-country
   lookup (not per-continent, unlike DamageTable above) for absolute maximum
   damage value in EUR/m^2, from the JRC MaxDamage companion workbook.
   Residential and Commercial buildings only -- agriculture and
   infrastructure/transport use a different unit basis (per hectare, or a
   GDP-proportional formula) and are out of scope. Multiplying a MaxDamage
   value by DamageTable's relative damage fraction and a building footprint
   area gives an absolute EUR damage estimate; see ghsl_step.py's Band 4.
   Values are 2010 prices, not inflation-adjusted to the present day.

   Country matching for MaxDamageTable has the same kind of uncertainty as
   _WLD_RGN_KEYWORDS above, for a different reason: resolve_continent()
   returns an LSIB country name, but this workbook keys its rows by World
   Bank-style names (e.g. "Korea, Rep.", "Egypt, Arab Rep.", "Cote
   d'Ivoire"), which often don't match LSIB's wording verbatim even after
   normalizing case/punctuation/accents. MaxDamageTable.lookup() also
   consults _COUNTRY_NAME_ALIASES for the names most likely to differ in
   wording; that list is a best-effort guess, not verified against a live
   Earth Engine session. Run check_country_name_matches() once against your
   real GEE session and extend _COUNTRY_NAME_ALIASES for anything that falls
   through to the World fallback unexpectedly -- the same pattern
   check_wld_rgn_values() already establishes above.
"""

from __future__ import annotations

import re
import unicodedata
import warnings
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
# table (e.g. Commercial buildings has no AFRICA column, Transport is missing
# North America/Africa/Oceania); DamageTable falls back to GLOBAL in that
# case -- see DamageTable._curve_for. "Residential buildings" is a special
# case: its GLOBAL column is None for every depth bin too -- see the module
# docstring and DamageTable._curve_for.
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
        # Classes we've already warned about needing a synthesized GLOBAL
        # curve (see _curve_for), so repeated per-pixel/per-scene lookups
        # don't spam the log.
        self._warned_classes: set[str] = set()
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
            depths, global_values = columns["GLOBAL"]
            if np.isnan(global_values).any():
                # This class's own GLOBAL column is missing too (currently
                # only "Residential buildings") -- naively "falling back" to
                # it would just reselect the same all-NaN array. Synthesize a
                # stand-in GLOBAL curve instead: the unweighted average of
                # whichever continent curves this class does have. Not the
                # real JRC-published GLOBAL figure -- see the module
                # docstring.
                if damage_class not in self._warned_classes:
                    warnings.warn(
                        f"JRC damage table has no GLOBAL curve for "
                        f"{damage_class!r}; using the average of its "
                        "available continent curves as a fallback.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    self._warned_classes.add(damage_class)
                available = [
                    columns[col][1]
                    for col in self._COLUMNS
                    if col != "GLOBAL" and not np.isnan(columns[col][1]).any()
                ]
                if not available:
                    # No continent data at all for this class -- nothing
                    # sensible to average; surface it plainly rather than
                    # quietly returning NaN.
                    raise ValueError(
                        f"JRC damage table has no usable data (GLOBAL or "
                        f"continent-specific) for damage class {damage_class!r}"
                    )
                values = np.mean(np.stack(available), axis=0)
            else:
                values = global_values
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


# ============================================================================
# MaxDamage: per-country absolute maximum damage (EUR/m^2), residential and
# commercial buildings only. See section 4 of the module docstring.
# ============================================================================

# (country name, ISO A3, Residential total EUR/m2 2010, Commercial total EUR/m2 2010)
# "Building based, Total" column of MaxDamage-Residential / MaxDamage-Commercial.
# 0.0 entries are small territories the source workbook has no data for.
_MAX_DAMAGE_ROWS: tuple[tuple[str, str, float, float], ...] = (
    ('Afghanistan', 'AFG', 148.93, 232.67),
    ('Albania', 'ALB', 322.62, 476.7),
    ('Algeria', 'DZA', 327.75, 483.72),
    ('American Samoa', 'ASM', 0, 0),
    ('Andorra', 'AND', 0, 0),
    ('Angola', 'AGO', 323.91, 478.47),
    ('Antigua and Barbuda', 'ATG', 499.95, 715.72),
    ('Argentina', 'ARG', 476.01, 683.87),
    ('Armenia', 'ARM', 288.55, 429.8),
    ('Aruba', 'ABW', 635.73, 894.46),
    ('Australia', 'AUS', 851.09, 1172.52),
    ('Austria', 'AUT', 816.05, 1127.65),
    ('Azerbaijan', 'AZE', 367.21, 537.53),
    ('Bahamas, The', 'BHS', 611.33, 862.55),
    ('Bahrain', 'BHR', 596.04, 842.52),
    ('Bangladesh', 'BGD', 167.62, 259.65),
    ('Barbados', 'BRB', 538.85, 767.25),
    ('Belarus', 'BLR', 366.63, 536.74),
    ('Belgium', 'BEL', 801.73, 1109.28),
    ('Belize', 'BLZ', 332.85, 490.7),
    ('Benin', 'BEN', 161.26, 250.51),
    ('Bermuda', 'BMU', 1044.77, 1418.2),
    ('Bhutan', 'BTN', 252.57, 379.83),
    ('Bolivia', 'BOL', 239.89, 362.12),
    ('Bosnia and Herzegovina', 'BIH', 328.65, 484.95),
    ('Botswana', 'BWA', 393.26, 572.82),
    ('Brazil', 'BRA', 468.2, 673.44),
    ('Brunei Darussalam', 'BRN', 697.33, 974.6),
    ('Bulgaria', 'BGR', 384.43, 560.88),
    ('Burkina Faso', 'BFA', 152.08, 237.25),
    ('Burundi', 'BDI', 103.74, 166.37),
    ('Cabo Verde', 'CPV', 298.53, 443.58),
    ('Cambodia', 'KHM', 169.28, 262.04),
    ('Cameroon', 'CMR', 196.03, 300.25),
    ('Canada', 'CAN', 822.91, 1136.45),
    ('Cayman Islands', 'CYM', 0, 0),
    ('Central African Republic', 'CAF', 137.55, 216.13),
    ('Chad', 'TCD', 179.35, 276.48),
    ('Channel Islands', 'CHA', 0, 0),
    ('Chile', 'CHL', 494.95, 709.07),
    ('China', 'CHN', 330.17, 487.03),
    ('Colombia', 'COL', 375.23, 548.41),
    ('Comoros', 'COM', 167.11, 258.92),
    ('Congo, Dem. Rep.', 'COD', 123.71, 195.88),
    ('Congo, Rep.', 'COG', 281.13, 419.53),
    ('Costa Rica', 'CRI', 409.9, 595.27),
    ("Cote d'Ivoire", 'CIV', 206.52, 315.13),
    ('Croatia', 'HRV', 507.02, 725.11),
    ('Cuba', 'CUB', 363.78, 532.87),
    ('Curacao', 'CUW', 0, 0),
    ('Cyprus', 'CYP', 670.49, 939.75),
    ('Czech Republic', 'CZE', 587.2, 830.92),
    ('Denmark', 'DNK', 886.89, 1218.2),
    ('Djibouti', 'DJI', 209.03, 318.69),
    ('Dominica', 'DMA', 392.09, 571.24),
    ('Dominican Republic', 'DOM', 353.56, 518.96),
    ('Ecuador', 'ECU', 335.92, 494.9),
    ('Egypt, Arab Rep.', 'EGY', 276.74, 413.45),
    ('El Salvador', 'SLV', 299.58, 445.02),
    ('Equatorial Guinea', 'GNQ', 549.52, 781.33),
    ('Eritrea', 'ERI', 126.68, 200.25),
    ('Estonia', 'EST', 522.95, 746.22),
    ('Ethiopia', 'ETH', 123.3, 195.27),
    ('Faeroe Islands', 'FRO', 0, 0),
    ('Fiji', 'FJI', 306.32, 454.31),
    ('Finland', 'FIN', 814.41, 1125.55),
    ('France', 'FRA', 775.63, 1075.74),
    ('French Polynesia', 'PYF', 0, 0),
    ('Gabon', 'GAB', 440.34, 636.18),
    ('Gambia, The', 'GMB', 149.45, 233.44),
    ('Georgia', 'GEO', 269.37, 403.22),
    ('Germany', 'DEU', 783.04, 1085.27),
    ('Ghana', 'GHA', 207.41, 316.39),
    ('Greece', 'GRC', 660.87, 927.23),
    ('Greenland', 'GRL', 0, 0),
    ('Grenada', 'GRD', 401.48, 583.92),
    ('Guam', 'GUM', 0, 0),
    ('Guatemala', 'GTM', 279.71, 417.57),
    ('Guinea', 'GIN', 135.06, 212.51),
    ('Guinea-Bissau', 'GNB', 146.12, 228.6),
    ('Guyana', 'GUY', 279.4, 417.13),
    ('Haiti', 'HTI', 159.37, 247.78),
    ('Honduras', 'HND', 246.6, 371.51),
    ('Hong Kong SAR, China', 'HKG', 711.62, 993.12),
    ('Hungary', 'HUN', 499.08, 714.56),
    ('Iceland', 'ISL', 782.84, 1085.02),
    ('India', 'IND', 212.78, 323.98),
    ('Indonesia', 'IDN', 282.1, 420.87),
    ('Iran, Islamic Rep.', 'IRN', 363.11, 531.96),
    ('Iraq', 'IRQ', 331.32, 488.61),
    ('Ireland', 'IRL', 825.81, 1140.17),
    ('Isle of Man', 'IMN', 0, 0),
    ('Israel', 'ISR', 694.46, 970.87),
    ('Italy', 'ITA', 738.79, 1028.25),
    ('Jamaica', 'JAM', 343.6, 505.39),
    ('Japan', 'JPN', 793.02, 1098.1),
    ('Jordan', 'JOR', 328.36, 484.56),
    ('Kazakhstan', 'KAZ', 435.01, 629.03),
    ('Kenya', 'KEN', 184.44, 283.74),
    ('Kiribati', 'KIR', 219.66, 333.69),
    ('Korea, Dem. Rep.', 'PRK', 233.32, 352.9),
    ('Korea, Rep.', 'KOR', 613.57, 865.48),
    ('Kosovo', 'KOS', 294.11, 437.47),
    ('Kuwait', 'KWT', 759.8, 1055.35),
    ('Kyrgyz Republic', 'KGZ', 177.11, 273.26),
    ('Lao PDR', 'LAO', 194.54, 298.13),
    ('Latvia', 'LVA', 475.79, 683.57),
    ('Lebanon', 'LBN', 429.13, 621.14),
    ('Lesotho', 'LSO', 191.85, 294.3),
    ('Liberia', 'LBR', 120.9, 191.75),
    ('Libya', 'LBY', 490.31, 702.9),
    ('Liechtenstein', 'LIE', 0, 0),
    ('Lithuania', 'LTU', 482.22, 692.13),
    ('Luxembourg', 'LUX', 1108.48, 1498.26),
    ('Macao SAR, China', 'MAC', 858.91, 1182.51),
    ('Macedonia, FYR', 'MKD', 330.42, 487.38),
    ('Madagascar', 'MDG', 132.48, 208.73),
    ('Malawi', 'MWI', 125.46, 198.45),
    ('Malaysia', 'MYS', 429.1, 621.1),
    ('Maldives', 'MDV', 383.79, 560.01),
    ('Mali', 'MLI', 159.79, 248.37),
    ('Malta', 'MLT', 586.41, 829.89),
    ('Marshall Islands', 'MHL', 288.61, 429.88),
    ('Mauritania', 'MRT', 184.39, 283.68),
    ('Mauritius', 'MUS', 406.08, 590.13),
    ('Mexico', 'MEX', 432.22, 625.3),
    ('Micronesia, Fed. Sts.', 'FSM', 278.06, 415.28),
    ('Moldova', 'MDA', 224.65, 340.72),
    ('Monaco', 'MCO', 1266.01, 1694.86),
    ('Mongolia', 'MNG', 255.8, 384.35),
    ('Montenegro', 'MNE', 385.67, 562.56),
    ('Morocco', 'MAR', 277.47, 414.46),
    ('Mozambique', 'MOZ', 133.7, 210.52),
    ('Myanmar', 'MMR', 203.99, 311.54),
    ('Namibia', 'NAM', 350.51, 514.81),
    ('Nepal', 'NPL', 152.4, 237.7),
    ('Netherlands', 'NLD', 841.75, 1160.57),
    ('New Caledonia', 'NCL', 0, 0),
    ('New Zealand', 'NZL', 714.11, 996.34),
    ('Nicaragua', 'NIC', 219.44, 333.39),
    ('Niger', 'NER', 125.49, 198.5),
    ('Nigeria', 'NGA', 256.89, 385.86),
    ('Northern Mariana Islands', 'MNP', 0, 0),
    ('Norway', 'NOR', 1035.07, 1405.97),
    ('Oman', 'OMN', 600.23, 848.01),
    ('Pakistan', 'PAK', 187.69, 288.39),
    ('Palau', 'PLW', 445.53, 643.14),
    ('Panama', 'PAN', 411.13, 596.93),
    ('Papua New Guinea', 'PNG', 210.74, 321.11),
    ('Paraguay', 'PRY', 287.69, 428.62),
    ('Peru', 'PER', 347.83, 511.15),
    ('Philippines', 'PHL', 249.21, 375.15),
    ('Poland', 'POL', 491.96, 705.1),
    ('Portugal', 'PRT', 617.68, 870.86),
    ('Puerto Rico', 'PRI', 656.84, 921.97),
    ('Qatar', 'QAT', 963.64, 1315.73),
    ('Romania', 'ROU', 417.22, 605.14),
    ('Russian Federation', 'RUS', 463.75, 667.51),
    ('Rwanda', 'RWA', 145.24, 227.33),
    ('Samoa', 'WSM', 299.99, 445.59),
    ('San Marino', 'SMR', 0, 0),
    ('Sao Tome and Principe', 'STP', 194.88, 298.61),
    ('Saudi Arabia', 'SAU', 582.16, 824.3),
    ('Senegal', 'SEN', 185.94, 285.89),
    ('Serbia', 'SCG', 356.21, 522.58),
    ('Seychelles', 'SYC', 465.96, 670.46),
    ('Sierra Leone', 'SLE', 136.57, 214.71),
    ('Singapore', 'SGP', 816.9, 1128.74),
    ('Sint Maarten (Dutch part)', 'SXM', 0, 0),
    ('Slovak Republic', 'SVK', 547.88, 779.18),
    ('Slovenia', 'SVN', 626.85, 882.85),
    ('Solomon Islands', 'SLB', 205.5, 313.69),
    ('Somalia', 'SOM', 114.56, 182.4),
    ('South Africa', 'ZAF', 397.46, 578.49),
    ('South Sudan', 'SSD', 222.01, 337.0),
    ('Spain', 'ESP', 696.07, 972.97),
    ('Sri Lanka', 'LKA', 260.66, 391.11),
    ('St. Kitts and Nevis', 'KNA', 503.04, 719.82),
    ('St. Lucia', 'LCA', 393.99, 573.81),
    ('St. Martin (French part)', 'MAF', 0, 0),
    ('St. Vincent and the Grenadines', 'VCT', 376.44, 550.06),
    ('Sudan', 'SDN', 214.07, 325.81),
    ('Suriname', 'SUR', 420.8, 609.95),
    ('Swaziland', 'SWZ', 293.35, 436.43),
    ('Sweden', 'SWE', 852.83, 1174.74),
    ('Switzerland', 'CHE', 977.84, 1333.7),
    ('Syrian Arab Republic', 'SYR', 348.47, 512.04),
    ('Tajikistan', 'TJK', 165.65, 256.82),
    ('Tanzania', 'TZA', 145.12, 227.15),
    ('Thailand', 'THA', 340.5, 501.16),
    ('Timor-Leste', 'TLS', 176.78, 272.79),
    ('Togo', 'TGO', 142.79, 223.77),
    ('Tonga', 'TON', 302.98, 449.7),
    ('Trinidad and Tobago', 'TTO', 536.45, 764.07),
    ('Tunisia', 'TUN', 323.7, 478.17),
    ('Turkey', 'TUR', 454.01, 654.5),
    ('Turkmenistan', 'TKM', 329.0, 485.43),
    ('Turks and Caicos Islands', 'TCA', 0, 0),
    ('Tuvalu', 'TUV', 292.55, 435.32),
    ('Uganda', 'UGA', 139.28, 218.66),
    ('Ukraine', 'UKR', 283.1, 422.27),
    ('United Arab Emirates', 'ARE', 722.73, 1007.5),
    ('United Kingdom', 'GBR', 758.12, 1053.19),
    ('United States', 'USA', 828.97, 1144.21),
    ('Uruguay', 'URY', 477.13, 685.36),
    ('Uzbekistan', 'UZB', 210.45, 320.68),
    ('Vanuatu', 'VUT', 282.8, 421.85),
    ('Venezuela, RB', 'VEN', 507.87, 726.23),
    ('Vietnam', 'VNM', 207.86, 317.03),
    ('Virgin Islands (U.S.)', 'VIR', 0, 0),
    ('West Bank and Gaza', 'WBK', 258.07, 387.51),
    ('Yemen, Rep.', 'YEM', 211.47, 322.13),
    ('Zambia', 'ZMB', 219.34, 333.24),
    ('Zimbabwe', 'ZWE', 164.21, 254.75),
)

# The workbook's own "World" aggregate row (Building based, Total EUR/m2,
# 2010): used as the fallback whenever a country can't be matched, mirroring
# the GLOBAL fallback pattern in DamageTable above.
_WORLD_FALLBACK: tuple[float, float] = (442.37, 638.9)

# Best-effort alternate spellings likely to appear in LSIB's country_na but
# not verbatim in the workbook's World Bank-style names. NOT verified against
# a live Earth Engine session -- see the module docstring and
# check_country_name_matches().
_COUNTRY_NAME_ALIASES: dict[str, str] = {
    "russia": "Russian Federation",
    "south korea": "Korea, Rep.",
    "republic of korea": "Korea, Rep.",
    "north korea": "Korea, Dem. Rep.",
    "democratic people's republic of korea": "Korea, Dem. Rep.",
    "ivory coast": "Cote d'Ivoire",
    "cote d ivoire": "Cote d'Ivoire",
    "democratic republic of the congo": "Congo, Dem. Rep.",
    "dr congo": "Congo, Dem. Rep.",
    "republic of the congo": "Congo, Rep.",
    "congo": "Congo, Rep.",
    "czechia": "Czech Republic",
    "syria": "Syrian Arab Republic",
    "iran": "Iran, Islamic Rep.",
    "laos": "Lao PDR",
    "myanmar burma": "Myanmar",
    "burma": "Myanmar",
    "viet nam": "Vietnam",
    "brunei": "Brunei Darussalam",
    "cape verde": "Cabo Verde",
    "eswatini": "Swaziland",
    "north macedonia": "Macedonia, FYR",
    "macedonia": "Macedonia, FYR",
    "the gambia": "Gambia, The",
    "gambia": "Gambia, The",
    "the bahamas": "Bahamas, The",
    "bahamas": "Bahamas, The",
    "micronesia": "Micronesia, Fed. Sts.",
    "federated states of micronesia": "Micronesia, Fed. Sts.",
    "east timor": "Timor-Leste",
    "slovakia": "Slovak Republic",
    "kyrgyzstan": "Kyrgyz Republic",
    "egypt": "Egypt, Arab Rep.",
    "venezuela": "Venezuela, RB",
    "yemen": "Yemen, Rep.",
    "hong kong": "Hong Kong SAR, China",
    "macau": "Macao SAR, China",
    "macao": "Macao SAR, China",
    "united states of america": "United States",
    "usa": "United States",
    "us": "United States",
    "uk": "United Kingdom",
    "great britain": "United Kingdom",
    "saint lucia": "St. Lucia",
    "saint kitts and nevis": "St. Kitts and Nevis",
    "saint vincent and the grenadines": "St. Vincent and the Grenadines",
    "west bank": "West Bank and Gaza",
    "gaza": "West Bank and Gaza",
    "gaza strip": "West Bank and Gaza",
    "palestine": "West Bank and Gaza",
    "palestinian territory": "West Bank and Gaza",
}


def _normalize_country_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace.

    Makes "Côte d'Ivoire", "COTE D IVOIRE" and "Cote d'Ivoire" compare equal.
    """
    stripped = unicodedata.normalize("NFKD", name)
    stripped = "".join(ch for ch in stripped if not unicodedata.combining(ch))
    stripped = stripped.lower()
    stripped = re.sub(r"[^a-z0-9\s]", " ", stripped)
    return re.sub(r"\s+", " ", stripped).strip()


@dataclass(frozen=True)
class MaxDamage:
    """Per-country maximum damage, EUR/m^2 (2010 prices), building based."""

    country_name: str  # the workbook's own name for the matched row
    residential_eur_per_m2: float
    commercial_eur_per_m2: float
    matched: bool  # False when no country matched and the World fallback was used


class MaxDamageTable:
    """Loaded JRC MaxDamage table, ready for per-country lookup."""

    def __init__(self, rows: tuple[tuple[str, str, float, float], ...]):
        self._by_name: dict[str, tuple[str, float, float]] = {}
        for name, _iso3, res, com in rows:
            self._by_name[_normalize_country_name(name)] = (name, float(res), float(com))

    def lookup(self, country_name: str) -> MaxDamage:
        """MaxDamage for ``country_name`` (as returned by resolve_continent()).

        Tries an exact normalized match first, then _COUNTRY_NAME_ALIASES,
        then falls back to the workbook's World aggregate.
        """
        key = _normalize_country_name(country_name)
        entry = self._by_name.get(key)
        if entry is None:
            alias = _COUNTRY_NAME_ALIASES.get(key)
            if alias is not None:
                entry = self._by_name.get(_normalize_country_name(alias))
        if entry is None:
            res, com = _WORLD_FALLBACK
            return MaxDamage(
                country_name="World (fallback)",
                residential_eur_per_m2=res,
                commercial_eur_per_m2=com,
                matched=False,
            )
        matched_name, res, com = entry
        return MaxDamage(
            country_name=matched_name,
            residential_eur_per_m2=res,
            commercial_eur_per_m2=com,
            matched=True,
        )


def load_max_damage_table() -> MaxDamageTable:
    """Build a :class:`MaxDamageTable` from the inline JRC data."""
    return MaxDamageTable(_MAX_DAMAGE_ROWS)


def check_country_name_matches() -> list[str]:
    """One-off helper: list every distinct LSIB country name that would fall
    back to the World aggregate instead of matching a real row.

    Run this once against your real GEE session to find gaps in
    _COUNTRY_NAME_ALIASES, e.g.:

        from flood_pipeline import gee
        from flood_pipeline.steps.depth_damage_prov import check_country_name_matches
        gee.init_gee("<your-gee-project>")
        for name in check_country_name_matches():
            print(name)
    """
    import ee

    table = load_max_damage_table()
    countries = ee.FeatureCollection(COUNTRIES_ASSET)
    names = sorted(set(countries.aggregate_array("country_na").getInfo()))
    return [name for name in names if not table.lookup(name).matched]