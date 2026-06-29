import pystac_client
import odc.stac
import rioxarray
import geopandas as gpd

aoi = r"D:\EAGLE\SoSe2026\openeo_flood\aoi.gpkg"
aoi = gpd.read_file(aoi)
aoi = aoi.to_crs(epsg=4326)
EODC = "https://stac.eodc.eu/api/v1"

west, south, east, north = aoi.total_bounds
spatial_extent = [west, south, east, north]
temporal_extent = ["2021-01-01", "2026-12-31"]

catalog = pystac_client.Client.open(EODC)
dtm_items = catalog.search(
    bbox=spatial_extent,
    datetime=temporal_extent,
    collections=["topo-dc-austria-dtm"]).item_collection()
print(dtm_items)
# odc-stac reprojects korrekt aus der nativen Equi7Grid-Projektion
dtm_cube = odc.stac.load(
    dtm_items,
    crs="EPSG:4326",
    resolution=0.0003,
    bbox=spatial_extent,
    resampling="nearest",
)
print(dtm_cube)

dtm = dtm_cube.median(dim="time")  # DTM ist zeitlich konstant, aber es gibt mehrere Aufnahmen → Mittelwert
dtm = dtm.rio.write_crs("EPSG:4326")
dtm.rio.to_raster("dtm.tif")

# <pystac.item_collection.ItemCollection object at 0x0000021F511A9D30>
# <xarray.Dataset> Size: 430MB
# Dimensions:      (latitude: 4866, longitude: 5522, time: 2)
# Coordinates:
#   * latitude     (latitude) float64 39kB 49.05 49.05 49.05 ... 47.59 47.59 47.59
#   * longitude    (longitude) float64 44kB 14.46 14.46 14.46 ... 16.12 16.12
#     spatial_ref  int32 4B 4326
#   * time         (time) datetime64[ns] 16B 2023-09-15 2024-09-15
# Data variables:
#     data         (time, latitude, longitude) float32 215MB nan nan ... nan nan
#     mask         (time, latitude, longitude) float32 215MB nan nan ... nan nan