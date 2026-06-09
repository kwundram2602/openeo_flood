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
temporal_extent = ["2024-09-15", "2024-09-20"]

catalog = pystac_client.Client.open(EODC)
gfm_items = catalog.search(
    bbox=spatial_extent,
    datetime=temporal_extent,
    collections=["GFM"],
    max_items=20
).item_collection()

# odc-stac reprojects korrekt aus der nativen Equi7Grid-Projektion
gfm_cube = odc.stac.load(
    gfm_items,
    bands=["ensemble_flood_extent"],
    crs="EPSG:4326",
    resolution=0.0003,
    bbox=spatial_extent,
    resampling="nearest",
)
print(gfm_cube)

flood = gfm_cube["ensemble_flood_extent"].astype("float32")
flood = flood.where(flood != 255)  # 255 = nodata → NaN
gfm_max = flood.max(dim="time")
gfm_sum = flood.sum(dim="time")
#print(gfm_max)

gfm_sum = gfm_sum.rio.write_crs("EPSG:4326")
gfm_sum.rio.to_raster("gfm_flood_sum.tif")
gfm_max = gfm_max.rio.write_crs("EPSG:4326")
gfm_max.rio.to_raster("gfm_flood_max.tif")