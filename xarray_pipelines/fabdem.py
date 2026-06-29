import ee 
import geemap
import geopandas

ee.Authenticate()
ee.Initialize(project="ee-kwun")
FABDEM_COLL = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")


def get_fabdem(aoi: geopandas.GeoDataFrame) -> ee.Image:
    """
    Get the FABDEM elevation mosaic for a given area of interest.

    FABDEM is a static DEM with no time dimension, so no date filtering is
    applied. The tiles intersecting the AOI are mosaicked into a single image.

    Args:
        aoi (geopandas.GeoDataFrame): Area of interest as a GeoDataFrame (EPSG:4326).

    Returns:
        ee.Image: The FABDEM elevation mosaic (band 'b1') over the AOI.
    """
    # Convert the GeoDataFrame to an Earth Engine geometry
    aoi_ee = geemap.geopandas_to_ee(aoi)

    # Filter the FABDEM collection by area of interest and mosaic the tiles
    return FABDEM_COLL.filterBounds(aoi_ee).mosaic()


if __name__ == "__main__":
    # Example usage
    aoi_path = r"D:\EAGLE\SoSe2026\openeo_flood\aoi.gpkg"
    aoi = geopandas.read_file(aoi_path)
    aoi = aoi.to_crs(epsg=4326)
    aoi_ee = geemap.geopandas_to_ee(aoi)

    fabdem_image = get_fabdem(aoi)
    clip = fabdem_image.clip(aoi_ee)
    geemap.ee_export_image_to_drive(
        clip,
        description="fabdem_export",
        folder="fabdem_exports",
        fileNamePrefix="fabdem",
        scale=30,
        region=aoi_ee.geometry(),
        fileFormat="GeoTIFF",
    )