import re
import rasterio
from rasterio import features
import numpy
import os
import geopandas

import rioxarray  # for the extension to load
import xarray

# https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

INPUT_PATH = "./input/"
OUTPUT_PATH = "./output/"

NDVI_THRESHOLD = 0.5

numpy.seterr(divide="ignore", invalid="ignore")


def load_tiff(input_path):
    dict_of_imgs = {}
    for f in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, f)):
            path_file = os.path.join(input_path, f)
            year = re.search(r"\d+", f).group()
            dict_of_imgs.update({year: rasterio.open(path_file)})
    return dict_of_imgs


def raster_to_shape(np_array):
    """
    makes geojson from raster
    """
    shapes = []
    shapes.extend(
        list(
            features.shapes(
                np_array,
                mask=(np_array != 1),
            )
        )
    )
    # convert json dict into geodataframe
    shapejson = (
        {"type": "Feature", "properties": {}, "geometry": s}
        for i, (s, v) in enumerate(shapes)
    )
    collection = {"type": "FeatureCollection", "features": list(shapejson)}
    flood_gdf = geopandas.GeoDataFrame.from_features(collection["features"])


def ndvi_mask(threshold, img_dict):
    """
    create tiff mask of ndvi threshold
    """
    veg_list = []
    a = ""
    for key in img_dict:
        a = key
        bandNIR = img_dict[key].read(1)
        bandRed = img_dict[key].read(2)

        kwargs = img_dict[key].meta
        kwargs.update(dtype=rasterio.float32, count=1, compress="lzw")

        ndvi = numpy.zeros(img_dict[key].shape, dtype=numpy.float16)
        veg = numpy.zeros(img_dict[key].shape, dtype=numpy.int16)
        ndvi = (bandNIR.astype(numpy.float16) - bandRed.astype(numpy.float16)) / (
            bandNIR.astype(numpy.float16) + bandRed.astype(numpy.float16)
        )

        # with rasterio.open(
        #     os.path.join(OUTPUT_PATH, f"ndvi_{key}.tif"), "w", **kwargs
        # ) as dst:
        #     dst.write_band(1, ndvi.astype(rasterio.float32))

        max_v = numpy.nanmax(ndvi)
        print(f"{key} max {max_v}")
        veg = ndvi
        veg[veg > threshold] = 1.0
        veg[veg <= threshold] = 0.0
        veg = veg.astype(numpy.int16)
        veg_list.append(veg)
        print("checkpoint")

        with rasterio.open(
            os.path.join(OUTPUT_PATH, f"veg_{key}.tif"), "w", **kwargs
        ) as dst:
            dst.write_band(1, veg.astype(numpy.int8))
        print("done mask" + key)

    img_dict[key].close()
    calc_diff()


def calc_diff():
    xds = xarray.open_dataarray(OUTPUT_PATH + "veg_16.tif", masked=True)
    xds_match = xarray.open_dataarray(OUTPUT_PATH + "veg_17.tif", masked=True)
    xds_repr_match = xds.rio.reproject_match(xds_match)
    xds_sum = xds_repr_match - xds_match
    xds_sum.rio.to_raster(
        OUTPUT_PATH + "veg_diff.tif", dtype=numpy.int8, compress="lzw"
    )

    with rasterio.open(os.path.join(OUTPUT_PATH, "veg_diff.tif"), "r+") as dst:
        ras = dst.read(1)
        ras[ras > 0] = 0
        ras[ras < 0] = 1
        dst.write_band(1, ras.astype(numpy.int8))


if __name__ == "__main__":

    tiffs = load_tiff(INPUT_PATH)
    ndvi_mask(NDVI_THRESHOLD, tiffs)
