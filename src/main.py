import re
import rasterio
from rasterio import features
import numpy
import os

# import geopandas

import rioxarray  # for the extension to load
import xarray

# https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

INPUT_PATH = "./input/"
OUTPUT_PATH = "./output/"
DIFF_OUTPUT = OUTPUT_PATH + "veg_diff.tif"

NDVI_THRESHOLD = 0.5

numpy.seterr(divide="ignore", invalid="ignore")


def load_tiff(input_path):
    dict_of_imgs = {}
    for f in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, f)):
            path_file = os.path.join(input_path, f)
            year = int(re.search(r"\d+", f).group())
            dict_of_imgs.update({year: rasterio.open(path_file)})
    # sort images by year
    dict_of_imgs = dict(sorted(dict_of_imgs.items()))
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

        with rasterio.open(
            os.path.join(OUTPUT_PATH, f"ndvi_{key}.tif"), "w", **kwargs
        ) as dst:
            dst.write_band(1, ndvi.astype(rasterio.float32))

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
    # calc_diff()


def calc_succession():
    path_dict = {}
    for f in os.listdir(INPUT_PATH):
        if os.path.isfile(os.path.join(INPUT_PATH, f)):
            path_file = os.path.join(INPUT_PATH, f)
            year = int(re.search(r"\d+", f).group())
            path_dict.update({year: path_file})
    path_dict = dict(sorted(path_dict.items()))

    i = 0
    while i < len(path_dict):
        print(i)
        if i == 0:
            xds = xarray.open_dataarray(list(path_dict.values())[0], masked=True)
            xds_match = xarray.open_dataarray(list(path_dict.values())[1], masked=True)
            xds_repr_match = xds.rio.reproject_match(xds_match)
            xds_sum = xds_repr_match - xds_match
            xds_sum.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")
            with rasterio.open(DIFF_OUTPUT, "r+") as dst:
                ras = dst.read(1)
                ras[ras > 0] = 0
                ras[ras < 0] = 1
                dst.write_band(1, ras.astype(numpy.int8))
        else:
            xds = xarray.open_dataarray(OUTPUT_PATH, masked=True)
            xds_match = xarray.open_dataarray(
                list(path_dict.values())[i + 1], masked=True
            )
            xds_repr_match = xds.rio.reproject_match(xds_match)
            xds_sum = xds_repr_match + xds_match
            xds_sum.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")
            with rasterio.open(DIFF_OUTPUT, "r+") as dst:
                ras = dst.read(1)
                ras[ras == 2] = 1
                ras[ras < 2] = 0
                dst.write_band(1, ras.astype(numpy.int8))


if __name__ == "__main__":
    calc_succession()
    # tiffs = load_tiff(INPUT_PATH)
    # ndvi_mask(NDVI_THRESHOLD, tiffs)
