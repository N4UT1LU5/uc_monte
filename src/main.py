import re
import rasterio
from rasterio import features
import rioxarray  # for the extension to load
import numpy
import os
import cv2

import geopandas


# import xarray

# https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

INPUT_PATH = "./input/"
OUTPUT_PATH = "./output/"
DIFF_OUTPUT = OUTPUT_PATH + "veg_diff.tif"

NDVI_THRESHOLD = 0.5
KERNEL_SIZE = 20
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


def raster_to_shape(path_to_file):
    """
    makes geojson from raster
    """
    with rasterio.open(path_to_file) as img:
        np_array = img.read(1).astype(numpy.int16)
    shapes = []
    shapes.extend(
        list(features.shapes(np_array, mask=(np_array == 1), transform=img.transform))
    )
    # convert json dict into geodataframe
    shapejson = (
        {"type": "Feature", "properties": {}, "geometry": s}
        for i, (s, v) in enumerate(shapes)
    )
    collection = {"type": "FeatureCollection", "features": list(shapejson)}
    veg_gdf = geopandas.GeoDataFrame.from_features(collection["features"])

    veg_gdf = veg_gdf.dissolve().explode(index_parts=True)
    veg_gdf.set_crs("epsg:25832")
    result_gdf = veg_gdf
    # result_gdf = geopandas.GeoDataFrame(geometry=veg_gdf.buffer(5, 4).buffer(-5, 4))
    result_gdf = geopandas.GeoDataFrame(geometry=veg_gdf.buffer(-0.5, 4).buffer(0.5, 4))
    # result_gdf = veg_gdf.simplify(1)
    result_gdf.to_file(
        os.path.join(OUTPUT_PATH, "mask_shape.geojson"), driver="GeoJSON"
    )


def morph_img(img, kernel_pxSize):
    img = img.astype(numpy.uint8)
    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel=numpy.ones((kernel_pxSize, kernel_pxSize), numpy.uint8),
    )
    img = cv2.morphologyEx(
        img,
        cv2.MORPH_OPEN,
        kernel=numpy.ones((kernel_pxSize + 5, kernel_pxSize + 5), numpy.uint8),
    )
    return img


def opening_img(path_to_file, kernel_pxSize):
    # https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
    with rasterio.open(path_to_file, "r+") as dst:
        img = dst.read(1).astype(numpy.uint8)
        img = cv2.morphologyEx(
            img,
            cv2.MORPH_CLOSE,
            kernel=numpy.ones((kernel_pxSize * 2, kernel_pxSize * 2), numpy.uint8),
        )
        img = cv2.morphologyEx(
            img,
            cv2.MORPH_OPEN,
            kernel=numpy.ones((kernel_pxSize, kernel_pxSize), numpy.uint8),
        )

    with rasterio.open(OUTPUT_PATH + "erosion.tif", "r+", compress="lzw") as dst2:
        dst2.write_band(1, img)


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
            os.path.join(OUTPUT_PATH, "ndvi", f"ndvi_{key}.tif"), "w", **kwargs
        ) as dst:
            dst.write_band(1, ndvi.astype(numpy.float16))

        max_v = numpy.nanmax(ndvi)
        print(f"{key} max {max_v}")
        veg = ndvi
        veg[veg > threshold] = 1.0
        veg[veg <= threshold] = 0.0
        veg = veg.astype(numpy.byte)
        veg_list.append(veg)
        veg = morph_img(veg, KERNEL_SIZE)

        print("checkpoint")

        with rasterio.open(
            os.path.join(OUTPUT_PATH, "veg", f"veg_{key}.tif"), "w", **kwargs
        ) as dst:
            dst.write_band(1, veg.astype(numpy.int8))
        print(f"done mask {key}")

    img_dict[key].close()


def calc_succession():
    path_dict = {}
    veg_path = os.path.join(OUTPUT_PATH, "veg")
    for f in os.listdir(veg_path):
        if os.path.isfile(os.path.join(veg_path, f)):
            path_file = os.path.join(veg_path, f)
            try:
                year = int(re.search(r"\d+", f).group())
            except:
                continue
            path_dict.update({year: path_file})
    path_dict = dict(sorted(path_dict.items()))

    i = 0
    while i < (len(path_dict) - 1):
        print(i)
        if i == 0:
            path_1 = list(path_dict.values())[0]
            path_2 = list(path_dict.values())[1]
            print(f"p1: {path_1}, p2: {path_2}")
            with rioxarray.open_rasterio(
                path_1, masked=True
            ) as xds, rioxarray.open_rasterio(path_2, masked=True) as xds_match:
                xds_repr_match = xds.rio.reproject_match(xds_match)
                xds_diff = xds_repr_match - xds_match
            print("subrtraction complete")
            # xds_repr_match.close()
            xds_diff.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")
            # print(xds_diff)
            xds = xds_match = xds_diff = xds_repr_match = None
            print(f"closed {xds} and {xds_match}")
            with rasterio.open(DIFF_OUTPUT, "r+") as dst:
                ras = dst.read(1)
                ras[ras > 0] = 0
                ras[ras < 0] = 1
                dst.write_band(1, ras.astype(numpy.int8))
        else:
            with rioxarray.open_rasterio(
                DIFF_OUTPUT, masked=True
            ) as xds, rioxarray.open_rasterio(
                list(path_dict.values())[i + 1], masked=True
            ) as xds_match:
                xds_repr_match = xds.rio.reproject_match(xds_match)

                xds_match += xds_repr_match
                # print(xds_match)
                xds_match.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")

            # print(f"closed {xds} and {xds_match}")

            # xds_repr_match.close()
            with rasterio.open(DIFF_OUTPUT, "r+") as dst:
                ras = dst.read(1)
                print(ras.max())
                ras[ras < 2] = 0
                ras[ras == 2] = 1
                ras = morph_img(ras, KERNEL_SIZE)
                dst.write_band(1, ras.astype(numpy.int8))
        i += 1


if __name__ == "__main__":
    # tiffs = load_tiff(INPUT_PATH)
    # ndvi_mask(NDVI_THRESHOLD, tiffs)
    # calc_succession()
    raster_to_shape(DIFF_OUTPUT)
    # opening_img(DIFF_OUTPUT, 10)
    pass
