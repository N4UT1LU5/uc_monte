from cmath import isnan
import time
import rich
import re
import rasterio
from rasterio import features
from rasterio.enums import Resampling
import rioxarray  # for the extension to load
import numpy
import os
import cv2
import gc
from memory_profiler import profile


import geopandas

from rich.console import Console

console = Console()

INPUT_PATH = "./input/"
OUTPUT_PATH = "./output/"
DIFF_OUTPUT = OUTPUT_PATH + "veg_diff.tif"

NDVI_THRESHOLD = 0.4
KERNEL_SIZE = 2
GSD = 1.0  # 1m/px
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
    with console.status("[bold green]creating shape from raster...") as status:
        shapes = []
        shapes.extend(
            list(
                features.shapes(np_array, mask=(np_array == 1), transform=img.transform)
            )
        )
        # convert json dict into geodataframe
        shapejson = (
            {"type": "Feature", "properties": {}, "geometry": s}
            for i, (s, v) in enumerate(shapes)
        )
        collection = {"type": "FeatureCollection", "features": list(shapejson)}
        veg_gdf = geopandas.GeoDataFrame.from_features(collection["features"])

    with console.status("[green]Cleanup shapes...") as status:
        veg_gdf = veg_gdf.dissolve().explode(index_parts=True)
        veg_gdf.set_crs("epsg:25832")
        result_gdf = veg_gdf
        # result_gdf = geopandas.GeoDataFrame(geometry=veg_gdf.buffer(5, 4).buffer(-5, 4))
        result_gdf = geopandas.GeoDataFrame(
            geometry=veg_gdf.buffer(-0.5, 4).buffer(0.5, 4)
        )
        # result_gdf = veg_gdf.simplify(1)
        out_p = os.path.join(OUTPUT_PATH, "mask_shape.geojson")
    result_gdf.to_file(out_p, driver="GeoJSON")
    console.log(f"sucsessfully created GeoJSON Shape: {out_p}")


def morph_img(img, kernel_pxSize):
    kernel_pxSize = int(kernel_pxSize)
    img = img.astype(numpy.uint8)
    img = cv2.morphologyEx(
        img,
        cv2.MORPH_CLOSE,
        kernel=numpy.ones((kernel_pxSize, kernel_pxSize), numpy.uint8),
    )
    img = cv2.morphologyEx(
        img,
        cv2.MORPH_OPEN,
        kernel=numpy.ones(
            (int(kernel_pxSize * 1.2), int(kernel_pxSize * 1.2)), numpy.uint8
        ),
    )
    return img


# test. not in use
def opening_img(path_to_file, kernel_pxSize):
    # https://www.geeksforgeeks.org/erosion-dilation-images-using-opencv-python/
    with rasterio.open(path_to_file, "r+") as dst:
        img = dst.read(1).astype(numpy.uint8)
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

    with rasterio.open(OUTPUT_PATH + "erosion.tif", "r+", compress="lzw") as dst2:
        dst2.write_band(1, img)


# @profile
def ndvi_mask(threshold, gsd=0.5):
    """
    create tiff mask of ndvi threshold
    """
    # https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

    console.log(f"[yellow]Start process: Vegetation mask")
    veg_list = []
    cnt = 1
    img_dict = load_tiff(INPUT_PATH)
    for key in img_dict:
        dataset = img_dict[key]
        gsd_y, gsd_x = img_dict[key].res  # get GSD
        # console.log(gsd_y, gsd_x)
        with console.status(
            f"[green]Calculating NDVI for {dataset.name} - {cnt} of {len(img_dict)}"
        ) as status:
            scale = 1 / (1 / gsd_x * gsd)
            data = dataset.read(
                out_shape=(
                    dataset.count,
                    int(dataset.height * scale),
                    int(dataset.width * scale),
                ),
                resampling=Resampling.bilinear,
            )
            transform = dataset.transform * dataset.transform.scale(
                (dataset.width / data.shape[-1]), (dataset.height / data.shape[-2])
            )

            kwargs = dataset.meta

            kwargs.update(
                transform=transform,
                height=dataset.height * scale,
                width=dataset.width * scale,
                dtype=rasterio.float32,
                count=1,
                compress="lzw",
            )

            bandNIR = data[0].astype(numpy.float16)
            bandRed = data[1].astype(numpy.float16)
            ndvi = numpy.zeros(data.shape, dtype=numpy.float16)
            veg = numpy.zeros(data.shape, dtype=numpy.int16)
            ndvi = (bandNIR - bandRed) / (bandNIR + bandRed)

            # print(ndvi.max(), ndvi.min())
            with rasterio.open(
                os.path.join(OUTPUT_PATH, "ndvi", f"ndvi_{key}.tif"), "w", **kwargs
            ) as dst:
                dst.write_band(1, ndvi)

        with console.status(
            f"[green]Creating Vegetation mask for {img_dict[key].name} - {cnt} of {len(img_dict)}"
        ) as status:
            veg = ndvi
            veg[veg > threshold] = 1.0
            veg[veg <= threshold] = 0.0

            veg = veg.astype(numpy.byte)
            veg_list.append(veg)
            veg = morph_img(veg, 1 / gsd)
            if cnt == 1:
                veg = numpy.logical_not(veg)
                pass
            out_filename = os.path.join(OUTPUT_PATH, "veg", f"veg_{key}.tif")
            with rasterio.open(out_filename, "w", **kwargs) as dst:
                dst.write_band(1, veg.astype(numpy.int8))
        console.log(f"Successfuly created mask: {out_filename}")
        cnt += 1
    img_dict[key].close()
    console.log(f"[yellow]Finished process: Vegetation mask")


# @profile
def calc_succession():
    time.sleep(1)
    console.log(f"[yellow]Start process: Succession mask")
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
    com = f"cp {list(path_dict.values())[1]} {DIFF_OUTPUT}"
    os.system(com)
    # add all veg mask except first (oldest)
    for i in range(1, len(path_dict) - 1):
        path_1 = DIFF_OUTPUT
        path_2 = list(path_dict.values())[i + 1]
        with rioxarray.open_rasterio(
            path_1, masked=True
        ) as xds, rioxarray.open_rasterio(path_2, masked=True) as xds_match:
            # reproject raster 2 onto 1
            xds_repr_match = xds.rio.reproject_match(xds_match)
            xds_add = xds_repr_match + xds_match  # sum up rasters

            del xds_match, xds_repr_match
            gc.collect()

            xds_add.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")

            del xds_add
            gc.collect()

    path_2 = list(path_dict.values())[0]
    with rioxarray.open_rasterio(path_1, masked=True) as xds, rioxarray.open_rasterio(
        path_2, masked=True
    ) as xds_match:
        # reproject raster 2 onto 1
        xds_repr_match = xds.rio.reproject_match(xds_match)

        xds_add = xds_match * xds_repr_match

        del xds_match, xds_repr_match
        gc.collect()
        xds_add.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")

        del xds_add
        gc.collect()
    console.log(f"[yellow]Finished process: Succession mask")


if __name__ == "__main__":
    console.log("[blue bold]>>--<Program Start>--<<")
    T_start = time.time()

    ndvi_mask(NDVI_THRESHOLD, 0.5)
    calc_succession()
    # opening_img(DIFF_OUTPUT, 10)
    raster_to_shape(DIFF_OUTPUT)
    T_end = time.time()
    t = time.strftime("%M:%S", time.gmtime(T_end - T_start))
    console.log(f"[blue bold]Duration: {t}")
    pass
