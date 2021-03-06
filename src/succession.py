#!/usr/bin/env python3

import argparse
import time
import re
import rasterio
from rasterio import features
from rasterio.enums import Resampling
import rioxarray  # for the extension to load
import numpy
import os
import cv2
import gc
import geopandas
from rich.console import Console

# from memory_profiler import profile

console = Console()

NDVI_THRESHOLD = 0.5
KERNEL_SIZE = 2
GSD = 1.0  # 1m/px
numpy.seterr(divide="ignore", invalid="ignore")


def dir_path(path_string):
    if os.path.isdir(path_string):
        return path_string
    else:
        console.log(f'[red] "{path_string}" not a valid path')
        exit()


parser = argparse.ArgumentParser(
    description="Tool to detect ecological succession in a set of CIR-Orthoimages"
)
parser.add_argument(
    "-i", "-input", dest="in_path", help="Input Path", type=dir_path, required=True
)
parser.add_argument(
    "-o",
    "-output",
    dest="out_path",
    help="Output path",
    type=str,
    required=True,
)
parser.add_argument(
    "-gsd", type=float, help="Ground Sample Distance in meter per pixel", default=GSD
)
parser.add_argument(
    "-th",
    type=float,
    help="Vegetation Threshold. Range -1.0 to 1.0",
    default=NDVI_THRESHOLD,
)

args = parser.parse_args()

# set gloabal variables
INPUT_PATH = args.in_path
OUTPUT_PATH = os.path.abspath(args.out_path)
OUTPUT_PATH_VEG = os.path.join(OUTPUT_PATH, "veg")
OUTPUT_PATH_NDVI = os.path.join(OUTPUT_PATH, "ndvi")
DIFF_OUTPUT = os.path.join(OUTPUT_PATH, "veg_diff.tif")
NDVI_THRESHOLD = args.th
GSD = args.gsd

# create output directory structure
def create_out_diectory(path):
    try:
        os.makedirs(path)
        console.log(f"Create Path: {path}")
    except FileExistsError:
        console.log(f"Path already exists: {path}")
    pass


def get_dir_size(path):
    total_size = 0
    for elm in os.scandir(path):
        if elm.is_file():
            total_size += os.path.getsize(elm)
        elif elm.is_dir():
            total_size += get_dir_size(elm.path)
    return total_size


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def load_tiff(input_path):
    dict_of_imgs = {}
    for f in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path, f)):
            path_file = os.path.join(input_path, f)
            year = int(re.search(r"\d+", f).group())
            dict_of_imgs.update({year: path_file})
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
                features.shapes(np_array, mask=(np_array >= 1), transform=img.transform)
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
        result_gdf = geopandas.GeoDataFrame(
            geometry=veg_gdf.buffer(-0.5, 4).buffer(0.5, 4)
        )
        # result_gdf = veg_gdf.simplify(1)
        out_p = os.path.join(OUTPUT_PATH, "mask_shape.geojson")
    result_gdf.to_file(out_p, driver="GeoJSON")
    console.log(
        f"Sucsessfully created GeoJSON Shape: {out_p} [blue]{sizeof_fmt(os.path.getsize(out_p))}"
    )


def morph_img(image, kernel_pxSize):
    def cvmorph(img, kernel):
        kernel = int(kernel)
        img = img.astype(numpy.uint8)
        img = cv2.morphologyEx(
            img,
            cv2.MORPH_CLOSE,
            kernel=numpy.ones((kernel, kernel), numpy.uint8),
        )
        img = cv2.morphologyEx(
            img,
            cv2.MORPH_OPEN,
            kernel=numpy.ones((int(kernel * 1.2), int(kernel * 1.2)), numpy.uint8),
        )
        return img

    if os.path.exists(image):
        with rasterio.open(image, "r+") as dst:
            image = dst.read(1).astype(numpy.uint8)
            image = cvmorph(image, kernel_pxSize)
            dst.write_band(1, image)
            return
    else:
        return cvmorph(image, kernel_pxSize)


# @profile
def creat_veg_raster(threshold=0.5, gsd=0.5):
    """
    create tiff mask of ndvi threshold
    """
    # https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

    console.log(f"[yellow]Start process: Vegetation mask")
    cnt = 1
    img_dict = load_tiff(INPUT_PATH)
    for key in img_dict:
        dataset = rasterio.open(img_dict[key])
        gsd_y, gsd_x = dataset.res  # get GSD
        if gsd < gsd_x:
            gsd = gsd_x
            console.log(
                f"Selected GSD smaller than GSD of input file. Reset GSD to Input: {gsd}"
            )
        # console.log(gsd_y, gsd_x)
        with console.status(
            f"[green]Calculating NDVI for {dataset.name} - {cnt} of {len(img_dict)}"
        ) as status:
            scale = 1 / (gsd / gsd_x)
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
            ndvi_out_path = os.path.join(OUTPUT_PATH_NDVI, f"ndvi_{key}.tif")
            with rasterio.open(ndvi_out_path, "w", **kwargs) as dst:
                dst.write_band(1, ndvi)
        console.log(
            f"Successfuly created mask: {ndvi_out_path} [blue]{sizeof_fmt(os.path.getsize(ndvi_out_path))}"
        )
        with console.status(
            f"[green]Creating Vegetation mask for {dataset.name} - {cnt} of {len(img_dict)}"
        ) as status:
            veg = ndvi
            veg[veg > threshold] = 1.0
            veg[veg <= threshold] = 0.0

            veg = veg.astype(numpy.byte)
            veg = morph_img(veg, 1 / gsd)
            if cnt == 1:
                veg = numpy.logical_not(veg)
                pass
            veg_out_path = os.path.join(OUTPUT_PATH_VEG, f"veg_{key}.tif")
            with rasterio.open(veg_out_path, "w", **kwargs) as dst:
                dst.write_band(1, veg.astype(numpy.int8))
        console.log(
            f"Successfuly created mask: {veg_out_path} [blue]{sizeof_fmt(os.path.getsize(veg_out_path))}"
        )
        cnt += 1
    dataset.close()
    console.log(f"[yellow]Finished process: Vegetation mask")


# @profile
def analyse_succession():
    time.sleep(1)
    console.log(f"[yellow]Start process: Succession mask")
    path_dict = load_tiff(OUTPUT_PATH_VEG)
    os.system(f"cp {list(path_dict.values())[1]} {DIFF_OUTPUT}")
    # add all veg mask except first (oldest)
    for i in range(1, len(path_dict) - 1):
        path_1 = DIFF_OUTPUT
        path_2 = list(path_dict.values())[i + 1]
        with rioxarray.open_rasterio(
            path_1, masked=True
        ) as succ_ras, rioxarray.open_rasterio(path_2, masked=True) as veg_match:
            # reproject raster 2 onto 1
            succ_repr_match = succ_ras.rio.reproject_match(veg_match)
            raster_sum = succ_repr_match + veg_match  # sum up rasters

            del veg_match, succ_repr_match
            gc.collect()

            raster_sum.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")

            del raster_sum
            gc.collect()

    path_2 = list(path_dict.values())[0]
    with rioxarray.open_rasterio(
        path_1, masked=True
    ) as succ_ras, rioxarray.open_rasterio(path_2, masked=True) as veg_match:
        # reproject raster 2 onto 1
        succ_repr_match = succ_ras.rio.reproject_match(veg_match)
        raster_product = veg_match * succ_repr_match

        del veg_match, succ_repr_match
        gc.collect()
        raster_product.rio.to_raster(DIFF_OUTPUT, dtype=numpy.int8, compress="lzw")
        morph_img(DIFF_OUTPUT, 1 / GSD)
        del raster_product
        gc.collect()
        console.log(
            f"Successfuly created succession raster: {path_1} [blue]{sizeof_fmt(os.path.getsize(path_1))}"
        )
    console.log(f"[yellow]Finished process: Succession mask")


if __name__ == "__main__":
    if NDVI_THRESHOLD > 1 or NDVI_THRESHOLD < -1:
        console.log(
            f"[red] NDVI threshold out of range. Must be between -1.0 and 1.0. Your choice was: {NDVI_THRESHOLD}"
        )
        exit()
    console.log("[green bold]>>--<Program Start>--<<")
    console.log(f"[blue] GSD:{GSD} m/px | Threshold:{NDVI_THRESHOLD}")

    T_start = time.time()
    create_out_diectory(OUTPUT_PATH_NDVI)
    create_out_diectory(OUTPUT_PATH_VEG)
    creat_veg_raster(NDVI_THRESHOLD, GSD)
    analyse_succession()
    raster_to_shape(DIFF_OUTPUT)
    T_end = time.time()
    t = time.strftime("%M:%S", time.gmtime(T_end - T_start))

    console.log(
        f"[blue bold]Duration: {t} | Output folder size [blue]{sizeof_fmt(get_dir_size(OUTPUT_PATH))}"
    )
    pass
