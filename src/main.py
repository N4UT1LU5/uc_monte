import re
import rasterio
import numpy
import os

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


def ndvi_mask(threshold, img_dict):
    """
    create tiff mask of ndvi threshold
    """
    cnt = 0
    for key in img_dict:

        bandNIR = img_dict[key].read(1)
        bandRed = img_dict[key].read(2)

        kwargs = img_dict[key].meta
        kwargs.update(dtype=rasterio.float32, count=1, compress="lzw")

        ndvi = numpy.zeros(img_dict[key].shape, dtype=rasterio.float32)
        veg = numpy.zeros(img_dict[key].shape, dtype=rasterio.float32)
        ndvi = (bandNIR.astype(float) - bandRed.astype(float)) / (
            bandNIR.astype(float) + bandRed.astype(float)
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
        print("checkpoint")

        with rasterio.open(
            os.path.join(OUTPUT_PATH, f"veg_{key}.tif"), "w", **kwargs
        ) as dst:
            dst.write_band(1, veg.astype(rasterio.float32))
        img_dict[key].close()
        print("done mask" + key)


if __name__ == "__main__":
    tiffs = load_tiff(INPUT_PATH)
    # print(tiffs[1])
    ndvi_mask(NDVI_THRESHOLD, tiffs)
