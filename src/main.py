import rasterio
import numpy

# https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

INPUT_PATH = "./input/CIR16_orthomosaic_2.tif"
OUTPUT_PATH = "./output/veg.tif"

NDVI_THRESHOLD = 0.5

numpy.seterr(divide="ignore", invalid="ignore")


def ndvi_mask(threshold, path_to_img):
    """
    create tiff mask of ndvi threshold
    """
    im = rasterio.open(path_to_img)
    bandNIR = im.read(1)
    bandRed = im.read(2)

    ndvi = numpy.zeros(im.shape, dtype=rasterio.float32)
    ndvi = (bandNIR.astype(float) - bandRed.astype(float)) / (
        bandNIR.astype(float) + bandRed.astype(float)
    )

    veg = ndvi
    veg[veg > threshold] = 1.0
    veg[veg <= threshold] = 0.0
    print("checkpoint")

    kwargs = im.meta
    kwargs.update(dtype=rasterio.float32, count=1, compress="lzw")

    # with rasterio.open("./output/ndvi.tif", "w", **kwargs) as dst:
    #     dst.write_band(1, ndvi.astype(rasterio.float32))
    with rasterio.open(OUTPUT_PATH, "w", **kwargs) as dst:
        dst.write_band(1, veg.astype(rasterio.float32))


if __name__ == "__main__":
    ndvi_mask(NDVI_THRESHOLD, INPUT_PATH)