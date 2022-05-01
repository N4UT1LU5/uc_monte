import rasterio
import numpy

# https://gis.stackexchange.com/questions/138914/calculating-ndvi-with-rasterio

numpy.seterr(divide="ignore", invalid="ignore")

im = rasterio.open("./input/CIR16_orthomosaic_2.tif")
bandNIR = im.read(1)
bandRed = im.read(2)


# im = rasterio.open("./input/CIR17_orthomosaic.tif")

ndvi = numpy.zeros(im.shape, dtype=rasterio.float32)
ndvi = (bandNIR.astype(float) - bandRed.astype(float)) / (
    bandNIR.astype(float) + bandRed.astype(float)
)
# veg = ndvi
# ndvi[ndvi > 0.4] = 1.0
# ndvi[ndvi <= 0.4] = 0.0
print("checkpoint")


kwargs = im.meta
kwargs.update(dtype=rasterio.float32, count=1, compress="lzw")

with rasterio.open("./output/ndvi.tif", "w", **kwargs) as dst:
    dst.write_band(1, ndvi.astype(rasterio.float32))
# with rasterio.open("./output/veg.tif", "w", **kwargs) as dst:
#     dst.write_band(1, veg.astype(rasterio.float32))
# print(im.bounds)
