import os
import subprocess

import numpy as np
import rasterio
from rasterio.warp import reproject, calculate_default_transform, Resampling
from matplotlib import pyplot as plt
import os

def convert(src, ndvi, destinationFile):
    with rasterio.Env():
        dst_crs = ndvi.crs
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        transform = ndvi.transform
        width = ndvi.width
        height = ndvi.height
        t, w, h = calculate_default_transform(
            ndvi.crs, dst_crs, ndvi.width, ndvi.height, *ndvi.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rasterio.open(destinationFile, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)


rawDir = "./cropped-raw-s1/"
dateFiles = os.listdir(rawDir)

ndvi = rasterio.open("./cropped-ndvi-s2/20210115.tiff")

for dateFile in dateFiles:
    images = os.listdir(rawDir + dateFile)
    for img in images:
        sarImg = rasterio.open(rawDir + dateFile + "/" + img, 'r')
        if not os.path.exists("./resized_cropped-s1/" + dateFile):
            os.mkdir("./resized_cropped-s1/" + dateFile)
        convert(sarImg, ndvi, "./resized_cropped-s1/" + dateFile + "/" + img)

