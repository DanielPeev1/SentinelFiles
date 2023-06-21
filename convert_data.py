import os
from zipfile import ZipFile
import shutil
import rasterio
import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.mask import mask
import geopandas as gpd
import re
import numpy as np


# This script calculates the NDVI values and then crops the boundry and saves it into cropped_data_folder
# The generated files have as a name the date the data was collected

dir = "./data-s2"
tempZip = "./unzip"
boundry = "./farm.geojson"
cropped_data_folder = "./cropped-data-s2"

fileNames = os.listdir(dir)
eps = 1e-5


def crop(red_file, nir_file, destinationFile):
    red = rasterio.open(red_file) 
    nir = rasterio.open(nir_file) 

    with rasterio.open('ndvi.tiff','w',driver='GTiff',
                    width=red.width, height=red.height, count=1,
                    crs=red.crs,transform=red.transform, dtype=red.dtypes[0]) as ndvi:
        red_val = red.read(1)
        nir_val = nir.read(1)

        ndvi_val = (nir_val - red_val)/(red_val + nir_val + eps)
        ndvi.write(ndvi_val,1) 
        ndvi.close()

    # boundary for the field in Varna
    boundary = gpd.read_file(boundry)
    bound_crs = boundary.to_crs('epsg:32635')

    with rasterio.open('ndvi.tiff') as src:
        # uses the boundary and source image to crop the field
        out_image, out_transform = mask(src,
            bound_crs.geometry,crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(destinationFile + ".tiff", "w", **out_meta) as final:
        final.write(out_image)

def extract(fileName, dir):
    with ZipFile(fileName) as zip:
        zip.extractall(dir)
    zip.close()

for fileName in fileNames:
    zipFileName = dir + "/" + fileName
    extract(zipFileName, tempZip)
    acquisiotionDate = fileName[11:19]
    fileName = tempZip + "/" + fileName.split('.')[0] + ".SAFE"
    granuleFolder = fileName + "/GRANULE"
    imgDataFolder = granuleFolder + "/" + os.listdir(granuleFolder)[0] + "/IMG_DATA/R10m" 
    redImg = ''
    nirImg = ''
    for imgName in os.listdir(imgDataFolder):
        if re.match("^.*B04_10m.jp2$", imgName):
            redImg = imgDataFolder + "/" + imgName
        if re.match("^.*B08_10m.jp2$", imgName):
            nirImg = imgDataFolder + "/" + imgName

    destinationFile = cropped_data_folder + "/" + acquisiotionDate
    crop(redImg, nirImg, destinationFile)
    shutil.rmtree(fileName)
    

