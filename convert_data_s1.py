import os
from zipfile import ZipFile
from tqdm import tqdm
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

dir = "./data-s1"
tempZip = "./unzip"
boundry = "./farm.geojson"
cropped_raw_folder = "./cropped-raw-s1"
tempFile = "temp.tiff"
ndvi = True

fileNames = os.listdir(dir)
eps = 1e-5


def crop(sourceFileName, destinationFile):
    # boundary for the field in Varna
    boundary = gpd.read_file(boundry)
    bound_crs = boundary.to_crs('epsg:4326')
    with rasterio.open(sourceFileName, "r+") as src:

        gcps, gcp_crs = src.gcps
        affine_transform = rasterio.transform.from_gcps(gcps)
        src.crs = gcp_crs
        src.transform = affine_transform
        # uses the boundary and source image to crop the field
        out_image, out_transform = mask(src, bound_crs.geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(destinationFile, "w", **out_meta) as final:
        final.write(out_image)

def extract(fileName, dir):
    with ZipFile(fileName) as zip:
        zip.extractall(dir)
    zip.close()

for fileName in tqdm(fileNames):
    zipFileName = dir + "/" + fileName
    extract(zipFileName, tempZip)
    acquisiotionDate = fileName[17:25]
    fileName = tempZip + "/" + fileName.split('.')[0] + ".SAFE"
    measurements = fileName + "/measurement"
    destinationDir = cropped_raw_folder + "/" + acquisiotionDate
    measurementsFiles = os.listdir(measurements)
    if not os.path.exists(destinationDir):
        os.mkdir(destinationDir)

    for m in measurementsFiles:
        crop(measurements + "/" + m, destinationDir + "/" + m)
    
    shutil.rmtree(fileName)

