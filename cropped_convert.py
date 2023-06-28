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
# The generated files have as a name the date the data was collected. This script unlike convert_data_s2 uses the cropped_raw_s2 data and not
# the raw s2 images from sentinel

dir = "./cropped-raw-s2"
tempZip = "./unzip"
boundry = "./farm.geojson"
cropped_ndvi_folder = "./cropped-ndvi-s2"
cropped_raw_folder = "./cropped-raw-s2"
tempFile = "temp.tiff"
ndvi = True

fileNames = os.listdir(dir)
eps = 1e-5

def generateNDVI(red_file, nir_file, destFile):
    red = rasterio.open(red_file) 
    nir = rasterio.open(nir_file) 

    with rasterio.open(destFile,'w',driver='GTiff',
                    width=red.width, height=red.height, count=1,
                    crs=red.crs,transform=red.transform, dtype="float32") as ndvi:
        red_val = red.read(1)
        nir_val = nir.read(1)
        ndvi_val = (nir_val - red_val)/(red_val + nir_val)
        ndvi_val = np.nan_to_num(ndvi_val)
        ndvi.write(ndvi_val,1) 
        ndvi.close()

for fileName in tqdm(fileNames):
    acquisiotionDate = fileName
    imgDataFolder = dir + "/" + fileName
    redImg = ''
    nirImg = ''
    greenImg = ''
    blueImg = ''
    for imgName in os.listdir(imgDataFolder):
        if re.match("^.*B04_10m.jp2$", imgName):
            redImg = imgDataFolder + "/" + imgName
        elif re.match("^.*B08_10m.jp2$", imgName):
            nirImg = imgDataFolder + "/" + imgName

    destinationFile = cropped_ndvi_folder + "/" + acquisiotionDate + ".tiff"
    generateNDVI(redImg, nirImg, destinationFile)