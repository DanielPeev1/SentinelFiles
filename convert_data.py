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

dir = "./data-s2"
tempZip = "./unzip"
boundry = "./farm.geojson"
cropped_data_folder = "./cropped-ndvi-s2"
tempFile = "temp.tiff"
ndvi = True

fileNames = os.listdir(dir)
eps = 1e-5

def generateRGB(red_file, green_file, blue_file, destFile):
    red = rasterio.open(red_file) 
    green = rasterio.open(green_file) 
    blue = rasterio.open(blue_file) 
    
    with rasterio.open(destFile,'w',driver='GTiff',
                    width=red.width, height=red.height, count=3,
                    crs=red.crs,transform=red.transform, dtype=red.dtypes[0]) as rgb:
        rgb.write(blue.read(1),3) 
        rgb.write(green.read(1),2) 
        rgb.write(red.read(1),1) 
        rgb.close()
   

def generateNDVI(red_file, nir_file, destFile):
    red = rasterio.open(red_file) 
    nir = rasterio.open(nir_file) 

    with rasterio.open(destFile,'w',driver='GTiff',
                    width=red.width, height=red.height, count=1,
                    crs=red.crs,transform=red.transform, dtype=red.dtypes[0]) as ndvi:
        red_val = red.read(1)
        nir_val = nir.read(1)

        ndvi_val = (nir_val - red_val)/(red_val + nir_val + eps)
        ndvi.write(ndvi_val,1) 
        ndvi.close()

def crop(sourceFileName, destinationFile):
    # boundary for the field in Varna
    boundary = gpd.read_file(boundry)
    bound_crs = boundary.to_crs('epsg:32635')

    with rasterio.open(sourceFileName) as src:
        # uses the boundary and source image to crop the field
        out_image, out_transform = mask(src,
            bound_crs.geometry,crop=True)
        out_meta = src.meta.copy()
        out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                           "photometric": "RGB",
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
    acquisiotionDate = fileName[11:19]
    fileName = tempZip + "/" + fileName.split('.')[0] + ".SAFE"
    granuleFolder = fileName + "/GRANULE"
    imgDataFolder = granuleFolder + "/" + os.listdir(granuleFolder)[0] + "/IMG_DATA/R10m" 
    redImg = ''
    nirImg = ''
    greenImg = ''
    blueImg = ''
    for imgName in os.listdir(imgDataFolder):
        if re.match("^.*B04_10m.jp2$", imgName):
            redImg = imgDataFolder + "/" + imgName
        elif re.match("^.*B08_10m.jp2$", imgName):
            nirImg = imgDataFolder + "/" + imgName
        elif re.match("^.*B02_10m.jp2$", imgName):
            blueImg = imgDataFolder + "/" + imgName
        elif re.match("^.*B03_10m.jp2$", imgName):
            greenImg = imgDataFolder + "/" + imgName

    destinationFile = cropped_data_folder + "/" + acquisiotionDate + ".tiff"
    if ndvi:
        generateNDVI(redImg, nirImg, tempFile)
    else:
        generateRGB(redImg, blueImg, greenImg, tempFile)

    crop(tempFile, destinationFile)

    if not(ndvi):
        src = rasterio.open(destinationFile)
        fig, ax = plt.subplots(1, figsize=(6,6))
        plt.title(acquisiotionDate)
        plot.show(src, adjust='linear', ax=ax)
        plt.savefig("./img/" + acquisiotionDate + ".jpg")
        plt.close()
        shutil.rmtree(fileName)
    

