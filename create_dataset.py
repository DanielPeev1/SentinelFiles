# This script uses the cropped files of our field to create a dataset
# The dataset has an x of the concatenated SAR values from the VV + VH files
# The labels y are the NDVI indexes at those times
# The time delta between the images is 4 days by default
import os
import rasterio
import re
import numpy as np
from datetime import datetime

imageDelta = 4
ndviFolder = "./cropped-ndvi-s2"
ndviDates = os.listdir(ndviFolder)
s1Folder = "./cropped-raw-s1"
s1 = os.listdir(s1Folder)
datasetDir = "./dataset"


def addElement(sar, y, lastNDVIS2Data, path, dataset):
    polarisation = sar[0][11:13] 
    if polarisation == "vh":
        sar[0], sar[1] = sar[1], sar[0]

    sarVV = rasterio.open(path + "/" + sar[0]).read(1)
    sarVH = rasterio.open(path + "/" + sar[1]).read(1)
    x = np.array([sarVV, sarVH])
    dataset.append({
        "sarImage": x,
        "lastNDVI": lastNDVIS2Data,
        "y": y,
    })

s1DateTime = [(datetime.strptime(s, "%Y%m%d"), s) for s in s1]

dataset = []

ndviDates.sort()

for idx, ndvi in enumerate(ndviDates):
    ndviAcquisitionDate = ndvi.split(".")[0]
    ndviDateTime = datetime.strptime(ndviAcquisitionDate, "%Y%m%d")
    date, s1ClosestDateFolderName = min(s1DateTime, key=lambda x:abs(x[0]-ndviDateTime))

    delta = date - ndviDateTime
    if abs(delta.days) > imageDelta:
        continue

    if idx != 0:
        lastNDVIDate = ndviDates[idx - 1]
    else:
        lastNDVIDate = ndviDates[idx + 1]

    lastDate = rasterio.open(ndviFolder + "/" + lastNDVIDate)
    lastNDVIData = lastDate.read(1)

    label = rasterio.open(ndviFolder + "/" + ndvi)
    y = label.read(1)

    s1ClosestDatePath = s1Folder + "/" + s1ClosestDateFolderName
    sarFiles =  os.listdir(s1ClosestDatePath) 
    sarA = list(filter(lambda x: "s1a" in x, sarFiles))
    sarB = list(filter(lambda x: "s1b" in x, sarFiles))

    if len(sarB) != 0:
        addElement(sarB, y, lastNDVIData, s1ClosestDatePath, dataset)
    if len(sarA) != 0:
        addElement(sarA, y, lastNDVIData, s1ClosestDatePath, dataset)

np.save("./dataset", np.array(dataset))