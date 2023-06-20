'''Copyright 2023 Daniel Peev <danipeev1@gmail.com>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.''' 

from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date, timedelta
import numpy as np
import pandas as pd
import rasterio
import os
from zipfile import ZipFile
import geopandas as gpd
from rasterio.mask import mask

# In this line you have to enter your username and password for the Copernicus system
api = SentinelAPI('your_username', 'your_password', 'https://apihub.copernicus.eu/apihub')


# This specifies the vector polygon which we use to specify the geolocation of the images.
# This polygon approximately encompasses an area of a field growing corn.
footprint = geojson_to_wkt(read_geojson('TsarevitsiPolygon.geojson'))

# This line queries Sentinel-2 images which follow cetain criteria and converts the
# resulting dictionary with their properties into a Pandas dataframe
products_df = api.to_dataframe(api.query(footprint, date=(date(2023, 1, 1), date(2023, 6, 19)),
                                         platformname='Sentinel-2', cloudcoverpercentage=(0, 50)))
print(products_df)

# This loop goes through the images received from the query and appends
# ordered lists with the the id, size, and beginposition i.e. the date and time of all
#  images with size greater than 700 MB to the list sentinel2list.
# This is done because smaller images are usually obscured and thus less useful.

j = 0

sentinel2list = []
print(products_df.size)
for i in products_df['size']:
    p = products_df.iloc[j]
    if (np.float64((i[0:3])) > 700 or i[4] == 'G' or i[5] == 'G'):
        sentinel2list.append(
            [products_df.iloc[j]['uuid'], products_df.iloc[j]['size'], products_df.iloc[j]['beginposition'],
             products_df.iloc[j]['title']])
    j = j + 1

# This line queries images from the Sentinel-1 satellite and
# converts the result into a Pandas Dataframe
products_df2 = api.to_dataframe(api.query(footprint, date=(date(2023, 1, 1), date(2023, 6, 19)),
                                          platformname='Sentinel-1'))

# This loop appends ordered lists containing the id, size, and date
# of Sentinel-1 images to the list sentinel1list
j = 0
sentinel1list = []
print(products_df2.columns)
print(products_df2.size)
for i in products_df2['size']:
    p = products_df2.iloc[j]
    sentinel1list.append([p['uuid'], p['size'], p['beginposition'], p['title']])
    j += 1

# A list of the names of all properties of the images
print(products_df2.columns)

# The lists of Sentinel-2 and Sentinel-1 images
print(len(sentinel2list))
print(len(sentinel1list))

#This function is used in the conversion of Sentinel-1 images
#to rgb format images of our desired polygon
def scaleToRGB(val):
    max_val = np.max(val)
    min_val = np.min(val)
    return (val - min_val) * 255 / (max_val - min_val)


# This loop goes through all ordered pairs of Sentinel-1
# and Sentinel-2 images, takes the ones taken within at most
# 4 days of each other, converts only the portion of the Sentinel-1
# images which are covered by our desired polygon to an rgb image
# and adds the numpy array representing them to the inputs list.
# The loop also calculates the ndvi index for each pixel in the Sentinel-2 image
# under the selected polygon and stores the numpy array in the labels list
# The four day window is chosen is done because over a longer
# period greater changes may occur to the observed land and thus comparisons
# between Sentinel-1 and Sentinel-2 images may become more impractical.

#The Sentinel-1 rgb images
inputs = []
#The ndvi indices of the corresponding Sentinel-2 images
labels = []

k = 0
for image1 in sentinel1list:
    for image2 in sentinel2list:

        if (abs(image1[2] - image2[2]) < timedelta(days=4)):
            #This is the most useful Sentinel-1 image format
            if (image1[3][4:11] == 'IW_GRDH'):

                api.download(id=image1[0], directory_path="D:\Sentinel1")
                os.chdir('D:\Sentinel1')
                #We extract the folder from the zip file
                ZipFile(image1[3] + '.zip').extractall(os.curdir)
                os.chdir(image1[3] + '.SAFE\measurement')
                measurements = os.listdir(os.curdir)
                #The first two bands are taken
                co_pol_file = measurements[0]
                cross_pol_file = measurements[1]

                co_pol = rasterio.open(co_pol_file)
                cross_pol = rasterio.open(cross_pol_file)
                eps = 1e-5

                # Gets information like width and height from the co_pol image and says the new file will be rgb with 3 channels
                gcps, gcp_crs = co_pol.gcps
                # affine_transform and gcp are ways to map coordinates from SAR image to map. This converts one to the other
                affine_transform = rasterio.transform.from_gcps(gcps)

                # setting metadata/profile data for the tiff file we are about to create
                profile = co_pol.profile
                profile["photometric"] = "RGB"
                # number of bands. They are three cause of RGB
                profile["count"] = 3
                # coordinate referencing system
                profile["crs"] = gcp_crs
                profile["driver"] = "GTiff"
                profile["transform"] = affine_transform
                with rasterio.open(r'D:\color_sar' + str(k) + '.tiff', 'w', **profile) as rgb:
                    cross_pol_val = scaleToRGB(cross_pol.read(1))
                    co_pol_val = scaleToRGB(co_pol.read(1))
                    ratio = scaleToRGB(cross_pol_val // (co_pol_val + eps))

                    rgb.write(ratio, 3)
                    rgb.write(co_pol_val, 2)
                    rgb.write(cross_pol_val, 1)
                    rgb.close()

                # boundary for the field in Varna
                boundary = gpd.read_file(r'D:/PythonProjects/TsarevitsiPolygon.geojson')
                bound_crs = boundary.to_crs({'init': 'epsg:4326'})

                with rasterio.open(r'D:\color_sar' + str(k) + '.tiff') as src:

                    # Cutting out only the part covered by the polygon
                    out_image, out_transform = mask(src,
                                                    bound_crs.geometry, crop=True)

                inputs.append(out_image)
                os.chdir("D:\PythonProjects")
                api.download(image2[0], directory_path="D:/Sentinel2")

                os.chdir('D:\Sentinel2')
                ZipFile(image2[3] + '.zip').extractall(os.curdir)
                os.chdir(image2[3] + '.SAFE\GRANULE\\')
                os.chdir(os.listdir(os.curdir)[0] + '\IMG_DATA')
                #This has to do with the specific file hierarchy of Sentinel-2 image folders
                if (os.listdir(os.curdir)[0] == "R10m"):
                    os.chdir("R10m")
                    measurements = os.listdir(os.curdir)
                    red_file = rasterio.open(measurements[3])
                    near_infrared_file = rasterio.open(measurements[4])
                else:
                    measurements = os.listdir(os.curdir)
                    red_file = rasterio.open(measurements[3])
                    near_infrared_file = rasterio.open(measurements[7])
                # We need the boundary in a different coordinate system for the Sentinel-2 images
                boundary = gpd.read_file(r'D:/PythonProjects/TsarevitsiPolygon.geojson').to_crs(32635)
                # Cutting out only the part covered by the polygon
                out_red, out_red_transform = mask(red_file, boundary.geometry, crop=True)
                out_near_infrared, out_near_infrared_transform = mask(near_infrared_file, boundary.geometry, crop=True)
                labels.append(out_image)
                k += 1
                break

