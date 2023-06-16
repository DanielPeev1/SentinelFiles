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

#In this line you have to enter your username and password for the Copernicus system
api = SentinelAPI('your_username', 'your_password', 'https://apihub.copernicus.eu/apihub')

#The timedelta is used to specify the time intervals for which we query Sentinel satellite images
#The Sentinel-2 satellite for the most part considers only full day intervals
hours = timedelta (hours = 24)

#This specifies the vector polygon which we use to specify the geolocation of the images.
#This polygon approximately encompasses an area of a field growing corn.
footprint = geojson_to_wkt(read_geojson('TsarevitsiPolygon.geojson'))

#This line queries Sentinel-2 images which follow cetain criteria and converts the
# resulting dictionary with their properties into a Pandas dataframe
products_df = api.to_dataframe(api.query (footprint, date= (date(2022, 6, 1), date(2023, 6, 1)),
                                          platformname='Sentinel-2', cloudcoverpercentage=(0, 50)))
print (products_df)

#This loop goes through the images received from the query and appends
# ordered lists with the the id, size, and beginposition i.e. the date and time of all
#  images with size greater than 700 MB to the list sentinel2list.
# This is done because smaller images are usually obscured and thus less useful.

j = 0

sentinel2list = []
print(products_df.size)
for i in products_df['size']:
    p = products_df.iloc[j]
    if(np.float64((i[0:3])) > 700 or i [4] == 'G' or i [5] == 'G'):

        sentinel2list.append([products_df.iloc[j]['uuid'], products_df.iloc[j]['size'], products_df.iloc[j]['beginposition']])
    j = j + 1

#This line queries images from the Sentinel-1 satellite and
# converts the result into a Pandas Dataframe
products_df2 = api.to_dataframe(api.query (footprint, date= (date(2022, 6, 1), date(2023, 6, 1) ),
                                           platformname='Sentinel-1'))

#This loop appends ordered lists containing the id, size, and date
#of Sentinel-1 images to the list sentinel1list
j = 0
sentinel1list = []
print(products_df2.columns)
print(products_df2.size)
for i in products_df2['size']:
    p = products_df2.iloc[j]
    sentinel1list.append([p['uuid'], p['size'], p['beginposition']])
    j += 1

#A list of the names of all properties of the images
print (products_df2.columns)

#The lists of Sentinel-2 and Sentinel-1 images
print (len(sentinel2list))
print(len(sentinel1list))

#This loop goes through all ordered pairs of Sentinel-1
#and Sentinel-2 images and prints the ones which have been taken
#within at most 4 days of eachother. This is done because over a longer
#period greater changes may occur to the observed land and thus comparisons
#between Sentinel-1 and Sentinel-2 images may become more impractical
k = 0
for image1 in sentinel1list:
    for image2 in sentinel2list:
        #print (abs(image1 [2] - image2 [2]))
        if (abs(image1 [2] - image2 [2])  < timedelta(days = 4)):
            #api.download(image1[0])
            # api.download(image2[0])
            print (image1)
            print (image2)
            print ("-" * 100)
            k += 1

#This prints the number of such ordered pairs
print (k)



