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
#This polygon approximately encompasses an area in and around the territory of Bulgaria.
kmlFile = "<fileName>"
srcDS = gdal.OpenEx(kmlFile)
ds = gdal.VectorTranslate('farm.geojson', srcDS, format='GeoJSON')
#Dereference and close dataset so the file is written
del ds 

#This specifies the vector polygon which we use to specify the geolocation of the images.
#This polygon approximately encompasses an area in and around the territory of Bulgaria.
footprint = geojson_to_wkt(read_geojson('farm.geojson'))

#This line queries Sentinel-2 images which follow cetain criteria and converts the
# resulting dictionary with their properties into a Pandas dataframe
products_df = api.to_dataframe(api.query (footprint, date= (date(2023, 4, 1), date(2023, 4, 1) + 25 * hours),
                                          platformname='Sentinel-2', cloudcoverpercentage=(0, 50)))
if products_df.empty:
    raise Exception("No Sentinel 2 data was found in area")

print (products_df)

#This loop goes through the images received from the query and outputs the id,
# footprint, and size of all images with size greater than 700 MB.
# This is done because smaller images are usually obscured and thus less useful.

j = 0
for i in products_df['size']:
    p = products_df.iloc[j]
    if(np.float64((i[0:3])) > 700 or i [4] == 'G' or i [5] == 'G'):
        print(products_df.iloc[j]['uuid'])
        print (products_df.iloc[j]['gmlfootprint'])
        print(products_df.iloc[j]['size'])
    j = j + 1

#This line queries images from the Sentinel-1 satellite
products2 = api.query (footprint, date= (date(2023, 5, 1), date(2023, 5, 1) + 10* hours),
                                           platformname='Sentinel-1')
#Convert to pandas dataframe
products_df2 = api.to_dataframe(products2)
                                         

if products_df2.empty:
    raise Exception("No Sentinel 1 data was found in area")

#This loop outputs some of their properties
j = 0
print(products_df2.columns)
for i in products_df2['size']:
    p = products_df2.iloc[j]
    print (p ['title'])
    print (p ['uuid'])
    print(p['footprint'])
    print(p['gmlfootprint'])
    print(i)
    print ('-------------------------------------------------------------------------------------------------------------------------')
    j += 1


#api.download_all(products2)
#A list of the names of all properties of the images
print (products_df2.columns)
#Sample code for downloading a specific image by its id
api.download ('4549bd87-ef14-4e72-a3c7-c3adb6239e14')
