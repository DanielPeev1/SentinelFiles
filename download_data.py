from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date, timedelta
import numpy as np
import pandas as pd
import sys, getopt
from osgeo import gdal, ogr

# A script that is used to more easily download sentinel data from the command line
# format is python3 dowdload_data.py -f <footprint> -p <Sentinel-1/2> --start DD/MM/YYYY
#    --end DD/MM/YYYY -t <productType> -d <destination directory>

argv = sys.argv[1:]

# In this line you have to enter your username and password for the Copernicus system
api = SentinelAPI('<username>', '<pass>', 'https://apihub.copernicus.eu/apihub')

opts, args = getopt.getopt(argv,"hp:f:t:d:",["end=","start="])
start = []
end = []
file = ''
platform = ''
ptype = ''
dir = '.'

for opt, arg in opts:
  if opt == '-f':
    file = arg
  elif opt in ("-p", "--platform"):
    platform = arg
  elif opt in ("--start"):
    start = arg.split("/")
  elif opt in ("--end"):
    end = arg.split("/")
  elif opt in ("-t"):
    ptype = arg
  elif opt in ("-d"):
    dir = arg

if len(end) == 0: 
   raise ValueError("You need to specify until when you need the data. Use --end d/m/y format")

if len(start) == 0: 
   raise ValueError("You need to specify until when you need the data. Use --start d/m/y format")

if file == '': 
   raise ValueError("Please pass a kml or geojson file to specify where to look for images with -f <file>")

if platform == '': 
   raise ValueError("Please specify if you want to use Sentinel-1 or 2 with -p <platform name>")

# This specifies the vector polygon which we use to specify the geolocation of the images.
# This polygon approximately encompasses an area in and around the territory of Bulgaria.
if file.endswith("kml"):
    srcDS = gdal.OpenEx(file)
    newFileName = file.partition(".")[0] + '.geojson'
    ds = gdal.VectorTranslate(newFileName, srcDS, format='GeoJSON')
    #Dereference and close dataset so the file is written
    del ds 
    file = newFileName

# This specifies the vector polygon which we use to specify the geolocation of the images.
# This polygon approximately encompasses an area in and around the territory of Bulgaria.
footprint = geojson_to_wkt(read_geojson(file))

products = []

# This line queries Sentinel images which follow cetain criteria and later downloads them
if platform == "Sentinel-2":
  products = api.query(footprint, date= (date(int(start[2]), int(start[1]), int(start[0])), date(int(end[2]), int(end[1]), int(end[0]))),
                                          platformname=platform, producttype=ptype, cloudcoverpercentage=(0, 90))
else: 
  products = api.query(footprint, date= (date(int(start[2]), int(start[1]), int(start[0])), date(int(end[2]), int(end[1]), int(end[0]))),
                                          platformname=platform, producttype=ptype)
api.download_all(products, dir)