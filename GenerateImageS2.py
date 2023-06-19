import rasterio
import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.mask import mask
import numpy as np
import geopandas as gpd


# Source for color bands: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi/resolutions/spatial
# red file is B04
red_file = "./S2B_MSIL2A_20230606T085559_N0509_R007_T35TNH_20230606T104740.SAFE/GRANULE/L2A_T35TNH_A032637_20230606T090233/IMG_DATA/R10m/T35TNH_20230606T085559_B04_10m.jp2"
# green file is B03
green_file = "./S2B_MSIL2A_20230606T085559_N0509_R007_T35TNH_20230606T104740.SAFE/GRANULE/L2A_T35TNH_A032637_20230606T090233/IMG_DATA/R10m/T35TNH_20230606T085559_B03_10m.jp2"
# blue file is B02
blue_file = "./S2B_MSIL2A_20230606T085559_N0509_R007_T35TNH_20230606T104740.SAFE/GRANULE/L2A_T35TNH_A032637_20230606T090233/IMG_DATA/R10m/T35TNH_20230606T085559_B02_10m.jp2"


red = rasterio.open(red_file) 
green = rasterio.open(green_file) 
blue = rasterio.open(blue_file) 

with rasterio.open('s2_color.tiff','w',driver='GTiff',
                    width=red.width, height=red.height, count=3, crs=red.crs,transform=red.transform, dtype=red.dtypes[0]) as rgb:
    rgb.write(blue.read(1),3) 
    rgb.write(green.read(1),2) 
    rgb.write(red.read(1),1) 
    rgb.close()

# boundary for the field in Varna
boundary = gpd.read_file(r'./varna.geojson')
bound_crs = boundary.to_crs({'init': 'epsg:32635'})

with rasterio.open('s2_color.tiff') as src:
    # uses the boundary and source image to crop the field
    out_image, out_transform = mask(src,
        bound_crs.geometry,crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "photometric": "RGB",
                 "transform": out_transform})

with rasterio.open("s2_cropped.tiff", "w", **out_meta) as final:
    final.write(out_image)

src = rasterio.open(r'./s2_cropped.tiff')
plt.figure(figsize=(6,6))
plt.title('Final Image')
plot.show(src, adjust='linear')
