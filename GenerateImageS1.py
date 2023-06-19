import rasterio
import sys
import matplotlib.pyplot as plt
from rasterio import plot
from rasterio.mask import mask
import numpy as np
import geopandas as gpd


# scales array values from 0 to 255
def scaleToRGB(val):
    max_val = np.max(val)
    min_val = np.min(val)
    return (val - min_val) * 255/(max_val - min_val)

# co-polarization file such as hh or vv
co_pol_file = "./S1A_IW_GRDH_1SDV_20230507T160047_20230507T160112_048430_05D353_F6A5.SAFE/measurement/s1a-iw-grd-vh-20230507t160047-20230507t160112-048430-05d353-002.tiff"
# cross polarization file such as hv or vh
cross_pol_file = "./S1A_IW_GRDH_1SDV_20230507T160047_20230507T160112_048430_05D353_F6A5.SAFE/measurement/s1a-iw-grd-vv-20230507t160047-20230507t160112-048430-05d353-001.tiff"

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
with rasterio.open('color_sar.tiff','w', **profile) as rgb:
    cross_pol_val = scaleToRGB(cross_pol.read(1))
    co_pol_val = scaleToRGB(co_pol.read(1))
    ratio = scaleToRGB(cross_pol_val//(co_pol_val + eps))

    rgb.write(ratio,3) 
    rgb.write(co_pol_val,2) 
    rgb.write(cross_pol_val,1) 
    rgb.close()

# boundary for the field in Varna
boundary = gpd.read_file(r'./varna.geojson')
bound_crs = boundary.to_crs({'init': 'epsg:4326'})

with rasterio.open('color_sar.tiff') as src:
    # uses the boundary and source image to crop the field
    out_image, out_transform = mask(src,
        bound_crs.geometry,crop=True)
    out_meta = src.meta.copy()
    out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "photometric": "RGB",
                 "transform": out_transform})

with rasterio.open("cropped_field.tiff", "w", **out_meta) as final:
    final.write(out_image)

src = rasterio.open(r'./cropped_field.tiff')
plt.figure(figsize=(6,6))
plt.title('Final Image')
plot.show(src, adjust='linear')
