import rasterio

red_filename = "./S2B_MSIL2A_20230606T085559_N0509_R007_T35TNH_20230606T104740.SAFE/GRANULE/L2A_T35TNH_A032637_20230606T090233/IMG_DATA/R10m/T35TNH_20230606T085559_B04_10m.jp2"
nir_filename = "./S2B_MSIL2A_20230606T085559_N0509_R007_T35TNH_20230606T104740.SAFE/GRANULE/L2A_T35TNH_A032637_20230606T090233/IMG_DATA/R10m/T35TNH_20230606T085559_B08_10m.jp2"


red = rasterio.open(red_filename)
nir = rasterio.open(nir_filename)


with rasterio.open('NDVI.tiff','w',driver='Gtiff',
                    width=red.width, height=red.height, count=1, crs=red.crs,transform=red.transform, dtype=red.dtypes[0]) as ndvi:
    red_val = red.read(1)
    nir_val = nir.read(1)

    ndvi_val = (nir_val - red_val)/(red_val + nir_val)
    print(ndvi_val)
    ndvi.write(ndvi_val,1) 
    ndvi.close()

red.close()
nir.close()
