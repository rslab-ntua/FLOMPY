from engine.sts import sentimeseries

import gc
from edges import edge_utils as eu

# datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/Example_Data/"
# #datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/2020/"
# AOI = "/home/tars/Desktop/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"

datapath = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/Sentinel-2/Example_Data"
AOI = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"

# Get data
eodata = sentimeseries("S2-timeseries")
#eodata.find_zip(datapath)
eodata.find(datapath)
eodata.sort_images()

# Get VIs
eodata.getVI("NDVI")
eodata.getVI("NDMI")

# Clip data
eodata.clipbyMask(AOI, resize = True)
eodata.clipbyMask(AOI, band = "NDMI", resize = True)
eodata.clipbyMask(AOI, band = "NDVI")

eodata.remove_orbit("136")

# Show
eodata.show_metadata()

# TODO: Add resampled-clipped SCL image as attribute
# TODO: create an scl-cloud-mask for each date, as 2D array, based on values [0, 1, 2, 3, 8, 9, 10, 11]

epm = eu.EPM(eodata, datapath)

print('bla')