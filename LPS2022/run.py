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
eodata.sort_images(date=True)

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

# create obj
crop_map = eu.CropDelineation(eodata, datapath)

# compute edge intensity (0-100) map
crop_map.edge_intensity()

# cloud mask
crop_map.cloud_mask(write=True)

# crop intensity map
crop_map.mask_ndviseries(write=True)


print('bla')