from engine.sts import sentimeseries

import os
from edges.delineate import CropDelineation

# datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/Example_Data/"
# #datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/2020/"
# AOI = "/home/tars/Desktop/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"

datapath = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/Sentinel-2/Example_Data"
corinepath = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/corine_2018/greece_2018_corine.shp"
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
parcels = CropDelineation(eodata, datapath, corinepath)
# corine and scl masks
parcels.town_mask(write=False)
parcels.cloud_mask(write=False)

# edge intensity (0-100) map
parcels.edge_intensity(write=True)

# ndvi series
parcels.ndviseries(write=False)
# fill values removed by cloud mask
parcels.interpolate_series(
    parcels.ndviseries,
    parcels.ndviseries_meta,
    outfname=os.path.join(parcels.epm_path, 'crop_prob_map.tif'))
