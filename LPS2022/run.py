from engine.sts import sentimeseries
from edges.delineate import CropDelineation

# datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/Example_Data/"
# #datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/2020/"
# AOI = "/home/tars/Desktop/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"

datapath = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/FLOMPY_Palamas/Sentinel_2_L2A"
# datapath = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/Sentinel-2/Example_Data"
corinepath = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/corine_2018/greece_2018_corine.shp"
AOI = "/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"
unet_pred_path='/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/Data/Sentinel-2/UNet_example_result/UNet3_crop_delineation.tif'
flood_tif_path='/mnt/a202d601-6efc-44f7-8408-f8322b69b445/RSLab/FLOMPY/FLOMPY_Palamas/Sentinel_1_GRD_imagery/Flood_map.tif'


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
# create corine and scl masks
parcels.town_mask(write=False)
parcels.cloud_mask(write=False)

# create edge intensity map and apply any created mask (clouds, towns)
parcels.edge_probab_map(write=True)

# create ndvi series and apply any created mask (clouds, towns)
parcels.create_series(write=False)
# fill values removed by cloud mask
parcels.crop_probab_map(
    cube = parcels.ndviseries,
    cbmeta = parcels.ndviseries_meta,
    write=True,
    )

# edges, active and inactive fields Map
parcels.active_fields()


# Delineate fields: Combine EPM and UNet
parcels.delineation(AOI, unet_pred_path)

# Characterize Cultivated and Not-Cultivated fields
parcels.flooded_fields(flood_tif_path)