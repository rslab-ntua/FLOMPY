from engine.sts import sentimeseries

datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/Example_Data/"
AOI = "/home/tars/Desktop/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"

# Get data
eodata = sentimeseries("S2-timeseries")
eodata.find(datapath)
eodata.sort_images(date = True)

# Get VIs
eodata.getVI("NDVI")
eodata.getVI("NDMI")

# Clip data
eodata.clipbyMask(AOI, resize = True)
eodata.clipbyMask(AOI, band = "NDMI", resize = True)
eodata.clipbyMask(AOI, band = "NDVI")

# Show
eodata.show_metadata()
