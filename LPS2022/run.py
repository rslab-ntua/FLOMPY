from engine.sts import sentimeseries

#datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/Example_Data/"
datapath = "/home/tars/Desktop/RSLab/FLOMPY/Data/Sentinel-2/2020/"
AOI = "/home/tars/Desktop/RSLab/FLOMPY/Data/AOI/Flompy_ianos_aoi.geojson"

# Get data
eodata = sentimeseries("S2-timeseries")
#eodata.find_zip(datapath)
# Show
eodata.show_metadata()
print(eodata.total)

eodata.find(datapath)
print(eodata.total)
print(eodata.tiles)
eodata.sort_images()

# Get VIs
#eodata.getVI("NDVI")
#eodata.getVI("NDMI")

# Clip data
#eodata.clipbyMask(AOI, resize = True)
#eodata.clipbyMask(AOI, band = "NDMI", resize = True)
#eodata.clipbyMask(AOI, band = "NDVI")

# Show
#eodata.show_metadata()
