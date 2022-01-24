from cmath import isnan
from engine.sts import sentimeseries
import rasterio as rio

import os
import numpy as np
import pandas as pd
import multiprocessing as mp
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


# Kernels to compute diagonal spectral diference.
_conv_kern_p1 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]]) # ul
_conv_kern_p2 = np.array([[0, -1, 0], [0, 1, 0], [0, 0, 0]]) # uc
_conv_kern_p3 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]]) # ur
_conv_kern_p4 = np.array([[0, 0, 0], [0, 1, -1], [0, 0, 0]]) # cr
_conv_kern_p5 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, -1]]) # br
_conv_kern_p6 = np.array([[0, 0, 0], [0, 1, 0], [0, -1, 0]]) # cb
_conv_kern_p7 = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0]]) # bl
_conv_kern_p8 = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]) # cl
kernels = [_conv_kern_p1, _conv_kern_p2, _conv_kern_p3, _conv_kern_p4, _conv_kern_p5,
            _conv_kern_p6, _conv_kern_p7, _conv_kern_p8]

# Weight for every direction. math.sqrt(2)/2, 1, math.sqrt(2)/2, 1, math.sqrt(2)/2, 1, math.sqrt(2)/2, 1
weights = [1, 1, 1, 1, 1, 1, 1, 1]

# Pass kernels & weights to dataframe, and read them from equation() functions.
_list = [(kernels[i], weights[i]) for i in range(0, len(weights))]
kernels_N_weights = pd.DataFrame(_list, columns=['kernels', 'weights'])



# Find bands b3, b4, b8, b11, b12, ndvi
count = 0
pool = mp.Pool(mp.cpu_count() - 2)

for ms_im in eodata.data:
    src = ms_im.ReadData(band='NDVI_masked')
    metadata = src.meta
    ndvi = src.read(1)

    dndvi = pool.starmap_async(
        eu.equation_4, [(ndvi, kernels_N_weights.iloc[i]) for i in range(0, len(kernels_N_weights))]).get()

    five_bands = [getattr(ms_im, im) for im in ['B03_masked','B04_masked','B08_masked','B11_masked','B12_masked']]

    dweeks = pool.starmap_async(
        eu.equation_3, [(five_bands, kernels_N_weights.iloc[i]) for i in range(0, len(kernels_N_weights))]).get()

    edge_estim = eu.equation_5(ndvi, dweeks, dndvi, kernels_N_weights)

    edge_estim[ndvi == metadata['nodata']] = metadata['nodata']
    edge_estim[edge_estim > 100] = metadata['nodata']
    edge_estim[edge_estim < 0] = metadata['nodata']

    # Construct new filename.
    imPath = os.path.join(ms_im.datapath_10,
                        f"T{ms_im.tile_id}_{ms_im.str_datetime}_edge.tif")
    ms_im.edge = imPath

    metadata.update(count=1, dtype=edge_estim.dtype)
    with rio.open(imPath, 'w', **metadata) as dst:
        dst.write(edge_estim, 1)


# create cube using edge estimations
edge_paths = [ms_im.edge for ms_im in eodata.data]
cbarr, metadata, band_names = eu.edge_cube(edge_paths)

# Equation (6). Compute the sum of each pixel in depth.
pix_sum = np.sum(cbarr, axis=0)

# Count how many NaNs every pixel has, through the weeks
not_nan_dates = np.sum(~np.isnan(cbarr) * 1, axis=0)

res = pix_sum / not_nan_dates

print(res.min(), res.max())

# Construct new filename.
imPath = os.path.join(datapath,
                    f"edge_prob_map_daterange.tif")

metadata.update(count=1, dtype=res.dtype, nodata=0)
with rio.open(imPath, 'w', **metadata) as dst:
    dst.write(res, 1)
