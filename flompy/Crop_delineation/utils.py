import os
import math
import numpy as np
import pandas as pd
import rasterio as rio
import geopandas as gpd
from scipy import signal
import requests
from requests import exceptions
import json
import shapely.wkt

def equation_3(fpaths:list, kernel_N_weight:pd.DataFrame)->list:
    """Square root of gradient between 2 pixels, in selected orientantion, for entire image.
    Equation (3) of original paper. After the convolution, the result image has a pad which must be
    ignored to fearther computations.
    For all bands in an fpaths, for one kernel and corresponding weight. All kernels are flipped
    before convolution.
    Args:
        fpaths (list): List of fullpaths for bands b03, b04, b08, b11, b12.
        kernel_N_weight (pd.DataFrame): Row of iterable dataframe with 2 columns with kernels
            and their weights. 
    Returns:
        list: List containing 2D arrays, one for each kernel. Result for one week (or date),
        for one pixel (orientation of convolution).
    """
    res = []
    for im_path in fpaths:
        with rio.open(im_path) as src:
            b = src.read(1)

            # TODO: mask with scl-cloud-mask
            grad = signal.convolve2d(
                b,
                kernel_N_weight['kernels'],
                mode='same',
                boundary='fill',
                fillvalue=0)
            pow_grad = np.power(grad, 2)
            res.append(pow_grad)
    dweek_onePixel = np.sqrt(sum(res))
    return dweek_onePixel


def equation_4(ndvi:np.array, kernel_N_weight:pd.DataFrame)->list:
    """Convolution only for NDVI image.
    Args:
        ndvi (np.array): Image as 2D array.
        kernel_N_weight (pd.DataFrame): Row of iterable dataframe with 2 columns with kernels
            and their weights. 
    Returns:
        list: List containg 2D arrays, one for each kernel.
    """
    abs_grad = np.absolute(signal.convolve2d(ndvi,
                                            kernel_N_weight['kernels'],
                                            mode='same',
                                            boundary='fill',
                                            fillvalue=0))
    return abs_grad


def equation_5(ndvi:np.array, dweeks:list, dndvi:list, kernels_N_weights:pd.DataFrame)->np.array:
    """Compute edge estimation for one date.
    Args:
        ndvi (np.array): NDVI image as 2D array.
        dweeks (list): Result of equation_3. List containing 2D arrays, one for each kernel.
        dndvi (list): Result of equation_4. List containg 2D arrays, one for each kernel.
        kernels_N_weights (pd.DataFrame): Row of iterable dataframe with 2 columns with kernels
            and their weights.
    Returns:
        np.array: Edge estimation image as 2D array.
    """
    # Compute every convolutioned image with it's corresponding weight.
    bands = []
    indices = []
    for i in range(0, len(kernels_N_weights)):
        temp01 = dweeks[i] * kernels_N_weights['weights'][i]
        temp02 = dndvi[i] * kernels_N_weights['weights'][i]
        bands.append(temp01)
        indices.append(temp02)
    # Sum of weights.
    sw = sum(kernels_N_weights['weights'])
    # Return their multiply, with ndvi as the most important contributor.
    return (ndvi**2) * (sum(bands)/sw) * (sum(indices)/sw)


def wkernels()->pd.DataFrame:
    """[summary]
    Returns:
        pd.DataFrame: [description]
    """
    # Kernels to compute spectral diference.
    _conv_kern_p1 = np.array([[-1,  0,  0], [ 0, 1,  0], [ 0,  0,  0]]) # ul
    _conv_kern_p2 = np.array([[ 0, -1,  0], [ 0, 1,  0], [ 0,  0,  0]]) # uc
    _conv_kern_p3 = np.array([[ 0,  0, -1], [ 0, 1,  0], [ 0,  0,  0]]) # ur
    _conv_kern_p4 = np.array([[ 0,  0,  0], [ 0, 1, -1], [ 0,  0,  0]]) # cr
    _conv_kern_p5 = np.array([[ 0,  0,  0], [ 0, 1,  0], [ 0,  0, -1]]) # br
    _conv_kern_p6 = np.array([[ 0,  0,  0], [ 0, 1,  0], [ 0, -1,  0]]) # cb
    _conv_kern_p7 = np.array([[ 0,  0,  0], [ 0, 1,  0], [-1,  0,  0]]) # bl
    _conv_kern_p8 = np.array([[ 0,  0,  0], [-1, 1,  0], [ 0,  0,  0]]) # cl
    kernels = [_conv_kern_p1, _conv_kern_p2, _conv_kern_p3, _conv_kern_p4,
                _conv_kern_p5, _conv_kern_p6, _conv_kern_p7, _conv_kern_p8]
    # Weight of each direction.
    weights = [math.sqrt(2)/2, 1, math.sqrt(2)/2, 1, math.sqrt(2)/2, 1, math.sqrt(2)/2, 1]
    # kernels and their weights
    _list = [(kernels[i], weights[i]) for i in range(0, len(weights))]
    return pd.DataFrame(_list, columns=['kernels', 'weights'])


def cube_by_paths(listOfPaths:list, outfname:str=None, **kwargs)->list:
    """Concatenate images as cube.
    Args:
        listOfPaths (list): List containg fullpaths of images to concatenate on
            time-axis.
        outfname (str, optional): Absolute filename for the resulted geotif file.
            Defaults to None. When given, the 3D cube array will be saved.
    Returns:
        list: Cube as 3D np.array, cube's metadata as dict, list of strings containing
            bandnames used to produce the cube.
    """
    # read random image's metadata
    with rio.open(listOfPaths[0], 'r') as src:
        meta = src.meta

    # Preallocate a zero array with appropriate dimensions
    temp = np.zeros((1, meta['height'], meta['width']))

    # Concatenate
    band_names = []
    for bandpath in listOfPaths:
        with rio.open(bandpath, 'r') as src:
            arr = src.read()
            descr = src.name
            band_names.append(os.path.basename(descr))
        cbarr = np.concatenate([temp, arr])
        temp = cbarr

    # Update metadata. Reduce one because of temp array
    meta.update(count=cbarr.shape[0]-1)
    cbarr = cbarr.astype(meta['dtype'])

    if outfname is not None:
        assert os.path.isabs(outfname)
        with rio.open(outfname, 'w', **meta) as dst:
            for id, layer in enumerate(listOfPaths, start=1):
                with rio.open(layer) as src:
                    dst.write_band(id, src.read(1))
                    dst.set_band_description(id, band_names[id-1])

    return cbarr[1:, :, :], meta, band_names



def cbdf2cbarr(cbdf:pd.DataFrame, cbmeta:dict)->np.ndarray:
    """Convert dataframe of cube to corresponding 3D cube array.
    Args:
        cbdf (pd.DataFrame): Indexed as (rows:bands, row wise read, columns:individual pixels)
        cbmeta (dict): Containing all cube metadata, as returned by rasterio.
    Returns:
        np.ndarray: Indexed as 3D tensor (count:bands, height:rows, width:columns)
    """
    # Convert dataframe to array
    temp = cbdf.to_numpy(dtype=cbmeta['dtype'])
    # Create axis, from 2D to 3D
    temp = temp[np.newaxis, :, np.newaxis]
    # Reshape array
    cbarr = np.reshape(temp, (cbmeta['count'], cbmeta['height'],  cbmeta['width']))
    return cbarr


def cbarr2cbdf(cbarr:np.ndarray, cbmeta:dict)->pd.DataFrame:
    """Convert 3D cube array to corresponding dataframe of the cube.
    Args:
        cbarr (np.ndarray): Indexed as tensor (count:bands, height:rows, width:columns)
        cbmeta (dict): Containing all cube metadata, as returned by rasterio.
    Returns:
        pd.DataFrame: Indexed as (rows:bands, row wise read, columns:individual pixels)
    """
    # Drop array to 2D.
    temp = np.reshape(cbarr, (cbmeta['count'], cbmeta['height'] *  cbmeta['width']))
    # Convert array to dataframe.
    cbdf = pd.DataFrame(
        temp, columns=["pix_"+str(i) for i in range(0, cbmeta['height'] *  cbmeta['width'])])
    return cbdf


def filter_corine(shppath:str)->gpd.GeoDataFrame:
    corine_data = gpd.read_file(shppath)

    keep = {
        '211':'Non-irrigated arable land',
        '212':'Permanently irrigated land',
        '213':'Rice fields',
        '221':'Vineyards',
        '222':'Fruit trees and berry plantations',
        '223':'Olive groves',
        '231':'Pastures',
        '241':'Annual crops associated with permanent crops',
        '242':'Complex cultivation patterns',
        '243':'Land principally occupied by agriculture',
        '244':'Agro-forestry areas',
        # '311':'Broad-leaved forest',
        # '312':'Coniferous forest',
        # '313':'Mixed forest',
        # '321':'Natural grasslands',
        # '322':'Moors and heathland',
        # '323':'Sclerophyllous vegetation',
        # '324':'Transitional woodland-shrub',
        # '331':'Beaches',
        # '332':'Bare rocks',
        # '333':'Sparsely vegetated areas',
        # '334':'Burnt areas',
        # '335':'Glaciers and perpetual snow',
        }

    corine_data['Code_18'] = corine_data['Code_18'].astype(str)
    corine_data=corine_data[corine_data['Code_18'].isin(list(keep.keys()))]
    return corine_data


def _wkt2esri(wkt:str)->str:
    """Converts WKT geometries to arcGIS geometry strings.
    Args:
        wkt (str): WKT geometry string
    Returns:
        str: ESRI arcGIS polygon geometry string
    """
    geom = shapely.wkt.loads(wkt)
    rings = None
    # Testing for polygon type
    if geom.geom_type == 'MultiPolygon':
        rings = []
        for pg in geom.geoms:
            rings += [list(pg.exterior.coords)] + [list(interior.coords) for interior in pg.interiors]    
    elif geom.geom_type == 'Polygon':
        rings = [list(geom.exterior.coords)] + [list(interior.coords) for interior in geom.interiors]
    else:
        print("Shape is not a polygon")
        return None
            
    # Convert to esri geometry json    
    esri = json.dumps({'rings': rings})

    return esri

def corine(aoi:str, to_file:bool = False, fname:str = "corine_2018.shp")->gpd.GeoDataFrame:
    """Downloads Corine Land Cover 2018 data from Copernicus REST API.
    Args:
        aoi (str): Path to file with the region of interest
        to_file (bool, optional): Save result to file. Defaults to False
        fname (str, optional): Path and name of the created file. Defaults to "corine_2018.shp"
    Returns:
        gpd.GeoDataFrame: Corine Land Cover 2018 data
    """
    HTTP_OK = 200

    geoms = gpd.read_file(aoi).dissolve()
    polygons = list(geoms.geometry)
    wkt = f"{polygons[0]}"
    esri = _wkt2esri(wkt)
    # Build URL for retrieving data
    server = "https://image.discomap.eea.europa.eu/arcgis/rest/services/Corine/CLC2018_WM/MapServer/0/query?"
    payload = {
        "geometry": esri, 
        "f": "GeoJSON",
        "inSR": geoms.crs.to_epsg(),
        "geometryType": "esriGeometryPolygon",
        "spatialRel": "esriSpatialRelIntersects",
        "returnGeometry": True
        }
    print ("Starting retrieval...")
    request = requests.get(server, params = payload)
    # Check if server didn't respond to HTTP code = 200
    if request.status_code != HTTP_OK:
        raise exceptions.HTTPError("Failed retrieving POWER data, server returned HTTP code: {} on following URL {}.".format(request.status_code, request.url))
    # In other case is successful
    print ("Successfully retrieved data!")
    json_data = request.json()
    data = gpd.GeoDataFrame.from_features(json_data)
    if to_file:
        data.to_file(fname)
    
    return data