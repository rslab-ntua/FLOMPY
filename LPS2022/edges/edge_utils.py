import os
import math
from pickletools import uint8
import numpy as np
import pandas as pd
import rasterio as rio
from scipy import signal
import multiprocessing as mp
from engine.sts import sentimeseries


def equation_3(fpaths:list, kernel_N_weight:pd.DataFrame):
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


def equation_4(ndvi:np.array, kernel_N_weight:pd.DataFrame):
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


def equation_5(ndvi:np.array, dweeks:list, dndvi:list, kernels_N_weights:pd.DataFrame):
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



class CropDelineation():
    def __init__(self, eodata:sentimeseries, epm_path:str):
        """Compute edge probability map for dates in given sentimeseries object.

        Args:
            eodata (sentimeseries): sentimeseries object, after having bands 11, 12 resamled
                to 10 meters and masked using an area of interest.
            epm_path (str): Fullpath where edge probability map is saved.

        Raises:
            TypeError: [description]
        """
        self.eodt = eodata
        self.tmp_rng = (self.eodt.dates[0].strftime('%Y%m%d'),
                            self.eodt.dates[-1].strftime('%Y%m%d'))
        self.epm_path = epm_path
        self.senbands = ['B03_masked',
                        'B04_masked',
                        'B08_masked',
                        'B11_masked',
                        'B12_masked',
                        'NDVI_masked']
        self.scl_unreliable = [1, 2, 3, 8, 9, 10, 11]

        if isinstance(self.eodt, sentimeseries):
            for i in range(len(self.eodt.data)):
                for im in self.senbands:
                    assert hasattr(self.eodt.data[i], im), f'Missing attribute {im}!'
        else:
            raise TypeError("Only sentimeseries objects are supported!")

    def estimate(self):
        def _wkernels():
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

        wk = _wkernels()
        self.estim_paths = []

        pool = mp.Pool(mp.cpu_count() - 2)
        for ms_im in self.eodt.data:
            # apply kernels on NDVI image 
            src = ms_im.ReadData(band=self.senbands[-1])
            metadata = src.meta
            ndvi = src.read(1)
            ndvi[ndvi == metadata['nodata']] = np.nan
            # TODO: mask with scl-cloud-mask
            dndvi = pool.starmap_async(
                equation_4, [(ndvi, wk.iloc[i]) for i in range(0, len(wk))]).get()

            # apply kernels on bands b03, b04, b08, b11, b12
            five_bands = [getattr(ms_im, im) for im in self.senbands[:-1]]
            dweeks = pool.starmap_async(
                equation_3, [(five_bands, wk.iloc[i]) for i in range(0, len(wk))]).get()

            # edge estimation of current data
            edge_estim = equation_5(ndvi, dweeks, dndvi, wk)

            # Cut values
            edge_estim[edge_estim > 100] = 100
            edge_estim[edge_estim < 0] = 0

            # Normalize to some limits
            up_limit = np.nanpercentile(edge_estim, 40)
            down_limit = np.nanpercentile(edge_estim, 5)
            edge_estim = (edge_estim - np.nanmin(edge_estim)) * ((up_limit-down_limit)/(np.nanmax(edge_estim) - np.nanmin(edge_estim))) + down_limit
            # Normalize to 0-100
            edge_estim = (edge_estim - np.nanmin(edge_estim)) * ((100-0)/(np.nanmax(edge_estim) - np.nanmin(edge_estim))) + 0
            edge_estim = edge_estim.astype(np.float32)

            imPath = os.path.join(ms_im.datapath_10,
                                f"T{ms_im.tile_id}_{ms_im.str_datetime}_edge.tif")

            metadata.update(count=1, dtype=edge_estim.dtype)
            with rio.open(imPath, 'w', **metadata) as dst:
                dst.write(edge_estim, 1)

            # gather edge estimation images fullpaths
            ms_im.edge = imPath
            self.estim_paths.append(ms_im.edge)
        self.estim_meta = metadata


    def cube_by_paths(self, listOfPaths:list, outfname:str=None):
        """Concatenate images as cube.

        Args:
            listOfPaths (list): List containg fullpaths of images to concatenate on
                time-axis.

        Returns:
            np.array: Cube as 3D array.
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

        if outfname is not None:
            fp = os.path.join(self.epm_path, f"{outfname}.tif")
            with rio.open(fp, 'w', **meta) as dst:
                for id, layer in enumerate(listOfPaths, start=1):
                    with rio.open(layer) as src:
                        dst.write_band(id, src.read(1))
                        dst.set_band_description(id, band_names[id-1])

        return cbarr[1:, :, :], meta, band_names


    def edge_intensity(self):
        """Compute final edge probability map, using Equation (6) of original paper.
        Also fix some issues due to nodata values. Epm refers to a temporal range.
        """
        if hasattr(self, 'estim_paths'):
            pass
        else:
            self.estimate()

        # create cube using edge estimations of all dates
        cbarr, cb_metadata, _ = self.cube_by_paths(self.estim_paths,
                                                outfname='estimations_cube')

        # replace nodata value with np.nan
        cbarr[cbarr == cb_metadata['nodata']] = np.nan
        # Equation (6). Compute the sum of each pixel in depth.
        pix_sum = np.nansum(cbarr, axis=0)
        # Count how many NaNs every pixel has, through the weeks
        not_nan_dates = np.sum(~np.isnan(cbarr) * 1, axis=0)
        res = pix_sum / not_nan_dates

        # replace negative values with nodata value
        res[res < 0] = self.estim_meta['nodata']

        imPath = os.path.join(self.epm_path,
                            f"epm__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")

        self.estim_meta.update(count=1, dtype=res.dtype)
        with rio.open(imPath, 'w', **self.estim_meta) as dst:
            dst.write(res, 1)
            dst.set_band_description(1, f"{self.tmp_rng[0]}_{self.tmp_rng[1]}")


    def cloud_mask(self, write:bool=False):
        """Create an image containing the number of dates having unreliable pixel value
        based on SCL bands.
        """
        listOfPaths = []
        for ms_im in self.eodt.data:
            listOfPaths.append(ms_im.SCL_masked)

        cbarr, meta, _ = self.cube_by_paths(listOfPaths)

        for c in self.scl_unreliable:
            cbarr[cbarr==c] = 1
        cbarr[cbarr!=1] = 0

        # sclmsk = np.sum(cbarr, axis=0).astype(np.uint8)
        self.scl_mask = cbarr.astype(np.uint8)

        if write:
            fp = os.path.join(self.epm_path,
                            f"scl_mask__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")

            meta.update(dtype=self.scl_mask.dtype, nodata=0)
            with rio.open(fp, 'w', **meta) as dst:
                dst.write(self.scl_mask)


    def mask_ndviseries(self, write:bool=False):
        if hasattr(self, "scl_mask"):
            pass
        else:
            self.cloud_mask()

        listOfPaths = []
        for ms_im in self.eodt.data:
            listOfPaths.append(ms_im.NDVI_masked)

        ndviseries, meta, _ = self.cube_by_paths(listOfPaths)
        ndviseries[self.scl_mask==0] = meta['nodata']
        self.ndviseries_masked = ndviseries

        if write:
            fp = os.path.join(self.epm_path,
                            f"ndviseries_masked__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")

            meta.update(dtype=self.ndviseries_masked.dtype, nodata=meta['nodata'])
            with rio.open(fp, 'w', **meta) as dst:
                dst.write(self.ndviseries_masked)
                for b in range(0, self.ndviseries_masked.shape[0]):
                    dst.set_band_description(b+1, f"{self.eodt.dates[b]}")




    def interpol_ndvi(self, write:bool=False):
        if hasattr(self, 'ndviseries_masked'):
            pass
        else:
            self.mask_ndviseries()

        