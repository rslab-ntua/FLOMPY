import os
import struct
from cupshelpers import Printer
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import rasterio as rio
from rasterio import features
import multiprocessing as mp
from engine.sts import sentimeseries
from edges import utils

class CropDelineation():
    def __init__(self, eodata:sentimeseries, epm_path:str, corine_path:str):
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
        self.corine_path = corine_path
        self.senbands = ['B03_masked',
                        'B04_masked',
                        'B08_masked',
                        'B11_masked',
                        'B12_masked',
                        'NDVI_masked']
        self.scl_unreliable = {
            1:'SATURATED_OR_DEFECTIVE',
            2:'DARK_AREA_PIXELS',
            3:'CLOUD_SHADOWS',
            8:'CLOUD_MEDIUM_PROBABILITY',
            9:'CLOUD_HIGH_PROBABILITY',
            10:'THIN_CIRRUS'}
        self.masks={}

        if isinstance(self.eodt, sentimeseries):
            for i in range(len(self.eodt.data)):
                for im in self.senbands:
                    assert hasattr(self.eodt.data[i], im), f'Missing attribute {im}!'
        else:
            raise TypeError("Only sentimeseries objects are supported!")

    def estimate(self):
        wk = utils.wkernels()
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
                utils.equation_4, [(ndvi, wk.iloc[i]) for i in range(0, len(wk))]).get()

            # apply kernels on bands b03, b04, b08, b11, b12
            five_bands = [getattr(ms_im, im) for im in self.senbands[:-1]]
            dweeks = pool.starmap_async(
                utils.equation_3, [(five_bands, wk.iloc[i]) for i in range(0, len(wk))]).get()

            # edge estimation of current data
            edge_estim = utils.equation_5(ndvi, dweeks, dndvi, wk)

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


    def edge_intensity(self, write:bool=True):
        """Compute final edge probability map, using Equation (6) of original paper.
        Also fix some issues due to nodata values. Epm refers to a temporal range.
        """
        if hasattr(self, 'estim_paths'):
            pass
        else:
            self.estimate()

        # create cube using edge estimations of all dates
        cbarr, cb_metadata, _ = utils.cube_by_paths(self.estim_paths,
            # outfname=os.path.join(self.epm_path, 'estimations_cube.tif')
            )
        # mask scl (nodata where clouds exist)
        if 'cloud_mask' in self.masks:
            cbarr[self.masks['cloud_mask']==1] = cb_metadata['nodata']

        # replace nodata value with np.nan
        cbarr[cbarr == cb_metadata['nodata']] = np.nan
        # Equation (6). Compute the sum of each pixel in depth.
        pix_sum = np.nansum(cbarr, axis=0)
        # Count how many NaNs every pixel has, through the weeks
        not_nan_dates = np.sum(~np.isnan(cbarr) * 1, axis=0)
        res = pix_sum / not_nan_dates

        # replace negative values with nodata value
        res[res < 0] = self.estim_meta['nodata']

        # mask corine (nodata where towns exist)
        if 'town_mask' in self.masks:
            res = res[np.newaxis,:,:]
            res[self.masks['town_mask']==1] = self.estim_meta['nodata']

        # res[res==self.estim_meta['nodata']] = 0
        # print(np.nanpercentile(res, 75))
        # res[res<np.nanpercentile(res, 75)] = 0

        # res[res>=np.nanpercentile(res, 75)] = 1
        # res=res.astype(np.uint8)
        
        # # import scipy
        # # kernel=np.ones((1,3,3), np.uint8)
        # # res=scipy.ndimage.binary_erosion(res).astype(np.uint8)
        # from skimage.morphology import skeletonize
        # res = skeletonize(res[0,:,:]).astype(np.uint8)
        # res = res[np.newaxis,:,:]

        if write:
            imPath = os.path.join(self.epm_path,
                                f"epm__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            self.estim_meta.update(count=1, dtype=res.dtype, nodata=0)
            with rio.open(imPath, 'w', **self.estim_meta) as dst:
                dst.write(res)
                dst.set_band_description(1, f"{self.tmp_rng[0]}_{self.tmp_rng[1]}")


    def ndviseries(self, write:bool=True):
        # gather absolute paths of ndvi images masked using aoi
        listOfPaths = []
        for ms_im in self.eodt.data:
            listOfPaths.append(ms_im.NDVI_masked)
        # crete ndvi series cube
        ndviseries, meta, _ = utils.cube_by_paths(listOfPaths)
        # mask scl (nodata where clouds exist)
        if 'cloud_mask' in self.masks:
            ndviseries[self.masks['cloud_mask']==1] = meta['nodata']
        # mask corine (nodata where towns exist)
        if 'town_mask' in self.masks:
            temp_townmask = np.vstack([self.masks['town_mask']] * meta['count'])
            ndviseries[temp_townmask==1] = meta['nodata']

        self.ndviseries = ndviseries
        self.ndviseries_meta = meta

        if write:
            fp = os.path.join(self.epm_path,
                            f"ndviseries__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            with rio.open(fp, 'w', **self.ndviseries_meta) as dst:
                dst.write(self.ndviseries)
                for b in range(0, self.ndviseries.shape[0]):
                    dst.set_band_description(b+1, f"{self.eodt.dates[b].strftime('%Y%m%d')}")


    def town_mask(self, write:bool=False):
        # maintain agricultural corine classes
        corine_data = utils.filter_corine(self.corine_path)
        # source metadata by a random image masked by aoi
        with rio.open(self.eodt.data[0].NDVI_masked, 'r') as src:
            mask_meta = src.meta
        
        # reproject corine to img crs
        corine_data = corine_data.to_crs(mask_meta['crs'])
        # iterable geometry-value pairs
        agri_regions = [[row.geometry, 2] for i, row in corine_data.iterrows()]
        # rasterize mask
        town_mask = features.rasterize(agri_regions,
            out_shape = (mask_meta['height'], mask_meta['width']),
            all_touched = False,
            transform = mask_meta['transform'])

        # not agri areas
        town_mask[town_mask==0] = 1
        # agri areas
        town_mask[town_mask==2] = 0
        # TODO: mask image at aoi without clipping
        # add mask to masks
        self.masks['town_mask'] = town_mask[np.newaxis,:,:]

        if write:
            fp = os.path.join(self.epm_path,
                            f"town_mask__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            mask_meta.update(dtype=self.masks['town_mask'].dtype, nodata=0, count=1)
            with rio.open(fp, 'w', **mask_meta) as dst:
                dst.write(self.masks['town_mask'])


    def cloud_mask(self, write:bool=False):
        """Create an image containing the number of dates having unreliable pixel value
        based on SCL bands.
        """
        # gather absolute paths of scl images masked using aoi
        listOfPaths = []
        for ms_im in self.eodt.data:
            listOfPaths.append(ms_im.SCL_masked)
        # crete 'bad_scl_values' series cube
        cbarr, meta, _ = utils.cube_by_paths(listOfPaths)

        # bad scl classes
        for c in list(self.scl_unreliable.keys()):
            cbarr[cbarr==c] = 1
        # not bad scl classes
        cbarr[cbarr!=1] = 0

        # add mask to masks
        self.masks['cloud_mask'] = cbarr.astype(np.uint8)

        if write:
            fp = os.path.join(self.epm_path,
                            f"cloud_mask__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            meta.update(dtype=self.masks['cloud_mask'].dtype, nodata=0)
            with rio.open(fp, 'w', **meta) as dst:
                dst.write(self.masks['cloud_mask'])
                for b in range(0, self.masks['cloud_mask'].shape[0]):
                    dst.set_band_description(b+1, f"{self.eodt.dates[b].strftime('%Y%m%d')}")


    def interpolate_series(self, cube:np.ndarray, cbmeta:dict, outfname:str=None):
        """Interpolate timeseries, where pixels have np.nan value or nodata value
        as defined by cube metadata.

        Args:
            cube (np.ndarray): Timeseries as 3D cube (count:bands, height:rows, width:columns).
            cbmeta (dict): Containing all cube metadata, as returned by rasterio.
            outfname (str, optional):  Absolute filename for the resulted geotif file.
                Defaults to None. When given, the 3D cube array will be saved.
        """
        # convert 3D ndviseries to pandas dataframe. Each column is a pixel's depth.
        cbdf = utils.cbarr2cbdf(cube, cbmeta)

        # datetimes as dataframe column
        cbdf['date'] = self.eodt.dates
        # column's type to datetime.
        cbdf['date'] = pd.to_datetime(cbdf['date'])
        # datetime index
        cbdf.set_index('date', drop=True, inplace=True, verify_integrity=True)

        # Replace nodata value with np.nan
        cbdf.replace(cbmeta['nodata'], np.nan, inplace=True)
        # df = df[df.columns[~df.isnull().all()]]
        print(cbdf['pix_4413300'])

        # print("Resample weekly...")
        # cbdf = cbdf.resample(rule='W').mean()
        # print(cbdf['pix_4413300'])

        print("Interpolate timely..")
        cbdf.interpolate(method='time', limit_direction='both',
            inplace=True, limit=len(cbdf)-2, axis=0)
        print(cbdf['pix_4413300'])

        print("Resample monthly max...")
        cbdf = cbdf.resample(rule='M').max()
        print(cbdf['pix_4413300'])

        max_monthly = cbdf.max(axis=0)
        # 95th Percentile of all the NDVI values in TILE & in depth. -or NDVImax for scenario No.1-.
        img_percentile = np.nanpercentile(max_monthly, 95)

        # EQUATION (2)
        max_monthly['crop_prob'] = max_monthly.apply(
        [lambda x: 1 if x>=img_percentile else (0 if x<0 else x/img_percentile)])

        # Convert dataframe to array.
        temp_arr = max_monthly['crop_prob'].to_numpy()

        # Reshape as rasterio needs.
        temp_arr = np.reshape(temp_arr, (1, cbmeta['height'], cbmeta['width']))

        cbmeta.update(count=1, dtype=temp_arr.dtype)
        if outfname is not None:
            assert os.path.isabs(outfname)
            with rio.open(outfname, 'w', **cbmeta) as dst:
                dst.write(temp_arr)


