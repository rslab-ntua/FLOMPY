import os
import datetime as dt
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
from geopandas import gpd
from shapely.geometry import Polygon
import rasterio as rio
from rasterio import features
from rasterio.mask import mask
from scipy.ndimage import binary_opening
from rasterstats import zonal_stats
import multiprocessing as mp
from Preprocessing_S2_data.sts import sentimeseries
from Crop_delineation import utils

class CropDelineation():
    def __init__(self, eodata:sentimeseries, dst_path:str, corine_path:str):
        """Compute edge probability map for dates in given sentimeseries object.
        Args:
            eodata (sentimeseries): sentimeseries object, after having bands 11, 12 resamled
                to 10 meters and masked using an area of interest.
            dst_path (str): Fullpath where to save results.
        Raises:
            TypeError: [description]
        """
        self.eodt = eodata
        self.tmp_rng = (self.eodt.dates[0].strftime('%Y%m%d'),
                            self.eodt.dates[-1].strftime('%Y%m%d'))
        self.corine_path = corine_path

        self.dst_path = os.path.join(dst_path, f"result__{dt.datetime.now().strftime('%Y%m%dT%H%M')}")
        os.mkdir(self.dst_path)

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

            outfname = os.path.join(ms_im.datapath_10,
                                f"T{ms_im.tile_id}_{ms_im.str_datetime}_edge.tif")

            metadata.update(count=1, dtype=edge_estim.dtype)
            with rio.open(outfname, 'w', **metadata) as dst:
                dst.write(edge_estim, 1)

            # gather edge estimation images fullpaths
            ms_im.edge = outfname
            self.estim_paths.append(ms_im.edge)
        # self.estim_meta = metadata


    def edge_probab_map(self, write:bool=False):
        """Compute final edge probability map, using Equation (6) of original paper.
        Also fix some issues due to nodata values. Epm refers to a temporal range.
        """
        if hasattr(self, 'estim_paths'):
            pass
        else:
            self.estimate()

        # create cube using edge estimations of all dates
        cbarr, cb_metadata, _ = utils.cube_by_paths(self.estim_paths,
            # outfname=os.path.join(self.dst_path, 'estimations_cube.tif')
            )
        # mask scl (give nodata value where clouds exist)
        if 'cloud_mask' in self.masks:
            cbarr[self.masks['cloud_mask']==1] = cb_metadata['nodata']

        # replace cube's nodata value with np.nan to estimate missing dates
        cbarr[cbarr == cb_metadata['nodata']] = np.nan
        # Equation (6). Compute the sum of each pixel in depth.
        pix_sum = np.nansum(cbarr, axis=0)
        # Count how many NaNs every pixel has, through the weeks
        not_nan_dates = np.sum(~np.isnan(cbarr) * 1, axis=0)
        # divide pixel's sum with not nan dates
        res = pix_sum / not_nan_dates

        # replace negative values (if any) with nodata value
        res[res < 0] = cb_metadata['nodata']

        # mask corine (give nodata value where towns exist)
        if 'town_mask' in self.masks:
            res = res[np.newaxis,:,:]
            res[self.masks['town_mask']==1] = cb_metadata['nodata']

        # replace nodata value with zero and set this as the new nodata velue
        res[res==cb_metadata['nodata']] = 0

        # change range from 0-100 to 0-1 
        res = res/100

        self.epm = res
        cb_metadata.update(count=1, dtype=res.dtype, nodata=0)
        self.epm_meta = cb_metadata

        if write:
            outfname = os.path.join(self.dst_path,
                                f"epm__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")

            with rio.open(outfname, 'w', **cb_metadata) as dst:
                dst.write(res)
                dst.set_band_description(1, f"{self.tmp_rng[0]}_{self.tmp_rng[1]}")


    def create_series(self, write:bool=False):
        # TODO: able to create any series

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
            outfname = os.path.join(self.dst_path,
                            f"ndviseries__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            with rio.open(outfname, 'w', **self.ndviseries_meta) as dst:
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
        corine_data = corine_data.to_crs(mask_meta['crs'].to_epsg())
        # iterable geometry-value pairs
        agri_regions = [[row.geometry, 2] for i, row in corine_data.iterrows()]
        # rasterize mask
        town_mask = features.rasterize(agri_regions,
            out_shape = (mask_meta['height'], mask_meta['width']),
            all_touched = False,
            transform = mask_meta['transform'])

        # not agri areas
        town_mask[town_mask==0] = 1
        # agri areas  (set 0 as nodata value)
        town_mask[town_mask==2] = 0
        # TODO: mask image at aoi without clipping
        # add mask to masks
        self.masks['town_mask'] = town_mask[np.newaxis,:,:]

        if write:
            outfname = os.path.join(self.dst_path,
                            f"town_mask__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            mask_meta.update(dtype=self.masks['town_mask'].dtype, nodata=0, count=1)
            with rio.open(outfname, 'w', **mask_meta) as dst:
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

        # 1 for bad scl classes
        for c in list(self.scl_unreliable.keys()):
            cbarr[cbarr==c] = 1
        # 0 for not bad scl classes (set 0 as nodata value)
        cbarr[cbarr!=1] = 0

        # add mask to masks
        self.masks['cloud_mask'] = cbarr.astype(np.uint8)

        if write:
            outfname = os.path.join(self.dst_path,
                            f"cloud_mask__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
            meta.update(dtype=self.masks['cloud_mask'].dtype, nodata=0)
            with rio.open(outfname, 'w', **meta) as dst:
                dst.write(self.masks['cloud_mask'])
                for b in range(0, self.masks['cloud_mask'].shape[0]):
                    dst.set_band_description(b+1, f"{self.eodt.dates[b].strftime('%Y%m%d')}")


    def crop_probab_map(self, cube:np.ndarray, cbmeta:dict, write:bool=False):
        """Interpolate timeseries, where pixels have np.nan value or nodata value
        as defined by cube metadata.
        Args:
            cube (np.ndarray): Timeseries as 3D cube (count:bands, height:rows, width:columns).
            cbmeta (dict): Containing all cube metadata, as returned by rasterio.
            write (bool, optional): If True saves the result as tif image. Defaults to False.
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
        # print(cbdf['pix_4413300'])

        print("Resample monthly max...")
        cbdf = cbdf.resample(rule='M').max()
        # print(cbdf['pix_4413300'])

        max_monthly = cbdf.max(axis=0)
        # 95th Percentile of all the NDVI values in TILE & in depth. -or NDVImax for scenario No.1-.
        img_percentile = np.nanpercentile(max_monthly, 95)

        # EQUATION (2)
        max_monthly['crop_prob'] = max_monthly.apply(
        [lambda x: 1 if x>=img_percentile else (0 if x<0 else x/img_percentile)])

        # Convert dataframe to array.
        res = max_monthly['crop_prob'].to_numpy()

        # Reshape as rasterio needs.
        res = np.reshape(res, (1, cbmeta['height'], cbmeta['width']))

        self.cpm = res
        cbmeta.update(count=1, dtype=res.dtype)
        self.cpm_meta = cbmeta

        if write:
            outfname=os.path.join(self.dst_path,
                f"cpm__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")

            if outfname is not None:
                assert os.path.isabs(outfname)
                with rio.open(outfname, 'w', **cbmeta) as dst:
                    dst.write(res)



    def active_fields(self):

        meta = self.ndviseries_meta.copy()
        meta.update(count=1, dtype=np.uint8, nodata=0)

        active_fields = np.zeros((meta['count'], meta['height'], meta['width']), dtype=np.uint8)

        # edges
        active_fields[self.epm>0] = 1

        # active fields
        active_fields[(self.epm<0.1) & (self.cpm>0.45)] = 2

        # inactive fields
        active_fields[(self.epm<=0.1) & (self.cpm<=0.45)] = 3

        # nodata
        active_fields[self.epm == self.epm_meta['nodata']] = 0
        active_fields[self.cpm == self.cpm_meta['nodata']] = 0

        self.active_fields = active_fields

        self.active_fields_fpath=os.path.join(self.dst_path,
            f"active_fields__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")

        with rio.open(self.active_fields_fpath, 'w', **meta) as dst:
            dst.write_colormap(
            1, {
                0: (0, 0, 0, 0),
                1: (0, 0, 0, 255),
                2: (3, 100, 0, 255),
                3: (166, 217, 62, 255),
                })
            dst.write(active_fields)



    def delineation(self, aoi_path:str, unet_pred_path:str):
        print('Delineation')

        # Threshold epm to mean value of the image (edge=1, noedge=0)
        threshold = np.nanmean(self.epm)
        thres_epm = self.epm > threshold

        # read AOI in order to clip UNet predicted image; due to UNet padding
        # has different dimensions
        aoi=gpd.read_file(aoi_path)
        aoi=aoi.to_crs(self.epm_meta['crs'])

        # Read UNet predicted image (edge=1, noedge=0)
        with rio.open(unet_pred_path, 'r') as src:
            pred, _ = mask(src, aoi.geometry, crop=True, nodata=0)

        pred = pred.astype(np.bool)

        # Combine thresholded EPM and UNet prediction (edge=1, noedge=0)
        combined_edges = np.logical_or(thres_epm, pred)

        # Invert and convert from bool to binary (edge=0, noedge=1)
        combined_edges = 1-combined_edges
        self.combined_edges = combined_edges



    def flooded_fields(self, flood_tif_path:str):
        print('Flooded fields')

        # Morphological Opening
        opening = binary_opening(self.combined_edges, structure=np.ones((1,2,2))).astype(np.int16)

        opening[self.active_fields == 0] = 0

        opening_fpath=os.path.join(self.dst_path,
            f"opening__{self.tmp_rng[0]}_{self.tmp_rng[1]}.tif")
        with rio.open(opening_fpath, 'w', **self.epm_meta) as dst:
            dst.write(opening)

        # Vectorize fields
        vfields = ({'properties': {}, 'geometry': s} for i, (s, v) in enumerate(
                features.shapes(opening, connectivity=8, transform=self.epm_meta['transform'])) if v != 0)
        vfields = gpd.GeoDataFrame.from_features(vfields, crs=self.epm_meta['crs'])

        # Delete holes & zero buffer to correct self-intersected geometries
        vfields.geometry = vfields.geometry.apply(lambda x:Polygon(x.exterior.coords))
        vfields.geometry = vfields.geometry.buffer(0)

        # Read flood image
        with rio.open(flood_tif_path, 'r') as src:
            flood_meta = src.meta
            flood_img = src.read()

        # Vectorize flood
        flood = ({'properties': {}, 'geometry': s} for i, (s, v) in enumerate(
                features.shapes(flood_img, connectivity=8, transform=flood_meta['transform'])) if v != 0)
        flood = gpd.GeoDataFrame.from_features(flood, crs=flood_meta['crs']).to_crs(self.epm_meta['crs'])

        # Zero buffer to correct self-intersected geometries & save as shp
        flood.geometry = flood.geometry.buffer(0)
        # Many geometries to one multipolygon geometry
        flood = flood.dissolve()

        # Spatial join fields and flood (CRS !)
        flood_id, flooded_field_ids = vfields.sindex.query_bulk(flood.geometry, predicate='intersects')
        flooded_fields = vfields.loc[flooded_field_ids]

        def selected_most_flooded(x:gpd.GeoDataFrame, flood:gpd.GeoDataFrame):
            inter_area = flood.intersection(x['geometry']).area
            ratio = inter_area / x['geometry'].area
            ratio = ratio[0]
            if ratio > 0.1:
                return True
            else:
                return False


        # Select fields flooded over 10%
        flooded_fields['flooded'] = flooded_fields.apply(lambda x: selected_most_flooded(x, flood), axis=1)
        flooded_fields = flooded_fields.loc[flooded_fields['flooded']==True]

        # Major voting to determine cultivated/not_cultivated flooded fields
        flooded_fields = gpd.GeoDataFrame.from_features(
            zonal_stats(vectors=flooded_fields, 
                        raster=self.active_fields_fpath,
                        stats='majority',
                        geojson_out=True))

        def actInact(x):
            if x['majority'] == 2:
                return 'cultivated'
            elif x['majority'] == 3:
                return 'not_cultivated'
            elif x['majority'] == 1:
                return 'edge'
            else:
                return 'unknown_state'

        flooded_fields['status'] = flooded_fields.apply(lambda x: actInact(x), axis=1)

        # Save flooded fields
        flooded_fields_fpath=os.path.join(self.dst_path,
            f"flooded_fields__{self.tmp_rng[0]}_{self.tmp_rng[1]}.shp")

        flooded_fields.to_file(flooded_fields_fpath)