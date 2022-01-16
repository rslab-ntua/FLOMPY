#!/usr/bin/env python
# -*- coding: utf-8 -*-

import geopandas as gpd
import logging
from pyproj import Proj, transform

class parcel():
    """A parcel object is created with geometry and some other attributes."""
    
    def __init__(self, file, id = 0, name = "Unknown", farmer = "Unknown", crop = "Unknown"):
        """A parcel object is created with geometry and some other attributes.

        Args:
            file (str, path-like): Path to shapefile or geoJSON with the polygon of the parcel
            id (int, optional): ID number of the parcel. Defaults to 0
            name (str, optional): Name of the parcel. Defaults to "Unknown"
            farmer (str, optional): Name of the farmer. Defaults to "Unknown"
            crop (str, optional): Crop type of the parcel. Defaults to "Unknown"
        """
        self.file = file
        self.id = id
        self.farmer = farmer
        self.crop = crop
        self.name = name
        self.readGeometry()

    def readGeometry(self):
        """Reads geometry with geopandas and gets average coordinates as attributes.
        """
        
        gdf = gpd.read_file(self.file)
        setattr(self, 'geometry', gdf)
        self.avgcoords = self.avgCoordinates(self.geometry)

    def avgCoordinates(self, geometry, out_epsg = 'origin', to_file = False):
        """Getting average (centroid) coordinates of a polygon or a multipolygon.

        Args:
            geometry (GeoDataFrame): Shape of the parcel
            out_epsg (str, optional): Output CRS system in EPSG codes (i.e 'epsg:4326'). Defaults to 'origin'
            to_file (bool, optional): Saves centroid coordinates to a shapefile (NOT WORKING). Defaults to False

        Returns:
            tuple: Coordinates as lists. If the CRS is not projected: lon -> x, lat -> y
        """

        points = geometry.copy()
        origin_epsg = points.crs

        # Convert to EPSG:3857 (WORLD PROJECTED) for the calculations
        points = points.to_crs('epsg:3857')

        points.geometry = points['geometry'].centroid
    
        x = points.geometry.x.values
        y = points.geometry.y.values

        epsg = points.geometry.crs

        if out_epsg == 'origin':
            out_epsg = origin_epsg
        
        # If the CRS is not projected: lat -> x, lon -> y
        (x, y) = self.TransformCoordinates(epsg, x, y, outproj = out_epsg)

        return (x, y)

    @staticmethod
    def TransformCoordinates(inproj, x, y, outproj = 'epsg:4326'):
        """Transform coordinates from one CRS to another.

        Args:
            inproj (str): Input CRS system
            x (float): Coordinate x (lon)
            y (float): Coordinate y (lat)
            outproj (str, optional): Output CRS system. Defaults to 'epsg:4326'

        Returns:
            tuple: New set of reprojected coordinates
        """
        inProj = Proj(inproj)
        outProj = Proj(outproj)
        (new_x, new_y) = transform(inProj, outProj, x, y)
        
        return (new_x, new_y)