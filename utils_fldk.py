"""AI-VIS utils module"""
import warnings
import gc
import numpy as np
from satpy.scene import Scene
from geopy.distance import geodesic
from .astronomy import get_observer_look, get_alt_az

class InvalidAreaError(Exception):
    pass

class SCENE2DATA:
    def __init__(self, crop_with_lonlat=False, flip_lon=False, lon=None, lat=None):
        """Init AI-VIS utils module.

        Args:
            crop_with_lonlat (bool, optional): Whether to crop data with given longitude and latitude.
            flip_lon (bool, optional): Whether to flip longitude data. If yes, will add 360 for longitude that lower than 0, or minus 360 that greater than 180.
            lon (float, optional): Center longitude to crop data.
            lat (float, optional): Center latitude to crop data.
        """
        self.crop_with_lonlat = crop_with_lonlat
        self.flip_lon = flip_lon
        self.lon = lon
        self.lat = lat
    
    def _proj_inverse_basemap(self, map_shape, lons, lats):
        """Backproject longitude/latitude into column/row to crop basemap.

        Args:
            map_shape (Tuple(int)): shape of map.
            lons (np.ndarray): Input longitude data to crop.
            lats (np.ndarray): Input latitude data to crop.
        """
        row, col = map_shape
        max_lat, max_lon = 180, 360
        
        _lons, _lats = lons.copy(), lats.copy()
        _lons[_lons<0] += 360
        _lats[:] += 90
        
        rows = row * (1 - _lats / max_lat)
        cols = col * (_lons / max_lon)
        
        rows[(np.isnan(rows)) | (np.isinf(rows))] = 0
        cols[(np.isnan(cols)) | (np.isinf(cols))] = 0
        
        return cols.astype(int), rows.astype(int)
    
    def _get_basemap(self, path_map, lons, lats):
        """Get basemap based on cropped area.
        If map data is not found, will return an array filled with 0.

        Args:
            path_map (str): Path of the map data to read (need a `.npz` file).
            lons (np.ndarray): Input longitude data to crop.
            lats (np.ndarray): Input latitude data to crop.
        """
        if not path_map.endswith('.npz'):
            raise ValueError('must give a .npz file to read.')
        
        try:
            with np.load(path_map) as npLoad:
                basemap = npLoad['basemap']
                map_shape = basemap.shape
            cols, rows = self._proj_inverse_basemap(map_shape, lons, lats)
            basemap = basemap[rows, cols]
        except BaseException:
            basemap = np.zeros((500, 500))
        
        return basemap
    
    def _get_band_data(self, ds, channels):
        """Get single channel data using Satpy modules.

        Args:
            ds (xarray.Dataset): xr.Dataset object transformed from Satpy.scene object.
            channels (List[str]): List of channels to get data.
        """
        datas = []
        for channel in channels:
            data = ds[channel].values
            datas.append(data - 273.15)
        datas = np.asarray(datas)
        return datas
    
    def get_datas_from_satpy(self, path_map, files, reader, load_channels):
        # load channels
        scn = Scene(files, reader=reader)
        scn.load(load_channels)
        ds = scn.to_xarray_dataset()
        del scn; gc.collect()
        
        # get satellite meta
        utc_time = ds.start_time
        sat_lon = ds[load_channels[0]].orbital_parameters['projection_longitude']
        sat_lat = ds[load_channels[0]].orbital_parameters['projection_latitude']
        sat_alt = ds[load_channels[0]].orbital_parameters['projection_altitude']
        sat_alt = sat_alt / 1000
        
        # extract all data
        datas = self._get_band_data(ds, load_channels)
        lons, lats = ds[load_channels[0]].attrs['area'].get_lonlats()
        basemap = self._get_basemap(path_map, lons, lats)
        
        del ds; gc.collect()
        
        return lons, lats, datas, basemap, utc_time, sat_lon, sat_lat, sat_alt
    
    def get_msg_from_satpy(self, lon, lat, sat_lon, sat_lat, sat_alt, utc_time):
        # calculate `sza` and `az`
        salt, saz = get_alt_az(utc_time, lon, lat)
        sza = 90 - np.absolute(salt * 180 / np.pi)
        az = saz * 180 / np.pi
        
        # calculate `sat_az` and `sat_za`
        sat_az, sat_el = get_observer_look(sat_lon, sat_lat, sat_alt, utc_time, lon, lat, 0)
        sat_za = 90 - sat_el
        
        return sza, az, sat_za, sat_az
