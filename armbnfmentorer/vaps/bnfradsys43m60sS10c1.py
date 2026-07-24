import json
import pathlib as pl
import productomator.worker as prowo
# import sqlite3
import xarray as xr
import pandas as pd
import atmPy.radiation.retrievals.broadband_shortwave_radiation as atmbrad
import atmPy.general.measurement_site as atmsite
import socket
import atmPy.radiation.radflux.radflux_db as atmraddb

class BnfRadsys43m60sS10C1Radflux(prowo.Workplanner):
    def __init__(self, *args, radflux_parameters_db, **kwargs):
        """BNF Radsys value added product for the tower system with radflux parameter database.
        Features
        --------
        
        Cheanlog
        ------------
        version 0.1: initial version
        
        """
        self.version = '0.1'
        kwargs['version'] = self.version
        self.radflux_parameters_db = pl.Path(radflux_parameters_db)
        kwargs['database'] = (self.radflux_parameters_db, 'radflux_parameters', 'row_timestamp', 'None')# 'input_file')
        super().__init__(*args, **kwargs)
        self.site = atmsite.Station(
                lat=34.3437276,
                lon=-87.35044401,
                alt=284,
                name='tower',
                abbreviation='S10_top',
                active=None,
                operation_period=None,
                info=None,
                state='',
                country='',
                parent_network=None,
                # **kwargs,
            )
        # self.combine_masterplan_duplicates()
        self.radflux_db = atmraddb.RadfluxParameterDatabase(self.radflux_parameters_db, create_if_not_exist=True, verbose=self.verbose, version = self.version)

    def open_p2f_in(self, row):
        """Opens the input file(s) for a given row and returns an xarray dataset."""
        if isinstance(row.p2f_in, list):
            ds = xr.open_mfdataset(row.p2f_in)
        else:
            ds = xr.open_dataset(row.p2f_in)
        return ds

    def process_row(self, row = None, iloc = None, loc = None, save = True):
        out = {}
        if iloc is not None:
            row = self.workplan.iloc[iloc]
        elif loc is not None:
            row = self.workplan.loc[loc]
        self.tp_row = row

        clearsky_parameters_previous = self.radflux_db.read_previous_valid_clearsky_parameters(row.name)
        self.tp_previous_clearsky_parameters = clearsky_parameters_previous

        #######
        ## Open input files
        #######
        ds = self.open_p2f_in(row)

        self.tp_ds = ds.copy()

        bbi_rename_dict = {'down_short_hemisp': 'global_horizontal',
                        'down_short_diffuse_hemisp': 'diffuse_horizontal',
                        'down_short_direct_hemisp': 'direct_horizontal',
                        'time':'datetime'}
        ds = ds.rename(bbi_rename_dict)
        bbicore = atmbrad.CombinedGlobalDiffuseDirect(ds, site= self.site, verbose = self.verbose)

        ####
        # Test if we need the next day to have a complete day
        # But first check if we need to get rid of the beginning of the day bacause the sun is still up from the previous day.
        bbicore.sun_position #just to trigger the calculation of sun position
        sunup_start = ds.solar_elevation.isel(datetime = 0) > 0 #still in the sky in beginning of file
        # sunrising_start = ds.solar_elevation.differentiate('time').isel(time = 0) > 0 #sun is rising at the begining
        sunup_end = ds.solar_elevation.isel(datetime = -1) > 0 #still in the sky at end of file
        # sunrising_end = ds.solar_elevation.differentiate('time').isel(time = -1) > 0 #sun is rising at the end
        next_day_needed = False
        dslist = []
        if sunup_start:
        # delete everything before first solar_elevation minimum
            dslist.append(ds.sel(datetime = slice(ds.solar_elevation.idxmin(), None)))
        else:
            dslist.append(ds)
        
        if sunup_end:
            try:
                row_next = self.masterplan.iloc[self.masterplan.index.get_loc(row.name)+1]
            except IndexError:
                print('We have to wait for the next day to to finish this local day.')
                return None
            
            next_day_needed = True

            dsnext = self.open_p2f_in(row_next)
            dsnext = dsnext.rename(bbi_rename_dict)
            bbinext = atmbrad.CombinedGlobalDiffuseDirect(dsnext, site= self.site, verbose = self.verbose)
            bbinext.sun_position #just to trigger the calculation of sun position
            dsnext = dsnext.sel(datetime = slice(None,dsnext.solar_elevation.idxmin()))
            dslist.append(dsnext)
            self.tp_dsnext = dsnext.copy()

        ds_wholeday = xr.concat(dslist, dim = 'datetime')
        ds_wholeday = ds_wholeday.where(ds_wholeday.solar_elevation > 0, drop = True)
        self.tp_ds_wholeday = ds_wholeday.copy()
        file_shape = ds_wholeday.datetime.shape[0]
        if file_shape > 1:
            file_duration = (ds_wholeday.datetime.data[-1] - ds_wholeday.datetime.data[0])/pd.to_timedelta(1,'h')
        else:
            file_duration = 0
        file_too_long = False
        min_required_points = 2
        file_too_short = False
        if file_shape < min_required_points:
            if self.verbose:
                print(f'Not enough data points to optimize clearsky parameters. Found {file_shape} points, but need at least {min_required_points}.')
            file_too_short = True
        elif file_duration > 24:
            print(f'The whole day dataset is longer than 24 hours. It is {file_duration} hours. This probably means that there was a data gap and this file should not be processed, and just marked as incomplete. The file is {row.p2f_in}.')
            file_too_long = True

        out['ds_wholeday'] = ds_wholeday

        bbi = atmbrad.CombinedGlobalDiffuseDirect(ds_wholeday, site= self.site, verbose = self.verbose)
        bbi.direct_normal_irradiation #just to trigger the calculation of direct normal
        # return bbi, clearsky_parameters_previous
        if file_too_short and not file_too_long:
            pass
        else:
            if clearsky_parameters_previous is None:
                if self.verbose:
                    print('No previous optimized clearsky parameters found, using default parameters.')
            else:
                bbi.clearsky_parameters = clearsky_parameters_previous #set starting conditions
            bbi.optimize_clearsky_parameters()
        # min_required_points = 2
        # file_too_short = False
        # if bbi.dataset.datetime.shape[0] < min_required_points:
        #     if self.verbose:
        #         print(f'Not enough data points to optimize clearsky parameters. Found {bbi.dataset.datetime.shape[0]} points, but need at least {min_required_points}.')
        #     file_too_short = True
        # else:
        # if not file_too_short and not file_too_long:
            # bbi.optimize_clearsky_parameters()

        current_clearsky_parameters = bbi.clearsky_parameters
        processing_date = pd.Timestamp.now().isoformat()
        processing_server = socket.gethostname()

        if file_too_long:
            current_clearsky_parameters = {k: None for k in current_clearsky_parameters}
            bbi.dataset.attrs['clear_sky_params_optimized'] = "File longer than 24 hours"
        elif file_too_short:
            current_clearsky_parameters = {k: None for k in current_clearsky_parameters}
            bbi.dataset.attrs['clear_sky_params_optimized'] = "Not enough data points"
        elif bbi.dataset.attrs.get('clear_sky_params_optimized') != "True":
            current_clearsky_parameters = {k: None for k in current_clearsky_parameters}

        add2db = save
        if add2db:
            self.radflux_db.write_radflux_parameters(
                row,
                current_clearsky_parameters,
                processing_date,
                processing_server,
                bbi.dataset.attrs.get('clear_sky_params_optimized'),
                next_day_needed,
            )

        return bbi
        

class BnfRadsys43m60sS10C1(prowo.Workplanner):
    """BNF Radsys value added product for the tower system.
    Features
    --------
    general:
        - sun information added
    shortwave down:
        - direct horizontal converted to direct normal
        - clearsky mask according to radflux

    Changelog
    ------------
    version 0.1: 
       - direct horizontal converted to direct normal
       - clearsky mask according to radflux
    version 0.2:
         - added radflux parameter database to store the optimized parameters for each processed file
       """
    def __init__(self, *args, radflux_parameters_db, **kwargs):
        self.version = '0.2'
        kwargs['version'] = self.version
        self.radflux_parameters_db = atmraddb.RadfluxParameterDatabase(radflux_parameters_db)
        super().__init__(*args, **kwargs)
        self.site = atmsite.Station(
                lat=34.3437276,
                lon=-87.35044401,
                alt=284,
                name='tower',
                abbreviation='S10_top',
                active=None,
                operation_period=None,
                info=None,
                state='',
                country='',
                parent_network=None,
                # **kwargs,
            )
        self.combine_masterplan_duplicates()



    def process_row(self, row = None, iloc = None, loc = None, save = True):
        """This is the method that does the particular work and will need to be overwritten in your subclass.
        Typical components:
        1. read the input file(s) (row.p2f_in)
        3. convert to xarray dataset (if needed)
        2. format the netcdf file
            2.1 add dataset attributes, creation datetime, creation software, server, site details, etc.
            2.2 add variable attributes, units, long_name, standard_name, etc.
        3. save the output file (row.p2f_out)
        
        Parameters
        ----------
        row : pandas.Series, optional
            A row from the workplan dataframe. This is how the process method callse this function.
        iloc : int, optional
            An integer index to select a row from the workplan dataframe.
        loc : index label, optional
            select a row by timestamp.
            """
        
        out = {}
        if iloc is not None:
            row = self.workplan.iloc[iloc]
        elif loc is not None:
            row = self.workplan.loc[loc]
        self.tp_row = row

        clearsky_parameters = self.radflux_parameters_db.get_clearsky_parameter(row.name)
        self.tp_clearsky_parameters = clearsky_parameters.copy()
        #######
        ## Open input files
        #######
        try:
            ds = xr.open_dataset(row.p2f_in)
        except:
            print(row.p2f_in)
            raise

        bbi_rename_dict = {'down_short_hemisp': 'global_horizontal',
                        'down_short_diffuse_hemisp': 'diffuse_horizontal',
                        'down_short_direct_hemisp': 'direct_horizontal',
                        'time':'datetime'}
        ds = ds.rename(bbi_rename_dict)
        bbi = atmbrad.CombinedGlobalDiffuseDirect(ds, site= self.site, verbose = self.verbose)

        bbi.direct_normal_irradiation #just to trigger the calculation of direct normal
        bbi.clearsky_parameters = clearsky_parameters

        # initiate the caluclation of clearsky values
        bbi.clearsky_global_horizontal
        bbi.clearsky_diffuse_horizontal
        #initiate the calculation of clearsky mask
        bbi.mask_clear_sky_radflux
        
        self.tp_bbi = bbi
        # return bbi
        ########
        # Format the dataset
        ########
        dropvar = ['lat', 'lon', 'alt', 
                #    'zenith_geometric', 
                #    'apparent_elevation', 
                #    'elevation', 'equation_of_time',
                     'mu0',
                # '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                ]
        self.tp_bbi = bbi
        self.tp_ds = bbi.dataset.copy()
        self.tp_dropvar = dropvar

        ds = bbi.dataset.drop_vars(dropvar)
        ds = ds.rename({'global_horizontal':'down_short_hemisp',
                        'diffuse_horizontal': 'down_short_diffuse_hemisp',
                        'direct_horizontal': 'down_short_direct_hemisp',
                        'direct_normal': 'down_short_direct_normal',
                        'datetime':'time',
                        'clearsky_global_horizontal': 'down_short_hemisp_clearsky',
                        'clearsky_diffuse_horizontal': 'down_short_diffuse_hemisp_clearsky',
                        # 'down_short_direct_normal': 'down_short_direct_normal',
                        })  

        # reoganize variables

        ds = ds[[
                'base_time',
                'time_offset',
                'time_bounds',
                'down_short_hemisp',
                'qc_down_short_hemisp',
                    'down_short_hemisp_clearsky',
                'down_short_hemisp_std',
                'down_short_hemisp_case_temp',
                'down_short_hemisp_spn1',
                'qc_down_short_hemisp_spn1',
                'down_short_hemisp_spn1_std',
                'down_short_diffuse_hemisp_spn1',
                'qc_down_short_diffuse_hemisp_spn1',
                'down_short_diffuse_hemisp_spn1_std',
                'down_short_diffuse_hemisp',
                'qc_down_short_diffuse_hemisp',
                        'down_short_diffuse_hemisp_clearsky',
                'down_short_direct_hemisp',
                'qc_down_short_direct_hemisp',
                    'down_short_direct_normal',
                'mask_normalized_global_magnitude',
                'mask_diffuse_magnitude',
                'mask_global_irradiance_variability',
                'mask_normalized_diffuse_ratio_variability',
                'mask_clear_sky_shortwave_radflux',
                'down_long_hemisp',
                'qc_down_long_hemisp',
                'down_long_hemisp_std',
                'down_long_hemisp_case_temp',
                'down_long_hemisp_dome_temp',
                'up_short_hemisp',
                'qc_up_short_hemisp',
                'up_short_hemisp_std',
                'up_short_hemisp_case_temp',
                'up_long_hemisp',
                'qc_up_long_hemisp',
                'up_long_hemisp_std',
                'up_long_hemisp_case_temp',
                'up_long_hemisp_dome_temp',
                'temp_mean',
                'qc_temp_mean',
                'temp_mean_std',
                'rh_mean',
                'qc_rh_mean',
                'rh_mean_std',
                'clean_flag',
                'solar_zenith',
                # 'solar_zenith_geometric',
                # 'solar_elevation_geometric',
                # 'solar_elevation',
                'solar_azimuth',
                # 'solar_equation_of_time',
                'solar_airmass',
                # 'solar_airmass_absolute',
                'solar_sun_earth_distance'
                ]]

        #########
        # Format the dataset attributes
        #########
        dropattrs = ['history','doi','averaging_interval','calib_info','command_line',
                    # '','','','','',
                    ]
        for a in dropattrs:
            ds.attrs.pop(a)
        ds.attrs['radflux_status'] = clearsky_parameters['status']
        ds.attrs['lat'] = self.site.lat
        ds.attrs['lon'] = self.site.lon
        ds.attrs['alt'] = self.site.alt
        ds.attrs['input_datastreams'] = ds.attrs['datastream']
        ds.attrs['input_files'] = row.p2f_in.name
        ds.attrs['data_level'] = 'c1'
        ds.attrs['process_version'] = self.version
        ds.attrs['datastream'] = 'bnfradsys43m60sS10.c1'
        processing_date = pd.Timestamp.now().isoformat()
        processing_server = socket.gethostname()
        ds.attrs['processing_date'] = processing_date
        ds.attrs['processing_server'] = processing_server



        ## Save the output file
        
        if save:
            row.p2f_out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(row.p2f_out)
        ds.close()
        out['ds'] = ds
        out['bbi'] = bbi
        out['row'] = row
        return out 
