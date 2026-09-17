
import pathlib as pl
import productomator.worker as prowo
# import sqlite3
import xarray as xr
import pandas as pd
import numpy as np
import atmPy.radiation.retrievals.broadband_shortwave_radiation as atmbrad
import atmPy.general.measurement_site as atmsite
import socket
import atmPy.radiation.radflux.radflux_db as atmraddb

class BnfRadsys43m60sS10C1Radflux(prowo.Workplanner):
    def __init__(self, *args, radflux_parameters_db, path2raflux_setting, **kwargs):
        """BNF Radsys value added product for the tower system with radflux parameter database.
        Features
        --------
        
        Cheanlog
        ------------
        version 0.1: initial version
        version 0.2: 
            - improved Radflux algorithm
            - updated databank structure
        
        """
        self.version = '0.2'
        kwargs['version'] = self.version
        self.radflux_parameters_db = pl.Path(radflux_parameters_db.format(version = self.version))
        kwargs['database'] = (self.radflux_parameters_db, 'radflux_parameters',
                            #   'row_timesta
                            # mp', 
                              'local_day',
                              'None')# 'input_file')
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
        self.path2raflux_setting = pl.Path(path2raflux_setting)
        assert(self.path2raflux_setting.exists()), f"Path does not exist: {self.path2raflux_setting}. Copy the example file from .../atm-py/atmPy/radiation/radflux/resources/clear_sky_shortwave.example.toml"

    def open_p2f_in(self, row):
        """Opens the input file(s) for a given row and returns an xarray dataset."""
        if isinstance(row.p2f_in, list):
            ds = xr.open_mfdataset(row.p2f_in)
        else:
            ds = xr.open_dataset(row.p2f_in)
        return ds

    def process_row(self, row = None, iloc = None, loc = None, save = True, test  = False):
        out = {}
        if iloc is not None:
            row = self.workplan.iloc[iloc]
        elif loc is not None:
            row = self.workplan.loc[loc]
        self.tp_row = row

        clearsky_parameters_previous = self.radflux_db.get_clearsky_parameters(row.name, method='previous')
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
        sunup_end = ds.solar_elevation.isel(datetime = -1) > 0 #still in the sky at end of file
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

        out['ds_wholeday'] = ds_wholeday

        bbi = atmbrad.CombinedGlobalDiffuseDirect(ds_wholeday, site= self.site, verbose = self.verbose)
        bbi = bbi.convert2RadFlux()
        bbi.clearsky_parameters = self.path2raflux_setting # this provides most of the settings that do not change overtime
        bbi.clearsky_parameters = clearsky_parameters_previous # this updates only the parameters that change over time, if there are no previous parameters, the default ones will be used.
        bbi.direct_normal_irradiation #just to trigger the calculation of direct normal
        out['bbi'] = bbi
        if test == 1:
            return out
        self.tp_bbi = bbi

        # some test of the file length and extend
        file_shape = ds_wholeday.datetime.shape[0]
        if file_shape > 1:
            file_duration = (ds_wholeday.datetime.data[-1] - ds_wholeday.datetime.data[0])/pd.to_timedelta(1,'h')
        else:
            file_duration = 0
        file_too_long = False
        min_required_points = bbi.get_attr('minimum_clear_sky_duration_for_daily_fit') #we don't even need to get started if the points are less than that
        file_too_short = False
        if file_shape < min_required_points:
            if self.verbose:
                print(f'Not enough data points to optimize clearsky parameters. Found {file_shape} points, but need at least {min_required_points}.')
            file_too_short = True
        elif file_duration > 24:
            print(f'The whole day dataset is longer than 24 hours. It is {file_duration} hours. This probably means that there was a data gap and this file should not be processed, and just marked as incomplete. The file is {row.p2f_in}.')
            file_too_long = True

        if file_too_short and not file_too_long:
            radflux_res = None
            pass
        else:
            radflux_res = bbi.optimize_clearsky_parameters()

        out['radflux_res'] = radflux_res

        processing_date = pd.Timestamp.now().isoformat()
        processing_server = socket.gethostname()

        if save:
            self.radflux_db.write_radflux_parameters(
                date = row.name,
                path2file = row.p2f_in,
                clearsky_parameters = radflux_res,
                processing_date = processing_date,
                processing_server = processing_server,
                # bbi.dataset.attrs.get('clear_sky_params_optimized'),
                next_day_needed = next_day_needed,
            )

        return out
        

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
    version 0.3:
         - implement new radflux retrieval (testing equivalent to the real radflux)
       """
    def __init__(self, *args, radflux_parameters_db, path2raflux_setting, real_time = False, **kwargs):
        self.version = '0.3'
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
        self.path2raflux_setting = pl.Path(path2raflux_setting)
        assert(self.path2raflux_setting.exists()), f"Path does not exist: {self.path2raflux_setting}. Copy the example file from .../atm-py/atmPy/radiation/radflux/resources/clear_sky_shortwave.example.toml"
        self.real_time = real_time

    @property
    def workplan(self):
        wp = super().workplan
        if self.real_time:
            return wp
        else:
            for idx, row in wp.iterrows():
                clearsky_parameters = self.radflux_parameters_db.get_clearsky_parameters(row.name)
                stop = False
                for v in clearsky_parameters:
                    status = clearsky_parameters[v].status
                    if status == 'Current day is a valid clearsky day':
                        pass
                    elif status.split(',')[0] == 'interpolated':
                        pass
                    elif ','.join(status.split(',')[:2]) == 'extrapolated, no previous parameters found':
                        pass
                    elif ','.join(status.split(',')[:2]) == 'extrapolated, no following parameters found':
                        if self.real_time:
                            pass
                        else:
                            stop = True
                            break
                    else:
                        raise ValueError(f'Unknown status for clearskyparameter: {status}')
                if stop:
                    wp = wp.loc[:idx]
                    wp.drop(idx, inplace=True)
                    break
                else:
                    continue

                # if clearsky_parameters['status'].split(',')[0]== 'extrapolated':
                #     print(f'found extrapolated on {row.name}')
                #     wp = wp.loc[:idx]
                #     wp.drop(idx, inplace=True)
                #     break
                # elif clearsky_parameters['status'].split(',')[0]== 'interpolated':
                #     continue
                # elif clearsky_parameters['status'].split(',')[0]== 'valid clearsky day':
                #                     continue
                # else:
                #     print(clearsky_parameters['status'].split(',')[0])
                #     assert(False), f'found unexpected status {clearsky_parameters["status"]} on {row.name}'
            return wp

    @staticmethod
    def open_input_files(input_files):
        try:
            if isinstance(input_files, list):
                ds = xr.open_mfdataset(input_files)
            else:   
                ds = xr.open_dataset(input_files)
        except:
            print(input_files)
            raise  
        ds = ds.drop_vars(['base_time', 'time_offset', 'time_bounds'])  
        return ds

    # def process(self, raise_errors = False):
    #     si = None
    #     for idx, row in self.workplan.iterrows():
    #         try:
    #             si = self.process_row(row)
    #             if si['status'] == 'break':
    #                 print(f'Breaking on row {idx} due to status: {si["status"]}. This is probably because radflux needs more days in the future.')
    #                 break
    #             self.reporter.clean_increment()

    #         except Exception as e:
    #             if raise_errors:
    #                 raise e
    #             else:
    #                 print(f'Error occurred while processing row {idx}: {e}')
    #                 self.reporter.errors_increment()
    #                 continue
            
    #         print('.', end = '')
    #     return si

    def process_row(self, row = None, iloc = None, loc = None, save = True, test = False):
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
        save : bool, optional
            Whether to save the output file.
        test : bool, optional
            Whether to run in test mode.
            """
        
        out = {'status': 'success'}
        if iloc is not None:
            row = self.workplan.iloc[iloc]
        elif loc is not None:
            row = self.workplan.loc[loc]
        self.tp_row = row

        clearsky_parameters = self.radflux_parameters_db.get_clearsky_parameters(row.name)
        self.tp_clearsky_parameters = clearsky_parameters.copy()

        # test the status of the clearsky parameters. Return depending on conditions.
        # for v in clearsky_parameters:
        #     status = clearsky_parameters[v].status
        #     if status == 'Current day is a valid clearsky day':
        #         pass
        #     elif ','.join(status.split(',')[:2]) == 'extrapolated, no previous parameters found':
        #         pass
        #     elif ','.join(status.split(',')[:2]) == 'extrapolated, no following parameters found':
        #         if self.real_time:
        #             pass
        #         else:
        #             out['status'] = 'break'
        #             return out
        #     else:
        #         raise ValueError(f'Unknown status for clearskyparameter: {status}')
            
        #######
        ## Open input files
        #######
        dslist =[]
        row_position = self.masterplan.index.get_loc(row.name)
        # load previous row
        if row_position > 0:
            row_prev = self.masterplan.iloc[row_position-1]
            if row_prev.name.date() == (row.name.date() - pd.Timedelta(days=1)):
                dslist.append(self.open_input_files(row_prev.p2f_in))    
                if self.verbose:
                    print(f'Loaded previous day {row_prev.name.date()} for {row.name.date()}')

        # load the actual row
        dslist.append(self.open_input_files(row.p2f_in))

        # load next row
        if row_position + 1 < len(self.masterplan):
            row_next = self.masterplan.iloc[row_position+1]
            if row_next.name.date() == (row.name.date() + pd.Timedelta(days=1)):
                dst = self.open_input_files(row_next.p2f_in)
                dslist.append(dst)   
                if self.verbose:
                    print(f'Loaded next day {row_next.name.date()} for {row.name.date()}')

        self.tp_dslist = dslist
        ds = xr.concat(dslist, dim = 'time')

        # format the dataset to be compatible with atmPy
        bbi_rename_dict = {'down_short_hemisp': 'global_horizontal',
                        'down_short_diffuse_hemisp': 'diffuse_horizontal',
                        'down_short_direct_hemisp': 'direct_horizontal',
                        'time':'datetime'}
        ds = ds.rename(bbi_rename_dict)
        bbi = atmbrad.CombinedGlobalDiffuseDirect(ds, site= self.site, verbose = self.verbose)
        bbi = bbi.convert2RadFlux()
        self.tp_bbi = bbi

        bbi.direct_normal_irradiation #just to trigger the calculation of direct normal

        bbi.clearsky_parameters = self.path2raflux_setting # this provides most of the settings that do not change overtime
        bbi.clearsky_parameters = clearsky_parameters

        # initiate the caluclation of clearsky values
        bbi.clearsky_global_horizontal
        bbi.clearsky_diffuse_horizontal
        #initiate the calculation of clearsky mask
        bbi.mask_clear_sky_shortwave #tiggers retrieval
        bbi.cloud_fraction #tiggers retrieval
        bbi.direct_beam_proerties #triggers retrieval
        if test == 1:
            out['bbi'] = bbi
            return out
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
        day_end = row.name.date() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        ds = ds.sel(datetime=slice(row.name.date(), day_end))

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
                # 'base_time',
                # 'time_offset',
                # 'time_bounds',
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
                'mask_global_irradiance_temporal_gradient',
                'mask_normalized_diffuse_ratio_variability',
                'mask_clear_sky_shortwave',
                'shortwave_cloud_fraction',
                'shortwave_cloud_fraction_uncorrected',
                'direct_beam_transmittance', 
                'direct_beam_cloud_effect', 
                'direct_beam_state',
                #
                ### longwave
                #
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
        # attributes to some particular variables
        #########
        da = ds['down_short_hemisp_clearsky']
        da.attrs['radflux_status_coefficient'] = clearsky_parameters.normalized_total_shortwave_power_coefficient.status
        da.attrs['radflux_status_exponent'] = clearsky_parameters.normalized_total_shortwave_power_exponent.status
        da = ds['down_short_diffuse_hemisp_clearsky']
        da.attrs['radflux_status_coefficient'] = clearsky_parameters.normalized_diffuse_ratio_power_coefficient.status
        da.attrs['radflux_status_exponent'] = clearsky_parameters.normalized_diffuse_ratio_power_exponent.status

        #########
        # Format the dataset attributes
        #########
        dropattrs = ['history','doi','averaging_interval','calib_info','command_line',
                     'Conventions',
                     'dod_version',
                        'maximum_solar_zenith_angle',
                        'maximum_diffuse_shortwave_irradiance',
                        'maximum_diffuse_shortwave_cosine_exponent',
                        'normalized_diffuse_ratio_standard_deviation_limit',
                        'normalized_diffuse_ratio_standard_deviation_window_minutes',
                        'shortwave_temporal_gradient_envelope_constant',
                        'minimum_total_shortwave_irradiance',
                        'normalized_total_shortwave_intermediate_iteration_half_width',
                        'normalized_total_shortwave_final_iteration_half_width',
                        'normalized_total_shortwave_low_sun_cosine_boundary',
                        'minimum_clear_sky_duration_for_daily_fit',
                        'diffuse_ratio_exponent_lower_validity_limit',
                        'diffuse_ratio_exponent_upper_validity_limit',
                        'normalized_total_shortwave_lower_limit_low_sun',
                        'normalized_total_shortwave_lower_limit_high_sun',
                        'normalized_total_shortwave_upper_limit',
                        'normalized_diffuse_ratio_power_exponent',
                        'normalized_diffuse_ratio_power_coefficient',
                        'normalized_total_shortwave_power_exponent',
                        'normalized_total_shortwave_power_coefficient',
                    ]
        for a in dropattrs:
            ds.attrs.pop(a)
        # ds.attrs['radflux_status'] = clearsky_parameters['status']
        ds.attrs['lat'] = self.site.lat
        ds.attrs['lon'] = self.site.lon
        ds.attrs['alt'] = self.site.alt
        ds.attrs['input_datastreams'] = ds.attrs['datastream']
        if isinstance(row.p2f_in, list):
            input_files = ', '.join([p2f.name for p2f in row.p2f_in])
        else:
            input_files = row.p2f_in.name

        ds.attrs['input_files'] = input_files
        ds.attrs['data_level'] = 'c1'
        ds.attrs['process_version'] = self.version
        ds.attrs['datastream'] = 'bnfradsys43m60sS10.c1'
        processing_date = pd.Timestamp.now().isoformat()
        processing_server = socket.gethostname()
        ds.attrs['processing_date'] = processing_date
        ds.attrs['processing_server'] = processing_server

        # to save some storage
        for var in ds:
            if ds[var].dtype == np.float64:
                ds[var] = ds[var].astype(np.float32)

        #######
        ## tree structure - optional?
        attrs = ds.attrs
        prepend = {
          "file_format": (
                "Hierarchical netCDF-4 file using groups. "
                "Use a group-aware reader or explicitly select a group/path "
                "(e.g., xarray.open_datatree(), MATLAB ncinfo/ncread)."
            )
        }
        attrs = prepend | attrs
        tree = xr.DataTree(dataset=xr.Dataset(attrs=attrs), name="radflux")

        tree["measurements"] = ds[[
            # "base_time",
            # "time_offset",
            # "time_bounds",

            "down_short_hemisp",
            "qc_down_short_hemisp",
            "down_short_hemisp_std",
            "down_short_hemisp_case_temp",

            "down_short_hemisp_spn1",
            "qc_down_short_hemisp_spn1",
            "down_short_hemisp_spn1_std",
            "down_short_diffuse_hemisp_spn1",
            "qc_down_short_diffuse_hemisp_spn1",
            "down_short_diffuse_hemisp_spn1_std",

            "down_short_diffuse_hemisp",
            "qc_down_short_diffuse_hemisp",

            "down_short_direct_hemisp",
            "qc_down_short_direct_hemisp",
            "down_short_direct_normal",

            "down_long_hemisp",
            "qc_down_long_hemisp",
            "down_long_hemisp_std",
            "down_long_hemisp_case_temp",
            "down_long_hemisp_dome_temp",

            "up_short_hemisp",
            "qc_up_short_hemisp",
            "up_short_hemisp_std",
            "up_short_hemisp_case_temp",

            "up_long_hemisp",
            "qc_up_long_hemisp",
            "up_long_hemisp_std",
            "up_long_hemisp_case_temp",
            "up_long_hemisp_dome_temp",

            "temp_mean",
            "qc_temp_mean",
            "temp_mean_std",

            "rh_mean",
            "qc_rh_mean",
            "rh_mean_std",

            "clean_flag",
        ]].drop_attrs()

        tree["clearsky"] = ds[[
            "mask_clear_sky_shortwave",
            "down_short_hemisp_clearsky",
            "down_short_diffuse_hemisp_clearsky",
        ]].drop_attrs()

        tree["cloud"] = ds[[
            "shortwave_cloud_fraction",
            "shortwave_cloud_fraction_uncorrected",
            "direct_beam_transmittance",
            "direct_beam_cloud_effect",
            "direct_beam_state",
        ]].drop_attrs()

        tree["diagnostics"] = ds[[
            "mask_normalized_global_magnitude",
            "mask_diffuse_magnitude",
            "mask_global_irradiance_temporal_gradient",
            "mask_normalized_diffuse_ratio_variability",
        ]].drop_attrs()

        tree["solar"] = ds[[
            "solar_zenith",
            "solar_azimuth",
            "solar_airmass",
            "solar_sun_earth_distance",
        ]].drop_attrs()


        tree["clearsky"]["mask_clear_sky_shortwave"].attrs["ancillary_variables"] = (
            "../diagnostics/mask_normalized_global_magnitude "
            "../diagnostics/mask_diffuse_magnitude "
            "../diagnostics/mask_global_irradiance_temporal_gradient "
            "../diagnostics/mask_normalized_diffuse_ratio_variability"
        )

        ds = tree

        ## Save the output file

        if save:
            row.p2f_out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(row.p2f_out)
        ds.close()
        out['ds'] = ds
        out['bbi'] = bbi
        out['row'] = row
        return out 
    
