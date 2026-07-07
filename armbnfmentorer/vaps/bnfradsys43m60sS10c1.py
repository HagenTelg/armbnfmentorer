import json
import pathlib as pl
import productomator.worker as prowo
import sqlite3
import xarray as xr
import pandas as pd
import atmPy.radiation.retrievals.broadband_shortwave_radiation as atmbrad
import atmPy.general.measurement_site as atmsite
import socket



default_clearsky_params = {'nsw_exp': 1.202095545434091,
                            'nsw_min': 800,
                            'nsw_max': 1400,
                            'ndr_exp': -0.6827046137686424,
                            'mu0_min': 0.05,
                            'diffuse_max_coeff': 150,
                            'diffuse_max_exp': 0.5,
                            'max_dsw_dt': 8,
                            'ndr_std_max': 0.005,
                            'ndr_window': 11,}


radflux_parameter_table = 'radflux_parameters'
radflux_parameter_names = tuple(default_clearsky_params) + (
    'ndr_std_max_estimated',
    'diffuse_max_coeff_estimated',
    'diffuse_max_exp_estimated',
    'max_dsw_dt_estimated',
)
radflux_table_columns = {
    'row_timestamp': 'TEXT PRIMARY KEY',
    'input_file': 'TEXT NOT NULL',
    'next_day_needed': 'BOOLEAN',
    # 'output_file': 'TEXT NOT NULL',
    'processed_at': 'TEXT NOT NULL',
    'process_version': 'TEXT NOT NULL',
    'processing_server': 'TEXT NOT NULL',
    'clear_sky_params_optimized': 'TEXT',
    # 'parameters_json': 'TEXT NOT NULL',
    **{name: 'REAL' for name in radflux_parameter_names},
}


class RadfluxParameterDatabase:
    def __init__(self, radflux_parameters_db, create_if_not_exist=False, version=None, verbose=False):
        self.radflux_parameters_db = pl.Path(radflux_parameters_db)
        self.verbose = verbose
        self.version = version
        if not self.radflux_parameters_db.exists() and not create_if_not_exist:
            raise FileNotFoundError(f"{self.radflux_parameters_db} does not exist and create_if_not_exist is False.")
        elif not self.radflux_parameters_db.exists() and create_if_not_exist:
            self.radflux_parameters_db.parent.mkdir(parents=True, exist_ok=True)
        with self.connect_database() as conn:
            self.ensure_parameter_table(conn)

    def connect_database(self):
        conn = sqlite3.connect(self.radflux_parameters_db)
        conn.row_factory = sqlite3.Row
        return conn

    def ensure_parameter_table(self, conn):
        parameter_columns = ',\n                '.join(
            f'{name} {dtype}' for name, dtype in radflux_table_columns.items()
        )
        table_exists = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            (radflux_parameter_table,),
        ).fetchone() is not None
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {radflux_parameter_table} (
                {parameter_columns}
            )
            """)
        existing_columns = {
            row['name']
            for row in conn.execute(f'PRAGMA table_info({radflux_parameter_table})')
        }
        if table_exists and 'row_timestamp' not in existing_columns:
            raise ValueError(
                f'{self.radflux_parameters_db} contains a '
                f'{radflux_parameter_table} table without row_timestamp'
            )
        for name, dtype in radflux_table_columns.items():
            if name not in existing_columns:
                dtype = dtype.replace(' NOT NULL', '').replace(' PRIMARY KEY', '')
                conn.execute(
                    f'ALTER TABLE {radflux_parameter_table} '
                    f'ADD COLUMN {name} {dtype}'
                )

    @staticmethod
    def timestamp2dbformat(timestamp):
        return pd.to_datetime(timestamp).isoformat()

    @staticmethod
    def _database_value(value):
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, 'item'):
            value = value.item()
        return value
    
    def dump_radflux_parameters(self):
        with self.connect_database() as conn:
            self.ensure_parameter_table(conn)
            df = pd.read_sql_query(
                f'SELECT * FROM {radflux_parameter_table} ORDER BY row_timestamp DESC',
                conn,
                index_col='row_timestamp',
            )
        return df

    def read_previous_valid_clearsky_parameters(self, timestamp):
        """Retrieves the last set of clearsky parameters before the given timestamp."""
        row_timestamp = self.timestamp2dbformat(timestamp)
        with self.connect_database() as conn:
            self.ensure_parameter_table(conn)
            previous = conn.execute(
                f"""
                SELECT *
                FROM {radflux_parameter_table}
                WHERE row_timestamp < ?
                  AND clear_sky_params_optimized = 'True'
                ORDER BY row_timestamp DESC
                LIMIT 1
                """,
                (row_timestamp,),
            ).fetchone()
            self.tp_prvious = previous

        if previous is None:
            if self.verbose:
                print(f'No previous optimized clearsky parameters found for {row_timestamp}.')
            self.tp_previous_radflux_parameters_record = None
            return None

        self.tp_previous_radflux_parameters_record = dict(previous)
        # parameters = json.loads(previous['parameters_json'])
        # parameters = {
        #     name: value
        #     for name, value in parameters.items()
        #     if value is not None
        # }
        # return {**default_clearsky_params, **parameters}
        return dict(previous)

    def write_radflux_parameters(self,
                                  row,
                                  clearsky_parameters,
                                  processing_date,
                                  processing_server,
                                  clear_sky_params_optimized,
                                  next_day_needed
                                  ):
        # cleanup the parameters to ensure they are JSON serializable and handle NaN values
        parameters = {
            name: self._database_value(value)
            for name, value in clearsky_parameters.items()
        }
        values = {
            'row_timestamp': self.timestamp2dbformat(row.name),
            'input_file': str(row.p2f_in),
            'next_day_needed': next_day_needed,
            # 'output_file': str(row.p2f_out),
            'processed_at': processing_date,
            'process_version': self.version,
            'processing_server': processing_server,
            'clear_sky_params_optimized': clear_sky_params_optimized,
            # 'parameters_json': json.dumps(parameters, sort_keys=True),
        }
        values.update({
            name: parameters.get(name)
            for name in radflux_parameter_names
        })
        columns = tuple(values)
        placeholders = ', '.join(['?'] * len(columns))
        update_columns = ', '.join(
            f'{column} = excluded.{column}'
            for column in columns
            if column != 'row_timestamp'
        )

        self.tp_columns = columns
        self.tp_placeholders = placeholders
        self.tp_values = values
        with self.connect_database() as conn:
            self.ensure_parameter_table(conn)
            conn.execute(
                f"""
                INSERT INTO {radflux_parameter_table}
                    ({', '.join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(row_timestamp) DO UPDATE SET
                    {update_columns}
                """,
                tuple(values[column] for column in columns),
            )
    def delete_rows_on_date(self, date, find_only = False):
        """Deletes all rows in the radflux parameter database for a specific date."""
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        with self.connect_database() as conn:
            self.ensure_parameter_table(conn)
            if find_only:
                rows = conn.execute(
                    f"""
                    SELECT *
                    FROM {radflux_parameter_table}
                    WHERE DATE(row_timestamp) = ?
                    """,
                    (date_str,),
                ).fetchall()
                return [dict(row) for row in rows]
            else:
                conn.execute(
                    f"""
                    DELETE FROM {radflux_parameter_table}
                    WHERE DATE(row_timestamp) = ?
                    """,
                    (date_str,),
                )
        


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
        kwargs['database'] = (self.radflux_parameters_db, radflux_parameter_table, 'row_timestamp', 'None')# 'input_file')
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
        self.radflux_db = RadfluxParameterDatabase(self.radflux_parameters_db, create_if_not_exist=True, verbose=self.verbose, version = self.version)

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
        self.radflux_parameters_db = pl.Path(radflux_parameters_db)
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
        self._initialize_radflux_database()



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

        clearsky_parameters = self._read_previous_clearsky_parameters(row)
        # self.tp_previous_clearsky_parameters = clearsky_parameters

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
        bbi.optimize_clearsky_parameters()
        current_clearsky_parameters = bbi.clearsky_parameters
        # self.tp_current_clearsky_parameters = current_clearsky_parameters
        self.tp_bbi = bbi
    
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
                        'datetime':'time'})        
        #########
        # Format the dataset attributes
        #########
        dropattrs = ['history','doi','averaging_interval','calib_info','command_line',
                    # '','','','','',
                    ]
        for a in dropattrs:
            ds.attrs.pop(a)

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
