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
    'output_file': 'TEXT NOT NULL',
    'processed_at': 'TEXT NOT NULL',
    'process_version': 'TEXT NOT NULL',
    'processing_server': 'TEXT NOT NULL',
    'clear_sky_params_optimized': 'TEXT',
    'parameters_json': 'TEXT NOT NULL',
    **{name: 'REAL' for name in radflux_parameter_names},
}


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

    def _initialize_radflux_database(self):
        self.radflux_parameters_db.parent.mkdir(parents=True, exist_ok=True)
        with self._connect_radflux_database() as conn:
            self._ensure_radflux_parameter_table(conn)

    def _connect_radflux_database(self):
        conn = sqlite3.connect(self.radflux_parameters_db)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_radflux_parameter_table(self, conn):
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
    def _row_timestamp(row):
        return pd.to_datetime(row.name).isoformat()

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

    def _read_previous_clearsky_parameters(self, row):
        row_timestamp = self._row_timestamp(row)
        with self._connect_radflux_database() as conn:
            self._ensure_radflux_parameter_table(conn)
            previous = conn.execute(
                f"""
                SELECT *
                FROM {radflux_parameter_table}
                WHERE row_timestamp < ?
                ORDER BY row_timestamp DESC
                LIMIT 1
                """,
                (row_timestamp,),
            ).fetchone()

        if previous is None:
            if self.verbose:
                print(f'No previous clearsky parameters found for {row_timestamp}. Returning default parameters.')
            self.tp_previous_radflux_parameters_record = None
            return default_clearsky_params.copy()

        self.tp_previous_radflux_parameters_record = dict(previous)
        parameters = json.loads(previous['parameters_json'])
        parameters = {
            name: value
            for name, value in parameters.items()
            if value is not None
        }
        return {**default_clearsky_params, **parameters}

    def _write_radflux_parameters(self,
                                  row,
                                  clearsky_parameters,
                                  processing_date,
                                  processing_server,
                                  clear_sky_params_optimized):
        parameters = {
            name: self._database_value(value)
            for name, value in clearsky_parameters.items()
        }
        values = {
            'row_timestamp': self._row_timestamp(row),
            'input_file': str(row.p2f_in),
            'output_file': str(row.p2f_out),
            'processed_at': processing_date,
            'process_version': self.version,
            'processing_server': processing_server,
            'clear_sky_params_optimized': clear_sky_params_optimized,
            'parameters_json': json.dumps(parameters, sort_keys=True),
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

        with self._connect_radflux_database() as conn:
            self._ensure_radflux_parameter_table(conn)
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
        
        if iloc is not None:
            row = self.workplan.iloc[iloc]
        elif loc is not None:
            row = self.workplan.loc[loc]
        self.tp_row = row

        clearsky_parameters = self._read_previous_clearsky_parameters(row)
        self.tp_previous_clearsky_parameters = clearsky_parameters

        #######
        ## Open input files
        #######
        try:
            ds = xr.open_dataset(row.p2f_in)
        except:
            print(row.p2f_in)
            raise

        ds = ds.rename({'down_short_hemisp': 'global_horizontal',
                        'down_short_diffuse_hemisp': 'diffuse_horizontal',
                        'down_short_direct_hemisp': 'direct_horizontal',
                        'time':'datetime'})
        self.tp_ds = ds.copy()
        ## Do some processing here, e.g. add attributes, format the dataset, etc.
        bbi = atmbrad.CombinedGlobalDiffuseDirect(ds, site= self.site, verbose = self.verbose)
        bbi.direct_normal_irradiation #just to trigger the calculation of direct normal
        bbi.clearsky_parameters = clearsky_parameters
        bbi.optimize_clearsky_parameters()
        current_clearsky_parameters = bbi.clearsky_parameters
        self.tp_current_clearsky_parameters = current_clearsky_parameters
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
        if ds.clear_sky_params_optimized == "Not enough clear sky points":
            pass
        elif ds.clear_sky_params_optimized == "True":
            self._write_radflux_parameters(
                row,
                current_clearsky_parameters,
                processing_date,
                processing_server,
                ds.attrs.get('clear_sky_params_optimized'),
            )
        else:
            raise ValueError(f"Unexpected value for clear_sky_params_optimized: {ds.attrs.get('clear_sky_params_optimized')}")
        
        if save:
            row.p2f_out.parent.mkdir(parents=True, exist_ok=True)
            ds.to_netcdf(row.p2f_out)
        ds.close()
        out = {}
        out['ds'] = ds
        out['bbi'] = bbi
        out['row'] = row
        return out 
