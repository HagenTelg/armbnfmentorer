import productomator.worker as prowo
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


class BnfRadsys2m60sS10C1(prowo.Workplanner):
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
       """
    def __init__(self, *args, **kwargs):
        self.version = '0.1'
        kwargs['version'] = self.version
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

    def process_row(self, row = None, iloc = None, loc = None):
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

        # #####
        # # get last processed instance - usefull if the next day depends on the last day
        # ######
        # lastrow = self.get_last_row_before_workplan()
        # if isinstance(lastrow, type(None)):
        #     # assert(False), 'set defaults?'
        #     clearsky_parameters = default_clearsky_params
        # else:
        #     dslast = xr.open_dataset(lastrow.p2f_out)
        #     dslast = dslast.rename({'down_short_hemisp': 'global_horizontal',
        #                     'down_short_diffuse_hemisp': 'diffuse_horizontal',
        #                     'down_short_direct_hemisp': 'direct_horizontal',
        #                     'time':'datetime'})            
        #     bbi = atmbrad.CombinedGlobalDiffuseDirect(dslast, site= self.site, verbose = self.verbose)
        #     clearsky_parameters = bbi.clearsky_parameters
        #     dslast.close()

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
        # bbi.clearsky_parameters = clearsky_parameters
        # bbi.optimize_clearsky_parameters()
        # self.tp_bbi = bbi
    
        ########
        # Format the dataset
        ########
        dropvar = ['lat', 'lon', 'alt', 'zenith_geometric', 'apparent_elevation', 'elevation', 'equation_of_time', 'mu0',
                # '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '',
                ]
        
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
        ds.attrs['datastream'] = 'bnfradsys2m60sS10.c1'
        ds.attrs['processing_date'] = pd.Timestamp.now().isoformat()
        ds.attrs['processing_server'] = socket.gethostname()



        ## Save the output file
        row.p2f_out.parent.mkdir(parents=True, exist_ok=True)
        ds.to_netcdf(row.p2f_out)
        ds.close()
        return ds
