"""
This peforms the Radflux analysis on the tower radsys data. It populates the database but does not generate the files
todo: upcate the below
Requirements:
- pvlib
- xarray
- pandas
- netcdf4  
- atmpy
- sklearn
- productomator
"""

import argparse

import pandas as pd
import xarray as xr
import armbnfmentorer.vaps.bnfradsys43m60sS10c1 as vap43
# import armbnfmentorer.vaps.bnfradsys2m60sS10c1 as vap2
import productomator.lab as prolab
import armbnfmentorer.qc as bnfqc

def run(log_folder='/home/grad/htelg/.processlogs/',
        path_in = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1/',#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
        # path_out = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/{version}/',#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
        path2raflux_setting = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/raflux_settings_0.1.toml',
        radflux_parameters_db = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/radflux_{version}.db',
        reporter = None,
        verbose = False,
        raise_errors = False,
        ):
    
    reporter = prolab.Reporter('bnf_vap_radflux', 
                            log_folder=log_folder,
                            reporting_frequency=(6, 'h'),)
    print('Processing BnfRadsys43m60sS10C1Radflux...')
    print('====================================')
    worker = vap43.BnfRadsys43m60sS10C1Radflux(
            p2fld_in = path_in,
            path2raflux_setting = path2raflux_setting,
            date_from_name = lambda name: pd.to_datetime(name.split('.')[2]),
            radflux_parameters_db=radflux_parameters_db,
            start=None,
            end=None,
            reporter=reporter,
            verbose=verbose,
            )
#     worker.process_row(iloc=0)
    worker.combine_masterplan_duplicates()
    worker.process(raise_errors = raise_errors)
    reporter.wrapup()
    return 


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Run the Radflux analysis on BNF tower radiation-system data.'
    )
    parser.add_argument(
        '--log-folder',
        '--log_folder',
        default='/home/grad/htelg/.processlogs/',
        help='Folder in which process logs are stored.',
    )
    parser.add_argument(
        '--path-in',
        '--path_in',
        default='/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1/',
        help='Folder containing input BNF radiation-system files.',
    )
    parser.add_argument(
        '--path2raflux-setting',
        '--path2raflux_setting',
        default='/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/raflux_settings_0.1.toml',
        help='Path to the Radflux settings TOML file.',
    )
    parser.add_argument(
        '--radflux-parameters-db',
        '--radflux_parameters_db',
        default= '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/radflux_{version}.db',
        help='Path to the Radflux parameters database.',
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument(
        '--raise-errors',
        '--raise_errors',
        action='store_true',
        help='Propagate processing errors instead of continuing.',
    )
    args = parser.parse_args(argv)
    return run(
        log_folder=args.log_folder,
        path_in=args.path_in,
        path2raflux_setting=args.path2raflux_setting,
        radflux_parameters_db=args.radflux_parameters_db,
        verbose=args.verbose,
        raise_errors=args.raise_errors,
    )


if __name__ == '__main__':
    main()
