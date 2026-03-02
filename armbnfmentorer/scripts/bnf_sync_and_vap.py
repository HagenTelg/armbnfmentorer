"""
Requirements:
- pvlib
- xarray
- pandas
- netcdf4  
- atmpy
- productomator
"""

import pandas as pd
import xarray as xr
import armbnfmentorer.vaps.bnfradsys43m60sS10c1 as vap
import productomator.lab as prolab
import armbnfmentorer.qc as bnfqc

def run(log_folder='/home/grad/htelg/.processlogs/',):
    print('Starting BNF Sync and VAP process...')
    print('====================================')
    reporter = prolab.Reporter('bnf_sync_and_vap', 
                            log_folder=log_folder,
                            reporting_frequency=(6, 'h'),
                            
                        )
    print('Syncing files from remote server...')
    print('====================================')
    bnfqc.rsync_bnfradsys(
                user_remote= 'hagentelg',
                path2localfld = '/nfs/stu3data2/bnf_radsys_data/',
                path2remote = ['/data/archive/bnf/bnfradsys*',] )
    run43(reporter = reporter)
    run2(reporter = reporter)
    return

def run2(path_in = '/nfs/stu3data2/bnf_radsys_data/bnfradsys2m60sS10.b1/',#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
        path_out = '/nfs/stu3data2/bnf_radsys_data/bnfradsys2m60sS10.c1/{version}/',#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
        reporter = None,):

    worker = vap.BnfRadsys2m60sS10C1(
            path_in,
            path_out,
            lambda name: pd.to_datetime(name.split('.')[2]),
            'bnfradsys2m60sS10.c1.{date}.nc',
            glob_pattern_in='*.nc',
            start=None,
            end=None,
            reporter=reporter,
            verbose=True,)
#     worker.process_row(iloc=0)
    worker.process(raise_errors = True)
    reporter.wrapup()
    return 

def run43(path_in = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1/',#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
        path_out = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/{version}/',#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
        reporter = None,):
    worker = vap.BnfRadsys43m60sS10C1(
            path_in,
            path_out,
            lambda name: pd.to_datetime(name.split('.')[2]),
            'bnfradsys43m60sS10.c1.{date}.nc',
            glob_pattern_in='*.nc',
            start=None,
            end=None,
            reporter=reporter,
            verbose=True,)
#     worker.process_row(iloc=0)
    worker.process(raise_errors = True)
    reporter.wrapup()
    return 
    

