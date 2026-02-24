
import pandas as pd
import xarray as xr
import armbnfmentorer.vaps.bnfradsys43m60sS10c1 as vap
import productomator.lab as prolab
import armbnfmentorer.qc as bnfqc

def run(path_in = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1/',#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
        path_out = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/{version}/',#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
        log_folder='/home/grad/htelg/.processlogs/',):

    reporter = prolab.Reporter('bnf_sync_and_vap', 
                            log_folder=log_folder,
                            reporting_frequency=(6, 'h'),
                            
                        )

    bnfqc.rsync_bnfradsys(
                user_remote= 'hagentelg',
                path2localfld = '/nfs/stu3data2/bnf_radsys_data/',
                path2remote = ['/data/archive/bnf/bnfradsys*',] )

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
    
    worker.process()

    reporter.wrapup()
    return 
    

