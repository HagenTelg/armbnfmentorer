"""
Script to run the BNF 43 m RadSys Ci VAP.

Requriements:
    # github
    - productomator
    - atmpy
    # conda
    - pandas
    - xarray
    - netCDF4
    - scipy
    - scikit-learn
    - pvlib
"""
import argparse
import armbnfmentorer.vaps.bnfradsys43m60sS10c1 as bnfvap43
import productomator.lab as prolab
import pandas as pd

# import atmPy.radiation.radflux.lab as atmradflux

def run(prefix = '/nfs',
        # path_in = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1',#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
        # path_out = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/{version}',#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
        # radflux_setting = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/raflux_settings_0.1.toml',
        # radflux_parameters_db = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/radflux_0.2.db',
        log_folder="/home/grad/htelg/.processlogs/",
        real_time = False,
        start=None,
        end=None,
        days=180, #
        test=0,
        raise_errors=False,
        verbose=True,
        ):


    path_in = f'{prefix}/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1'#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
    path_out = f'{prefix}/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/{{version}}'#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
    radflux_setting = f'{prefix}/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/raflux_settings_0.1.toml'
    radflux_parameters_db = f'{prefix}/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/radflux_0.2.db'

    if real_time:
        logfn = "bnf_vap_43_c1_realtime"
        path_out = path_out + "/near_real_time"
    else:
        logfn = "bnf_vap_43_c1"
        path_out = path_out + "/final"

    reporter = prolab.Reporter(
        logfn,
        log_folder=log_folder,
        reporting_frequency=(6, "h"),
    )


    vapi = bnfvap43.BnfRadsys43m60sS10C1(p2fld_in = path_in,
                                        p2fld_out = path_out, 
                                        file_name_format='*{date:%Y%m%d}*',                                        
                                        output_file_format = 'bnfradsys43m60sS10.c1.{date}.nc',
                                        radflux_parameters_db = radflux_parameters_db, 
                                        path2raflux_setting = radflux_setting,
                                        real_time = real_time,
                                        start=start,
                                        end=end,
                                        days = days,
                                        # start='2026-07-01',
                                        # end='2027-05-01',
                                        verbose = verbose,
                                        reporter = reporter,
                                        )
    if test == 1:
        print(vapi.workplan)
        return vapi.workplan
    
    elif test == 2:
        vapi.process_row(iloc=0, save=False)
        return vapi
    else:
        vapi.process(raise_errors=raise_errors)

    reporter.wrapup()
    return vapi

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the BNF 43 m radiation-system C1 VAP."
    )
    parser.add_argument(
        "--prefix",
        default="/nfs",
        help="Filesystem prefix containing the BNF data tree.",
    )
    parser.add_argument(
        "--log_folder",
        default="/home/grad/htelg/.processlogs/",
        help="Folder in which process logs are stored.",
    )
    parser.add_argument("--start", help="Start date/time accepted by pandas.")
    parser.add_argument("--end", help="End date/time accepted by pandas.")
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Number of days before end to process when start is omitted.",
    )
    parser.add_argument(
        "--real_time",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write near-real-time output and use near-real-time reporting.",
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Test mode: 0 processes all rows, 1 prints the workplan, 2 processes one row.",
    )
    parser.add_argument(
        "--raise_errors",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Propagate processing errors instead of continuing.",
    )
    parser.add_argument(
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable verbose processing output.",
    )
    args = parser.parse_args(argv)
    return run(
        prefix=args.prefix,
        log_folder=args.log_folder,
        real_time=args.real_time,
        start=args.start,
        end=args.end,
        days=args.days,
        test=args.test,
        raise_errors=args.raise_errors,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
