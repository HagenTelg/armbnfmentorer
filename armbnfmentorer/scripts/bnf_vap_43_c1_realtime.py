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

def run(path_in = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1/',#'/Users/htelg/data/arm/archive/bnf/bnfradsys43m60sS10.b1/'
        path_out = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1.realtime/{version}/',#'/Users/htelg/data/arm/vap/bnfradsys43m60sS10.c1/{version}/'
        radflux_setting = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/raflux_settings_0.1.toml',
        radflux_parameters_db = '/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/radflux_0.2.db',
        log_folder="/home/grad/htelg/.processlogs/",
        start=None,
        end=None,
        noofdays=60, #
        test=0,
        raise_errors=False,
        verbose=True,
        ):

    reporter = prolab.Reporter(
        "bnf_vap_43_c1_realtime",
        log_folder=log_folder,
        reporting_frequency=(6, "h"),
    )

    if end is None:
        end = pd.Timestamp.now()
    else:
        end = pd.to_datetime(end)
    if start is None:
        start = end - pd.to_timedelta(noofdays, "D")
    else:
        start = pd.to_datetime(start)

    vapi = bnfvap43.BnfRadsys43m60sS10C1(p2fld_in = path_in,
                                        p2fld_out = path_out, 
                                        date_from_name = lambda name: pd.to_datetime(name.split('.')[2]),
                                        output_file_format = 'bnfradsys43m60sS10.c1.{date}.nc',
                                        radflux_parameters_db = radflux_parameters_db, 
                                        path2raflux_setting = radflux_setting,
                                        real_time = True,
                                        start=start,
                                        end=end,
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
        "--path-in",
        "--path_in",
        default="/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.b1/",
        help="Folder containing input BNF radiation-system files.",
    )
    parser.add_argument(
        "--path-out",
        "--path_out",
        default="/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/{version}/",
        help="Output folder template for generated C1 files.",
    )
    parser.add_argument(
        "--radflux-setting",
        "--radflux_setting",
        default="/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/raflux_settings_0.1.toml",
        help="Path to the Radflux settings TOML file.",
    )
    parser.add_argument(
        "--radflux-parameters-db",
        "--radflux_parameters_db",
        default="/nfs/stu3data2/bnf_radsys_data/bnfradsys43m60sS10.c1/radflux_0.2.db",
        help="Path to the Radflux parameters database.",
    )
    parser.add_argument(
        "--log-folder",
        "--log_folder",
        default="/home/grad/htelg/.processlogs/",
        help="Folder in which process logs are stored.",
    )
    parser.add_argument("--start", help="Start date/time accepted by pandas.")
    parser.add_argument("--end", help="End date/time accepted by pandas.")
    parser.add_argument(
        "--noofdays",
        type=int,
        default=60,
        help="Number of days before end to process when start is omitted.",
    )
    parser.add_argument(
        "--test",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="Test mode: 0 processes all rows, 1 prints the workplan, 2 processes one row.",
    )
    parser.add_argument(
        "--raise-errors",
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
        path_in=args.path_in,
        path_out=args.path_out,
        radflux_setting=args.radflux_setting,
        radflux_parameters_db=args.radflux_parameters_db,
        log_folder=args.log_folder,
        start=args.start,
        end=args.end,
        noofdays=args.noofdays,
        test=args.test,
        raise_errors=args.raise_errors,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
