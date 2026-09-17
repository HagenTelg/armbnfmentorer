"""
Requirements:
- productomator
- pandas
"""

import argparse

import productomator.lab as prolab
import armbnfmentorer.qc as bnfqc

def run(log_folder='/home/grad/htelg/.processlogs/',):
    
    print('Starting BNF Sync and VAP process...')
    print('====================================')
    reporter = prolab.Reporter('bnf_sync_up', 
                            log_folder=log_folder,
                            reporting_frequency=(6, 'h'),)
    print('Syncing files from remote server...')
    print('====================================')
    try:
        bnfqc.rsync_bnfradsys(
                    user_remote= 'hagentelg',
                    path2localfld = '/nfs/stu3data2/bnf_radsys_data/',
                    path2remote = ['/data/archive/bnf/bnfradsys*',] )
        reporter.clean_increment()
    except:
        reporter.errors_increment()

    try:
        bnfqc.rsync_bnfradsys(
                    user_remote= 'hagentelg',
                    path2localfld = '/nfs/stu3data2/bnf_radsys_data/',
                    path2remote = ['/data/datastream/bnf/bnfradsys*.00',] )
        reporter.clean_increment()
    except:
        reporter.errors_increment()

    reporter.wrapup()
    return


def main(argv=None):
    parser = argparse.ArgumentParser(description='Sync BNF radiation-system files.')
    parser.add_argument(
        '--log-folder',
        '--log_folder',
        default='/home/grad/htelg/.processlogs/',
        help='Folder in which process logs are stored.',
    )
    args = parser.parse_args(argv)
    return run(log_folder=args.log_folder)


if __name__ == '__main__':
    main()
