import argparse

import numpy as np
import pandas as pd

from s_save_csv import dump_results

DFLT_TYPE = 'avg'
HELP_TYPE = 'Kind of pooling on the predictions'
HELP_OUTFILE = 'Name of output file'
HELP_INFILES = 'set of csv-files to pool'

def feat_pooling(X, pool_stg='avg', axis=0):
    stg = pool_stg.lower()
    if stg == 'avg':
        Y = X.mean(axis)
    elif stg == 'max':
        Y = X.max(axis)
    else:
        raise('Unknown pooling strategy')
    return Y

def read_cat_csvfiles(csv_files):
    pred_list = []
    for fid in csv_files:
        df = pd.io.parsers.read_csv(fid, index_col=0)
        pred_list.append(np.array(df, np.float32))
    cat_pred = np.asarray(pred_list) 
    return cat_pred

def main(csv_files, output_file, pool_stg):
    pred_arr = read_cat_csvfiles(csv_files)
    result = feat_pooling(pred_arr)
    dump_results(output_file, result)
    return None

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-t', '--pool_stg', type=str, default=DFLT_TYPE,
                   help=HELP_TYPE)
    p.add_argument('output_file', type=str, help=HELP_OUTFILE)
    p.add_argument('csv_files', type=str, nargs='+', help=HELP_INFILES)
    main(**vars(p.parse_args()))

