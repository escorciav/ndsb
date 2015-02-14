import argparse
from random import shuffle
import json
import cPickle as pickle
import os

import numpy as np

import dataset as data

DFLT_ANGLES = '[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]'
DFLT_AUGMENT_PRM = '{"angles":' + DFLT_ANGLES + ', "max_pixel": 58}'
HELP_AUGMENT_PRM = 'Parameters used to augment the training set'
DFLT_OUTDIR = os.path.join(data.train_folder(), '..', 'aug')
HELP_OUTDIR = 'Output directory to allocate augmented dataset'

def main(data_outdir='', aug_prm={}, pctil=0.75, nfold=3.0, **kwargs):
    dataset_info = data.train_list(data.train_folder())
    aug_dataset_info = data.save_augmented_dataset(dataset_info[0],
                                                   dataset_info[1],
                                                   data_outdir, aug_prm)
    idx_train, idx_test = data.stratified_partition(aug_dataset_info[1],
                                                    1/nfold) 

    train_info = ([aug_dataset_info[0][i] for i in idx_train],
                  [aug_dataset_info[1][i] for i in idx_train])
    samples_per_cat = data.samples_per_categories(np.array(train_info[1]))
    max_samples = max(samples_per_cat)
    min_samples = np.mean([max_samples, np.percentile(samples_per_cat, pctil)])
    idx_train = data.balanced_dataset(train_info[1], int(min_samples))
    shuffle(idx_train)

    fid_tr = open(data_outdir + '_train.txt', 'w')
    fid_ts = open(data_outdir + '_val.txt', 'w')
    for i, v in enumerate(idx_train):
        if i < len(idx_test):
            new_line = '{0} {1}\n'.format(aug_dataset_info[0][idx_test[i]],
                                          str(aug_dataset_info[1][idx_test[i]]))
            fid_ts.write(new_line)
        new_line = '{0} {1}\n'.format(train_info[0][v], str(train_info[1][v]))
        fid_tr.write(new_line)
    fid_tr.close()
    fid_ts.close()

    with open(data_outdir + '.p', 'w') as fid:
        pickle.dump((aug_dataset_info, train_info, idx_train, idx_test), fid)
    return None

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-do', '--data_outdir', type=str, default=DFLT_OUTDIR,
                   help=HELP_OUTDIR)
    p.add_argument('-p', '--aug_prm', type=json.loads, default=DFLT_AUGMENT_PRM,
                   help=HELP_AUGMENT_PRM)
    main(**vars(p.parse_args()))

