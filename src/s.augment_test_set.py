import argparse
import cPickle as pickle
import os

import numpy as np

import dataset as data

DFLT_OUTDIR = os.path.join(data.test_folder(), '..', 'aug_test')
HELP_OUTDIR = 'Output directory to allocate augmented test set'
DFLT_MAX_PIXEL, DFLT_CROP_SZ = 58, 48
HELP_MAX_PIXEL = 'Desired image resolution'
HELP_CROP_SZ = 'CNN receptive field'

def main(data_outdir, max_pixel, crop_size, **kwargs):
    img_list = data.test_list()
    img_list = data.save_augmented_test_set(img_list, data_outdir,
                                            max_pixel, crop_size)

    with open(data_outdir + '.txt', 'w') as fid:
        for i in img_list:
            new_line = '{0}\t{1}\n'.format(i, '0')
            fid.write(new_line)

    with open(data_outdir + '.p', 'w') as fid:
        pickle.dump((img_list), fid)
    return None

if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-do', '--data_outdir', type=str, default=DFLT_OUTDIR,
                   help=HELP_OUTDIR)
    p.add_argument('-r', '--resolution', type=int, default=DFLT_MAX_PIXEL,
                   help=HELP_MAX_PIXEL, dest='max_pixel')
    p.add_argument('-c', '--crop_size', type=int, default=DFLT_CROP_SZ,
                   help=HELP_CROP_SZ)
    main(**vars(p.parse_args()))

