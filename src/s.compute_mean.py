import argparse
import cPickle as pickle

import caffe
import numpy as np
from skimage.io import imread
from skimage.transform import resize

DFLT_OUTFILE = 'data/mean.bynaryproto'
HELP_OUTFILE = 'Name of the protobinary file with the mean'
DFLT_INFILE = 'data/aug.p'
HELP_INFILE = 'Name of pickle file with info about dataset'
DFLT_MAX_PIXEL = 58
HELP_MAXPIXEL = 'Max number of pixel in each dimension'

def average_image(list_of_files, max_pixel):
    """Compute the average image W x H x C of a collection of images
    """
    avg, n = resize(imread(list_of_files[0]), (max_pixel, max_pixel)), 1
    avg = np.array(avg, dtype=np.float32)
    for i, v in enumerate(list_of_files[1:]):
        try:
            im = resize(imread(v), (max_pixel, max_pixel))
            avg += im
            n += 1
        except Exception, e:
            print img, e
    return avg / float(n)

def main(in_file, out_file, max_pixel=58, **kwargs):
    with open(in_file, 'r') as fid:
        info = pickle.load(fid)
    mean_img = average_image(info[1][0], max_pixel)
    save_binaryproto(out_file, mean_img.reshape((1, 1, max_pixel, max_pixel)))
    return None

def save_binaryproto(filename, arr):
    """Save numpy array as binary protocol buffer
    """
    assert arr.ndim == 4, 'Caffe need blobs of 4-dim'
    blob = caffe.io.array_to_blobproto(arr)
    with open(filename, "wb") as fid:
        fid.write(blob.SerializeToString())
    return None

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-o', '--out_file', type=str, default=DFLT_OUTFILE,
                   help=HELP_OUTFILE)
    p.add_argument('-p', '--in_file', type=str, default=DFLT_INFILE,
                   help=HELP_INFILE)
    p.add_argument('-pxl', '--max_pixel', type=int, default=DFLT_MAX_PIXEL,
                   help=HELP_MAXPIXEL)
    main(**vars(p.parse_args()))
