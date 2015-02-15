import argparse
import os

import caffe
import leveldb
import numpy as np
import pandas as pd

import dataset as data

DFLT_INPUT = 'data/nn00/test_65001'
HELP_INPUT = 'Fullpath of file/folder with results'
DFLT_NKEYS = 1304000
HELP_NKEYS = 'Number of keys in the database'
HELP_OUTPUT ='Name of csv file for submission'

#TODO: save csv

def average_crops(arr):
    """Perform the average of arr every 10 rows
    """
    avg = np.empty((arr.shape[0]/10, arr.shape[1]), np.float32)
    for i in xrange(0, arr.shape[0]/10):
        idx1, idx2 = i * 10, (i + 1) * 10
        avg[i, :] = np.mean(arr[idx1:idx2, :], axis=0)
    return avg

def dump_results(filename, arr):
    """Save arr as CSV
    """
    labels = data.label_list(data.train_folder())
    img_names = [os.path.basename(i) for i in data.test_list()]
    data_frame = pd.DataFrame(arr, columns=labels, index=img_names)
    data_frame.index.name = 'image'
    data_frame.to_csv(filename)
    return None    

def get_blob_size(db, key='0'):
    """Return blob size
    """
    val = db.Get(key)
    datum = caffe.io.caffe_pb2.Datum()
    datum.ParseFromString(val)
    return (datum.channels, datum.height, datum.width)

def levedb_to_array(filename, n_keys):
    """Return caffe blobs stored on leveldb as ndarray
    """
    db = leveldb.LevelDB(filename)
    blob_sz = get_blob_size(db)
    dim = blob_sz[0] * blob_sz[1] * blob_sz[2]
    db_mem = np.empty((n_keys, dim), np.float32)
    for key, val in db.RangeIter():
        datum = caffe.io.caffe_pb2.Datum()
        datum.ParseFromString(val)
        arr = caffe.io.datum_to_array(datum)
        db_mem[int(key), :] = arr.flatten()
    return db_mem

def main(output_file, input_file, n_keys, **kwargs):
    aug_pred = levedb_to_array(input_file, n_keys)
    pred = average_crops(aug_pred)
    dump_results(output_file, pred)
    return None

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input_file', type=str, default=DFLT_INPUT,
                   help=HELP_INPUT)
    p.add_argument('-n', '--n_keys', type=int, default=DFLT_NKEYS,
                   help=HELP_NKEYS)
    p.add_argument('output_file', type=str, help=HELP_OUTPUT)
    main(**vars(p.parse_args()))

