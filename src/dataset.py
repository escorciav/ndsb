import glob
import os

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize, rotate

def data_augmentation(img_name, angles=range(0, 181, 30),
                      max_pixel=60, prefix=''):
    """Create many images from one training sample
    """
    n_images, img_list = len(angles), []
    img = imread(img_name)
    for th in angles:
        new_img = rotate(img, th, cval=1.0, resize=True)
        new_img = resize(new_img, (max_pixel, max_pixel))
        img_list.append(prefix + '_' + str(th) +  '.jpg')
        imsave(img_list[-1], new_img)
    return img_list

def folder_content1(folder, sort_flg=True):
    """Return all files on the first level of a folder
    """
    content = glob.glob(os.path.join(folder, '*'))
    if sort_flg:
        content.sort(key=str.lower)
    return content

def img_resolution_list(image_list):
   """Return a ndarray with the resolution (along axis 1) of a list of images
   """
   res_list = np.empty((len(image_list), 2), np.int32)
   for i, v in enumerate(image_list):
       try:
           img = imread(v)
       except:
           raise 'Cannot read image:', v
       res_list[i, :] = img.shape
   return res_list


def label_list(folder):
    """Return categories inside the passed folder
    
    Parameters
    ----------
    folder : string

    Returns
    -------
    labels : list

    Note
    ----
    It assumes that folder has a 1to1 mapping between subfolders and categories
    and there is no more files
    """
    labels = folder_content1(folder)
    for i, v in enumerate(labels):
        dummy, labels[i] = os.path.split(v)
    return labels

def root_folder():
    """Returns the absolute path of the folder where data is located
    """
    root, dummy =  os.path.split(os.path.realpath(__file__))
    return os.path.join(root, '..', 'data')

def test_folder():
    """Returns the absolute path of the train folder
    """
    root = root_folder()
    return os.path.join(root, 'test')

def test_list(folder):
    """Return a list with fullpath name of images on the testing set
    """
    return folder_content1(test_folder())

def train_folder():
    """Returns the absolute path of the test folder
    """
    root = root_folder()
    return os.path.join(root, 'train')

def train_list(folder):
    """Return two list: (1) fullpath name of images and (2) indexes of their
       categories (0-indexed)

    Parameters
    ----------
    folder : string

    Returns
    -------
    image_fullpath_names : list
    image_labels : list

    Note
    ----
    It assumes that folder has a 1to1 mapping between subfolders and categories
    and there is no more files
    """
    image_fullpath_names, image_labels = [], []
    labels = folder_content1(folder)
    for i, v in enumerate(labels):
        tmp = folder_content1(v)
        image_fullpath_names += tmp
        image_labels += [i] * len(tmp)
    return image_fullpath_names, image_labels

def samples_per_categories(argument):
   """Compute the number of samples per category
   """
   if isinstance(argument, basestring):
       samples_per_cat = []
       labels = folder_content1(argument)
       for i in labels:
           samples_per_cat.append(len(folder_content1(i, False)))
   else:
       raise 'Sorry, Im working on that'
   return samples_per_cat

