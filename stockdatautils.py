import os
import numpy as np
import matplotlib
matplotlib.use('TKAgg', warn=False)
import matplotlib.pyplot as plt

"""
Utils that manage our data. Returns everything as a list of numpy arrays 

General format:
[num_pics, width, height, 3] # width = height = 256

Assumes we are drawing from .npz (standardized) numpy versions of the pics
"""


def load_data_numpy(dir, ext):
    """Load all files that end with ext=ext into a numpy arr"""
    fnames = _get_valid_fnames(dir, ext)
    return _load_numpy_from_filenames(fnames)


def load_gray_numpy(dir, ext):
    fnames = _get_valid_fnames(dir, ext)
    tmp =  _load_gray_from_filenames(fnames)
    tmp = _standardize_data(tmp)
    print(tmp.shape)
    return tmp


def _shape_correct(one_img):
    """expands_dim of one_img"""
    return np.expand_dims(one_img, axis=0)


def _select_random(fnames, num):
    """Takes theta(N) time :/
    Can sample same image twice
    """
    fs = []
    for i in range(num):
        n = np.random.randint(i, len(fnames))
        fs.append(fnames[n])
    assert len(fs) == num, "oops, drew too many samples"
    return fs

def plot_sample(model, eval_dir, ext, num=10, savedir='samplepics/'):
    """Take a model, and a directory, create a directory of images to compare"""
    fnames = _get_valid_fnames(eval_dir, ext)
    fnames = _select_random(fnames, num)
    truth = _load_numpy_from_filenames(fnames)

    # predict on each image
    out = []
    for img in truth:
        out.append(model.predict(_shape_correct(img)))

    # if save dir doesn't exist, make it
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # save images, rename them to a number
    for (i, fname, pic, pred) in zip(range(len(fnames)), fnames, truth, out):
        pic = np.squeeze(pic)
        pred = np.squeeze(pred)
        picname = os.path.join(savedir, 'pic-{}a'.format(i))
        predictname = os.path.join(savedir, 'pic-{}b'.format(i))
        # standardize the predicted data, so it can be plotted
        plt.imsave(picname, pic, cmap='gray')
        plt.imsave(predictname, pred, cmap='gray')


def _unpack_numpy(fname):
    return np.load(fname)

def _load_numpy_from_filenames(fnames):
    return np.array([_unpack_numpy(fname) for fname in fnames])

def _load_gray_from_filenames(fnames):
    return np.array([_rgb2gray(_unpack_numpy(fname)) for fname in fnames])

def _get_valid_fnames(dir, ext):
    """Returns filenames that end with ext"""
    fnames = [os.path.join(dir, f) for f in os.listdir(dir) if f.endswith(ext)]
    return fnames

def _rgb2gray(img):
    """Returns a grayscale img (width, height, 1)"""
    img = _rgb(img)
    return np.expand_dims(img, axis=-1)

def _rgb(img):
    """Returns a rank-2 tensor representing grayscale image"""
    return np.dot(img[...,:3], [0.299, 0.587, 0.114])

