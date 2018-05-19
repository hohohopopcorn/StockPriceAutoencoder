import numpy as np
import os
from os.path import join
import cv2
import skimage
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler

"""This generator currently takes forever... might be worth caching somehow...

Should be an infinite stream that loops after all the photos have been passed in
    - Each new loop has a new ordering to the batches it yields

"""

class PixDataset(Dataset):
    DATAPATH="../small_zips/small/ios/"
    TRAIN="training_features"

    def __init__(self, directory, input_end='.npz', target_end='.npz', bsize=32):
        """
        Only performs augmentations on numpy files (.npy)
            assumes .npy files are stored with 'features' as their only key

        Throws ValueError if directory is invalid

        :param directory: path to datafiles
        All data should have same base file name.
        Input files differ from target files only by file extension

        :param input_end: file extension of input files

        :param target_end: file extension of target files

        :param bsize: batchsize

        """
        if not self._check_directory_valid(directory):
            raise ValueError("Directory {} is not a valid directory".format(directory))

        self.directory = directory
        self.input_end = input_end
        self.target_end = target_end

        # find and store the input_files & target_files based upon
        # input_end and target_end
        self.input_files = []
        self.target_files = []
        self.length = 0
        self.learn_data_dist()

    @staticmethod
    def _check_directory_valid(directory):
        """
        :param directory: path 2 directory of **numpy** files
        :return: True if directory has numpy files
        """
        if os.path.isdir(directory):
            """Return true if > 1 .gui & .npy in dir"""
            return True
        else:
            return False

    def learn_data_dist(self):
        """Stores the input files and target files based upon
        the input and output endings of the files, as set in the
        constructor.

        Also sets self.length:
        The length of the generator = # files ending with input_end
        """
        files = os.listdir(self.directory)
        input_files = []
        target_files = []
        for f in files:
            if f.endswith(self.input_end):
                input_files.append(f)
            if f.endswith(self.target_end):
                target_files.append(target_files)

        self.input_files = input_files
        self.target_files = target_files
        self.length = len(self.input_files)

    def summary(self):
        """
        :return:
         "path2dir: #num files in dir
            gui: #num files with .gui in dir
            npy: #num files with .png in dir"
        """
        rep = "Directory: {}\n".format(self.directory)
        in_fs = "\t# Input files ({}): {}\n".format(self.input_end, len(self.input_files))
        out_fs = '\t# Target files ({}): {}\n'.format(self.target_end, len(self.target_files))
        return "not yet implemented"

    def __len__(self):
        return self.length

    def __str__(self):
        return self.summary()

    def __iter__(self):
        """
        Iterates over the training directory, and yields batches of [ [x, y], [x_1, y_2], ... ] up to batchsize

        :return: generator that performs self.augments on data
                returns bsize tuples of (input, label)
        """
        # Create paths to the train directory
        train_dir = self.directory
        list_of_np = []

        # Create a list of files which are numpy arrays
        list_of_files = [fname for fname in os.listdir(train_dir) if fname[-3:]=='npz']
        list_of_files = np.array(list_of_files)

        # Do an initial shuffle
        np.random.shuffle(list_of_files)

        file_end = len(list_of_files)
        self.length = file_end
        idx = 0
        while True:
            # If your batch > size of dir, reshuffle & restart
            if idx + self.bsize >= file_end:
                np.random.shuffle(list_of_files)
                idx = 0

            # Create batch of file names
            file_batch = list_of_files[idx: idx+self.bsize]
            x_batch = []
            y_batch = []
            for fname in file_batch:
                # Load & preprocess image
                img = self.preprocess_image(np.load(join(train_dir, fname))['features'])

                # Each tuple of the batch is (x, y)
                x_batch.append(img)
                y_batch.append(img)

            yield np.array(x_batch), np.array(y_batch)


class ProcessSampler(RandomSampler):
    """
        Samples randomly over our dataset, and performs augmentation on sample

        :param data_source: instance of PixDataset
        :param augments: list containing tuples of
        ('func_name', [args]), where [args] must be at least []
        and func_name must be a valid static function in the Augments class
        **order of augment functions matters**
    """

    def __init__(self, data_source, augments=(('resize', [(128, 128)]))):
        super(ProcessSampler, self).__init__(data_source)
        self.augments = augments

    def __iter__(self):
        super_iter = self(ProcessSampler, self).__iter__()
        for item in super_iter:
            yield self.preprocess_image(item)

    def preprocess_image(self, numpy_arr):
        """
        Takes in a 3-channel image, and returns grayscale image, according to a stack overflow parameter map
        :param numpy_arr: 3-channel np.array
        :return: 1-channel np.array

        Yo s/o to Ajay Raj for the sick Augments.___dict___ stuff
        """
        img = numpy_arr

        # Make sure img is 3 channel
        if len(img.shape) < 3:
            img = Augments.extend(img)

        for augment_fn, augment_params in self.augments:
            img = Augments.__dict__[augment_fn].__func__(img, *augment_params)
        return img


class Augments:
    
    @staticmethod
    def rgb2gray(img):
        """Returns a rank-2 tensor representing grayscale image"""
        return np.dot(img[...,:3], [0.299, 0.587, 0.114])

    @staticmethod
    def resize(img, new_size):
        """Resizes an image using cv2's resize function"""
        return cv2.resize(img, new_size)
                                  
    @staticmethod
    def extend(img):
        """Adds a dimension to the last axis of image passed in"""
        while len(img.shape) < 3:
            img = np.expand_dims(img, axis=-1)
        return img
    
    @staticmethod
    def add_noise(img):
        """Adds noise to an image using skimage"""
        return skimage.util.random_noise(img)


if __name__ == '__main__':
    # Run test
    from time import time
    datadir = os.path.join(PixDataset.DATAPATH, PixDataset.TRAIN)

    iter_num = 10
    bsize = 30
    test_gen = PixDataset(datadir, bsize=bsize, augments=[('rgb2gray', []), ('add_noise', [])])

    # Assuming we have 300 files... :smiley:
    # Test to make sure each batch is (30, imsize, imsize, 1)
    iterator = test_gen.__iter__()
    start_time = time()
    for i in range(iter_num):
        batch = iterator.__next__()
        batch = np.array(batch)
        assert np.equal(batch.shape, (2, 30, 128, 128, 1)).all(), \
            "Batch {} was incorrect, given {}, need {}".format(i,batch.shape, [2,30,128,128,1])
    end_time = time()
    print("The first {} examples generated had the correct shape".format(iter_num*bsize))
    print("This took: {:.3f} seconds".format(end_time-start_time))

