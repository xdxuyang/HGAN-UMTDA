from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy
import os
import sys
import util
from functools import reduce
from urllib.parse import urljoin
import gzip
import struct
import operator
import numpy as np

import matplotlib.pyplot as pyplot
import tarfile
import os
import pickle as pkl
import numpy as np
import skimage
import skimage.io
import skimage.transform
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


# from preprocessing import preprocessing
def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class MNISTM:
    base_url = 'http://yann.lecun.com/exdb/mnist/'

    data_files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz',
    }

    def __init__(self, path=None, shuffle=True, output_size=[28, 28], output_channel=1, split='train', select=[],alfa=1):
        self.image_shape = (28, 28, 3)
        self.label_shape = ()
        self.path = path
        self.shuffle = shuffle
        self.output_size = output_size
        self.output_channel = output_channel
        self.split = split
        self.select = select
        self.download()
        self.pointer = 0
        self.alfa = alfa
        self.load_dataset()


    def download(self):
        data_dir = self.path
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.path + '/' + filename
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

    def _read_datafile(self, path, expected_dims):
        base_magic_num = 2048
        with gzip.GzipFile(path) as f:
            magic_num = struct.unpack('>I', f.read(4))[0]
            expected_magic_num = base_magic_num + expected_dims
            if magic_num != expected_magic_num:
                raise ValueError(
                    'Incorrect MNIST magic number (expected ''{}, got {})'.format(expected_magic_num, magic_num))
            dims = struct.unpack('>' + 'I' * expected_dims, f.read(4 * expected_dims))
            buf = f.read(reduce(operator.mul, dims))
            data = np.frombuffer(buf, dtype=np.uint8)
            data = data.reshape(*dims)
            return data

    def shuffle_data(self):
        images = self.images[:]
        labels = self.labels[:]
        self.images = []
        self.labels = []

        idx = np.random.permutation(len(labels))
        for i in idx:
            self.images.append(images[i])
            self.labels.append(labels[i])

    def load_dataset(self):
        abspaths = {name: self.path + '/' + path
                    for name, path in self.data_files.items()}

        BST_PATH = 'BSR_bsds500.tgz'

        rand = np.random.RandomState(42)

        f = tarfile.open(BST_PATH)
        train_files = []
        for name in f.getnames():
            if name.startswith('BSR/BSDS500/data/images/train/'):
                train_files.append(name)

        print('Loading BSR training images')
        background_data = []
        for name in train_files:
            try:
                fp = f.extractfile(name)
                bg_img = skimage.io.imread(fp)
                background_data.append(bg_img)
            except:
                continue

        if self.split == 'train':
            images = self._read_images(abspaths['train_images'])
            self.images = create_mnistm(images,rand,background_data,self.alfa)
            self.labels = self._read_labels(abspaths['train_labels'])
        elif self.split == 'test':
            images = self._read_images(abspaths['test_images'])

            self.images = create_mnistm(images,rand,background_data,self.alfa)
            self.labels = self._read_labels(abspaths['test_labels'])
        if len(self.select) != 0:
            self.images = self.images[self.select]
            self.labels = self.labels[self.select]

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()

    def class_next_batch(self, num_per_class):
        batch_size = 10 * num_per_class
        classpaths = []
        ids = []
        for i in range(10):
            classpaths.append([])
        for j in range(len(self.labels)):
            label = self.labels[j]
            classpaths[label].append(j)
        for i in range(10):
            ids += np.random.choice(classpaths[i], size=num_per_class, replace=False).tolist()
        selfimages = np.array(self.images)
        selflabels = np.array(self.labels)
        return np.array(selfimages[ids]), get_one_hot(selflabels[ids], 10)

    def next_batch(self, batch_size):
        if self.pointer + batch_size >= len(self.labels):
            self.reset_pointer()
        images = self.images[self.pointer:(self.pointer + batch_size)]
        labels = self.labels[self.pointer:(self.pointer + batch_size)]
        self.pointer += batch_size
        return np.array(images), get_one_hot(labels, 10)

    def _read_images(self, path):
        return (self._read_datafile(path, 3)
                .astype(np.float32)
                .reshape(-1, 28, 28, 1)
                / 255.0)

    def _read_labels(self, path):
        return self._read_datafile(path, 1)


def main():

    mnist = MNISTM(path='data/mnist')
    a, b = mnist.next_batch(1)
    print(a)
    print(b)




def compose_image(digit, background,alfa):
    """Difference-blend a digit and a random patch from a background image."""
    w, h, _ = background.shape
    dw, dh, _ = digit.shape
    x = np.random.randint(0, w - dw)
    y = np.random.randint(0, h - dh)

    bg = alfa*background[x:x + dw, y:y + dh]
    return np.abs((bg - digit)/255).astype(np.float32)


def mnist_to_img(x):
    """Binarize MNIST digit and convert to RGB."""
    x = (x > 0).astype(np.float32)
    d = x.reshape([28, 28, 1]) * 255
    return np.concatenate([d, d, d], 2)


def create_mnistm(X,rand,background_data,alfa):
    """
    Give an array of MNIST digits, blend random background patches to
    build the MNIST-M dataset as described in
    http://jmlr.org/papers/volume17/15-239/15-239.pdf
    """
    X_ = np.zeros([X.shape[0], 28, 28, 3], np.float32)
    for i in range(X.shape[0]):

        if i % 1000 == 0:
            print('Processing example', i)

        bg_img = rand.choice(background_data)

        d = mnist_to_img(X[i])
        d = compose_image(d, bg_img,alfa)
        X_[i] = d

    return X_


if __name__ == '__main__':
    main()
