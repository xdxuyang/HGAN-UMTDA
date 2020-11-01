import numpy
import os
import sys
import util
from urllib.parse import urljoin
import gzip
import struct
import operator
import numpy as np
from scipy.io import loadmat


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class USPS:
    base_url = 'http://statweb.stanford.edu/~tibs/ElemStatLearn/datasets/'

    data_files = {
        'train': 'zip.train.gz',
        'test': 'zip.test.gz',
        # 'extra': 'extra_32x32.mat',
    }

    def __init__(self, path=None, select=[], shuffle=True, output_size=[28, 28], output_channel=1, split='train'):
        self.image_shape = (16, 16, 1)
        self.label_shape = ()
        self.path = path
        self.shuffle = shuffle
        self.output_size = output_size
        self.output_channel = output_channel
        self.split = split
        self.select = select
        self.download()
        self.pointer = 0
        self.load_dataset()
        self.classpaths = []
        self.class_pointer = 10 * [0]
        for i in range(10):
            self.classpaths.append([])
        for j in range(len(self.labels)):
            label = self.labels[j]
            self.classpaths[label].append(j)

    def download(self):
        data_dir = self.path
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        for filename in self.data_files.values():
            path = self.path + '/' + filename
            if not os.path.exists(path):
                url = urljoin(self.base_url, filename)
                util.maybe_download(url, path)

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

        if self.split == 'train':
            train_images, train_labels = self._read_datafile(abspaths['train'])
            self.images = train_images
            self.labels = train_labels
        elif self.split == 'test':
            test_images, test_labels = self._read_datafile(abspaths['test'])
            self.images = test_images
            self.labels = test_labels
        if len(self.select) != 0:
            self.images = self.images[self.select]
            self.labels = self.labels[self.select]

    def reset_pointer(self):
        self.pointer = 0
        if self.shuffle:
            self.shuffle_data()

    def reset_class_pointer(self, i):
        self.class_pointer[i] = 0
        if self.shuffle:
            self.classpaths[i] = np.random.permutation(self.classpaths[i])

    def class_next_batch(self, num_per_class):
        batch_size = 10 * num_per_class
        selfimages = np.zeros((0, 16, 16, 1))
        selflabels = []
        for i in range(10):
            selfimages = np.concatenate((selfimages, self.images[
                self.classpaths[i][self.class_pointer[i]:self.class_pointer[i] + num_per_class]]), 0)
            selflabels += self.labels[self.classpaths[i][self.class_pointer[i]:self.class_pointer[i] + num_per_class]]
            self.class_pointer[i] += num_per_class
            if self.class_pointer[i] + num_per_class >= len(self.classpaths[i]):
                self.reset_class_pointer(i)
        return np.array(selfimages), get_one_hot(selflabels, 10)

    def next_batch(self, batch_size):
        images = self.images[self.pointer:(self.pointer + batch_size)]
        labels = self.labels[self.pointer:(self.pointer + batch_size)]
        self.pointer += batch_size
        if self.pointer + batch_size >= len(self.labels):
            self.reset_pointer()
        return np.array(images), get_one_hot(labels, 10)
    def _read_datafile(self, path):
        """Read the proprietary USPS digits data file."""
        labels, images = [], []
        with gzip.GzipFile(path) as f:
            for line in f:
                vals = line.strip().split()
                labels.append(float(vals[0]))
                images.append([float(val) for val in vals[1:]])
        labels = np.array(labels, dtype=np.int)
        images = np.array(images, dtype=np.float32).reshape(-1, 16, 16, 1)
        images = (images + 1) / 2
        return images, labels


def main():
    usps = USPS(path='data/svhn')
    a, b = usps.class_next_batch(1)
    print(a)
    print(b)


if __name__ == '__main__':
    main()
