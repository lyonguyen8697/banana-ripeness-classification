import os
import math
import numpy as np
import pandas as pd
import cv2
import imgaug.augmenters as iaa


class DataSet(object):
    def __init__(self,
                 image_dir,
                 labeled,
                 batch_size,
                 image_size,
                 shuffle=True,
                 augmented=False):
        self.image_dir = image_dir
        self.labeled = labeled
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augmented = augmented
        self.setup()

    def setup(self):

        self.image_files = []
        self.labels = []
        if self.labeled:
            labels = os.listdir(self.image_dir)
            for label in labels:
                image_files = os.listdir('{}/{}'.format(self.image_dir, label))
                image_files = ['{}/{}/{}'.format(self.image_dir, label, f) for f in image_files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]
                self.image_files.extend(image_files)
                self.labels.extend([int(label) - 1] * len(image_files))
        else:
            self.image_files = os.listdir(self.image_dir)
            self.image_files = ['{}/{}'.format(self.image_dir, f) for f in self.image_files if f.lower().endswith('.jpg') or f.lower().endswith('.png')]

        self.image_files = np.array(self.image_files)
        self.labels = np.array(self.labels)
        self.count = len(self.image_files)
        self.num_batches = math.ceil(self.count / self.batch_size)
        self.idxs = list(range(self.count))
        if self.augmented and self.labeled:
            self.build_augmentor()
        self.reset()

    def build_augmentor(self):
        self.augmentor = iaa.Sometimes(0.5, iaa.GammaContrast(gamma=(0.3, 1.5)))

    def reset(self):
        self.current_idx = 0
        if self.shuffle:
            np.random.shuffle(self.idxs)

    def next_batch(self):
        assert self.has_next_batch()

        start = self.current_idx
        end = self.current_idx + self.batch_size
        if end > self.count:
            end = self.count
        self.current_idx = end

        current_idxs = self.idxs[start:end]

        self.current_image_files = self.image_files[current_idxs]

        images = self.load_images(self.current_image_files)

        labels = self.labels[current_idxs]

        if self.labeled:
            return images, labels
        else:
            return images

    def has_next_batch(self):
        return self.current_idx < self.count

    def load_images(self, image_files):
        images = []
        for image_file in image_files:
            image = self.load_image(image_file)
            images.append(image)

        if self.augmented and self.labeled:
            self.augmentor.augment_images(images)

        images = np.array(images) / 255.0

        return images

    def load_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        image = cv2.resize(image, tuple(self.image_size[:2]))

        return image
