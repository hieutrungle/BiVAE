import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import utils
import matplotlib.pyplot as plt
import errno
from utils.normalizer import DataNormalizer
from utils.padder import Padder


class Data():
    def __init__(self, data_path, batch_size, tile_size):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_added_data = 0
        self.data_dim = None
        self.steps_per_execution = 0
        self.padder = Padder(tile_size)
        self.normalizer = None

    def load_data(self, data_name):
        if data_name == "mnist":
            data = self.load_mnist()
        elif data_name == "cesm":
            data = self.load_cesm()
        elif data_name == "isabel":
            data = self.load_isabel()
        else:
            raise FileNotFoundError(
                    errno.ENOENT, os.strerror(errno.ENOENT), data_name)

        # Make data_len divisible by batch_size

        data = self.complete_data_with_batch(data)
        self.data_dim = data.shape

        data = self.to_tf_dataset(data)
        return data.repeat().prefetch(tf.data.AUTOTUNE)

    def load_mnist(self):

        self.normalizer = DataNormalizer(min_scale=0.0, max_scale=1.0)

        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        print(f"x_train.shape: {x_train.shape}")
        print(f"x_test.shape: {x_test.shape}")
        data = np.vstack([x_train, x_test])
        data = self.normalizer.normalize_minmax(data)
        data = data[..., np.newaxis]

        return data

    def load_cesm(self):

        self.normalizer = DataNormalizer(min_scale=1.0, max_scale=10.0)

        data_dir = os.path.join(self.data_path, 'cesm_atm_data2')
        filenames = utils.get_filename(data_dir, ".f32")
        filename = [name for name in filenames if "CLOUD" in name][0]
        data = utils.load_data(filename)
        data = data.numpy()

        # reshape and flip dataset because it is flipped
        data = data.reshape(26, 1800, 3600)
        data = np.array([np.flipud(layer) for layer in data])
        
        # Normalizing
        data = self.normalizer.normalize_minmax(data)
        data = self.normalizer.normalize_log10(data)

        # Padding to match tile size
        data = self.padder.pad_image_to_tile_multiple(data)
        if (data.ndim == 2):
            data= data[np.newaxis, ...]
        if (data.ndim == 3):
            data = data[..., np.newaxis]
        
        # Plot data tiles and original image to check spliting method
        # self.plot_image(data)

        # Stack new separated partitions into one united set
        data = np.vstack([self.padder.split_image(data[i], self.padder.tile_size) 
                            for i in range(data.shape[0])])

        utils.get_data_info(data)

        return data

    def load_isabel(self):
        pass

    def complete_data_with_batch(self, data):

        len_last_batch = data.shape[0] - data.shape[0] // self.batch_size * self.batch_size
        num_added_data = self.batch_size - len_last_batch
        self.num_added_data = num_added_data
        number_of_rows = data.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_added_data, replace=False)
        additional_imgs = data[random_indices]
        data = np.append(data, additional_imgs, axis=0)
        return data

    def to_tf_dataset(self, data, dtype=tf.float32):
        data = tf.convert_to_tensor(data, dtype=dtype)
        data = tf.data.Dataset.from_tensor_slices(data)
        data = data.batch(self.batch_size)
        return data

    def plot_image(self, data):
        img_num = 21
        print(f"data[{img_num}].shape: {data[img_num].shape}")
        image = self.padder.split_image(data[img_num], self.padder.tile_size)

        fig1 = plt.figure(figsize=(12, 10))
        plt.imshow(data[img_num], cmap="gray")
        plt.yticks(np.arange(0, 1800, 600))
        plt.xticks(np.arange(0, 3600, 600))
        plt.title(f"Original image {img_num}")

        fig2 = plt.figure(figsize=(12, 10))
        for i in range(image.shape[0]):
            plt.subplot(4, 6, i + 1)
            plt.imshow(image[i], cmap="gray")
            plt.axis('off')
        plt.title(f"split image {img_num}")

        image = self.padder.unsplit_image(image, (1800, 3600, 1))
        fig3 = plt.figure(figsize=(12, 10))
        plt.imshow(image, cmap="gray")
        plt.yticks(np.arange(0, 1800, 600))
        plt.xticks(np.arange(0, 3600, 600))
        plt.title(f"unsplit image {img_num}")
        plt.show()