import tensorflow as tf
import numpy as np
import os
from sklearn.model_selection import train_test_split
from utils import utils
import matplotlib.pyplot as plt


class Data():
    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size
        self.data_path = data_path
        self.num_added_data = 0
        self.data_dim = None
        self.steps_per_execution = 0

    def load_data(self, data_name, normalizer, padder):
        if data_name == "mnist":
            return self.load_mnist(normalizer)
        elif data_name == "cloud":
            return self.load_cloud(normalizer, padder)
        elif data_name == "isabel":
            return self.load_isabel(normalizer, padder)
        else:
            raise NotImplementedError

    def load_mnist(self, normalizer):
        (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
        print(f"x_train.shape: {x_train.shape}")
        print(f"x_test.shape: {x_test.shape}")
        data = np.vstack([x_train, x_test])
        data = normalizer.normalize_minmax(data)
        data = data[..., np.newaxis]

        # Make data_len divisible by batch_size
        data = self.complete_data_with_batch(data)
        self.data_dim = data.shape

        data = self.to_tf_dataset(data)
        return data.repeat().prefetch(tf.data.AUTOTUNE)

    def load_process_cloud(self, normalizer, padder):
        data_dir = os.path.join(self.data_path, 'cesm_atm_data2')
        filenames = utils.get_filename(data_dir, ".f32")
        filename = [name for name in filenames if "CLOUD" in name][0]
        data = utils.load_data(filename)
        data = data.numpy()

        # reshape and flip dataset because it is flipped
        data = data.reshape(26, 1800, 3600)
        data = np.array([np.flipud(layer) for layer in data])
        x_train, x_test = train_test_split(data, train_size=0.9, random_state=2, shuffle=True)
        
        # Normalizing
        x_train = normalizer.normalize_minmax(x_train)
        x_train = normalizer.normalize_log10(x_train)
        x_test = normalizer.normalize_minmax(x_test)
        x_test = normalizer.normalize_log10(x_test)

        # Padding to match tile size
        x_train = padder.pad_image_to_tile_multiple(x_train)
        x_test = padder.pad_image_to_tile_multiple(x_test)

        if (x_train.ndim == 2):
            x_train, x_test = x_train[np.newaxis, ...], x_test[np.newaxis, ...]
        if (x_train.ndim == 3):
            x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]

        # Stack new separated partitions into one united set
        x_train = np.vstack([utils.split_image(x_train[i], padder.tile_size) for i in range(x_train.shape[0])])
        x_test = np.vstack([utils.split_image(x_test[i], padder.tile_size) for i in range(x_test.shape[0])])

        x_train = tf.data.Dataset.from_tensor_slices(x_train).shuffle(buffer_size=x_train.shape[0]).batch(self.batch_size)
        x_test = tf.data.Dataset.from_tensor_slices(x_test).batch(self.batch_size)

        return x_train, x_test

    def load_cloud(self, normalizer, padder):
        data_dir = os.path.join(self.data_path, 'cesm_atm_data2')
        filenames = utils.get_filename(data_dir, ".f32")
        filename = [name for name in filenames if "CLOUD" in name][0]
        data = utils.load_data(filename)
        data = data.numpy()

        # reshape and flip dataset because it is flipped
        data = data.reshape(26, 1800, 3600)
        data = np.array([np.flipud(layer) for layer in data])
        
        # Normalizing
        data = normalizer.normalize_minmax(data)
        data = normalizer.normalize_log10(data)

        # Padding to match tile size
        data = padder.pad_image_to_tile_multiple(data)
        if (data.ndim == 2):
            data= data[np.newaxis, ...]
        if (data.ndim == 3):
            data = data[..., np.newaxis]
        
        # img_num = 20
        # print(f"x_test[{img_num}].shape: {data[img_num].shape}")
        # image = utils.split_image(data[img_num], padder.tile_size)
        # fig1 = plt.figure(figsize=(12, 10))
        # plt.imshow(data[img_num], cmap="gray")
        # plt.yticks(np.arange(0, 1800, 600))
        # plt.xticks(np.arange(0, 3600, 600))
        # fig2 = plt.figure(figsize=(12, 10))
        # for i in range(image.shape[0]):
        #     plt.subplot(4, 6, i + 1)
        #     plt.imshow(image[i], cmap="gray")
        #     plt.axis('off')

        # image = utils.unsplit_image(image, (1800, 3600, 1))
        # fig3 = plt.figure(figsize=(12, 10))
        # plt.imshow(image, cmap="gray")
        # plt.yticks(np.arange(0, 1800, 600))
        # plt.xticks(np.arange(0, 3600, 600))
        # plt.show()

        # Stack new separated partitions into one united set
        data = np.vstack([utils.split_image(data[i], padder.tile_size) for i in range(data.shape[0])])

        # Add more tiles to complete batch size
        print(f"before append data.shape: {data.shape}")
        len_last_batch = data.shape[0] - data.shape[0] // self.batch_size * self.batch_size
        
        num_added_data = self.batch_size - len_last_batch
        self.num_added_data = num_added_data
        number_of_rows = data.shape[0]
        random_indices = np.random.choice(number_of_rows, size=num_added_data, replace=False)
        additional_imgs = data[random_indices]
        data = np.append(data, additional_imgs, axis=0)
        
        print(f"after append data.shape: {data.shape}")
        print(f"number of batches: {data.shape[0] / self.batch_size}")

        data = self.to_tf_dataset(data)

        return data

    def load_isabel(self, normalizer):
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