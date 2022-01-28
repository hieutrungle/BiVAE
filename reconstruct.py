import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import tensorflow as tf
from utils import utils


def reconstruct_img(model, iterator, dataio, img_folder="./", prefix_name=""):

    utils.mkdir_if_not_exist(img_folder)

    steps_per_execution = dataio.data_dim[0] // dataio.batch_size
    orig_imgs = np.array([]).reshape([0]+list(dataio.data_dim[1:]))
    recon_imgs = np.array([]).reshape([0]+list(dataio.data_dim[1:]))
    # print(f"orig_imgs.shape: {orig_imgs.shape}")
    # print(f"recon_imgs.shape: {recon_imgs.shape}")
    for i in range(steps_per_execution):
        cur_data = next(iterator)
        recon_img, _ = model.predict(cur_data)
        # print(f"cur_data.shape: {cur_data.shape}")
        # print(f"recon_img.shape: {np.array(recon_img).shape}")
        orig_imgs = np.vstack([orig_imgs, cur_data])
        recon_imgs = np.vstack([recon_imgs, recon_img])

    # print(f"orig_imgs.shape: {orig_imgs.shape}")
    # print(f"recon_imgs.shape: {recon_imgs.shape}")

    orig_imgs = orig_imgs[:-dataio.num_added_data]
    recon_imgs = recon_imgs[:-dataio.num_added_data]

    normalizer = dataio.normalizer
    padder = dataio.padder

    # sys.exit()

    padded_img_shape = list(padder.padded_img_shape) + [1]
    num_tile_img = np.prod(padded_img_shape) // (np.square(padder.tile_size))
    num_img = recon_imgs.shape[0] // num_tile_img
    # print(f"padder.padded_img_shape: {padder.padded_img_shape}")
    # print(f"padded_img_shape: {padded_img_shape}")
    # print(f"recon_imgs.shape {recon_imgs.shape}")
    # print(f"num_tile_img: {num_tile_img}")
    # print(f"num_img: {num_img}\n")

    # orig_img = data.unbatch()
    # orig_img = np.stack(list(orig_img))
    # for i in range(num_img):
    #     orig_img = utils.unsplit_image(orig_imgs[i*num_tile_img:(i+1)*num_tile_img], padded_img_shape)
    #     recon_img = utils.unsplit_image(recon_imgs[i*num_tile_img:(i+1)*num_tile_img], padded_img_shape)
    orig_imgs = np.array([padder.unsplit_image(orig_imgs[i*num_tile_img:(i+1)*num_tile_img], padded_img_shape) 
                        for i in range(num_img)])
    recon_imgs = np.array([padder.unsplit_image(recon_imgs[i*num_tile_img:(i+1)*num_tile_img], padded_img_shape) 
                        for i in range(num_img)])
    # print(f"After remove unspliting")
    # print(f"orig_imgs")
    # utils.get_data_info(orig_imgs)
    # print(f"recon_imgs")
    # utils.get_data_info(recon_imgs)

    padder.print_instance_attributes()

    orig_imgs = padder.remove_pad_ds(orig_imgs)
    recon_imgs = padder.remove_pad_ds(recon_imgs)
    # print(f"After remove padding")
    # print(f"orig_imgs")
    # utils.get_data_info(orig_imgs)
    # print(f"recon_imgs")
    # utils.get_data_info(recon_imgs)

    # Denormalizer
    # print(f"After remove denormalization")
    orig_imgs = normalizer.denormalize_log10(orig_imgs)
    orig_imgs = normalizer.denormalize_minmax(orig_imgs)
    # print(f"orig_imgs")
    # utils.get_data_info(orig_imgs)

    recon_imgs = normalizer.denormalize_log10(recon_imgs)
    recon_imgs = normalizer.denormalize_minmax(recon_imgs)
    # print(f"recon_imgs")
    # utils.get_data_info(recon_imgs)

    generate_plots(orig_imgs, recon_imgs, img_folder, num_img=3, prefix_name=prefix_name)

def generate_plots(orig_imgs, recon_imgs, img_folder, num_img, prefix_name=""):
    fig1 = plt.figure(figsize=(15, 13))
    cmap = 'viridis'
    num_img = num_img
    for i in range(num_img):
        plt.subplot(num_img, 1, i + 1)
        plt.imshow(orig_imgs[i], cmap=cmap)
        plt.axis('off')
    plt.savefig(os.path.join(img_folder, prefix_name+"original_images.png"))

    fig2 = plt.figure(figsize=(15, 13))
    for i in range(num_img):
        plt.subplot(num_img, 1, i + 1)
        plt.imshow(recon_imgs[i], cmap=cmap)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(img_folder, prefix_name+"generate_image.png"))
    plt.show()