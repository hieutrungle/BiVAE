import matplotlib.pyplot as plt
import os
import numpy as np
from utils import utils


def reconstruct_img(model, data, normalizer=None, padder=None, img_folder="./", prefix_name=None):

    recon_img, z_samples, ftrs = utils.predict(model, data)

    padded_img_size = list(padder.padded_img_shape) + [1]
    num_tile_img = np.prod(padded_img_size) // (np.square(padder.tile_size))
    num_img = recon_img.shape[0] // num_tile_img
    # print(f"padded_img_size: {padded_img_size}")
    # print(f"recon_img.shape {recon_img.shape}")
    # print(f"num_tile_img: {num_tile_img}")
    # print(f"num_img: {num_img}")

    orig_img = data.unbatch()
    orig_img = np.stack(list(orig_img))
    orig_img = np.array([utils.unsplit_image(orig_img[i*num_tile_img:(i+1)*num_tile_img], padded_img_size) for i in range(num_img)])
    recon_img = np.array([utils.unsplit_image(recon_img[i*num_tile_img:(i+1)*num_tile_img], padded_img_size) for i in range(num_img)])

    # Denormalizer 
    orig_img = normalizer.denormalize_log10(orig_img)
    orig_img = normalizer.denormalize_minmax(orig_img)
    # utils.get_data_info(orig_img)

    recon_img = normalizer.denormalize_log10(recon_img)
    recon_img = normalizer.denormalize_minmax(recon_img)
    # utils.get_data_info(recon_img)

    print(f"recon_img.shape: {recon_img.shape}")
    
    generate_plots(orig_img, recon_img, img_folder, num_img=3, prefix_name=None)

def generate_plots(orig_img, recon_img, img_folder, num_img=3, prefix_name=None):
    fig1 = plt.figure(figsize=(15, 13))
    cmap = 'viridis'
    for i in range(min(num_img,3)):
        plt.subplot(min(num_img,3), 1, i + 1)
        plt.imshow(orig_img[-i], cmap=cmap)
        plt.axis('off')
    plt.savefig(os.path.join(img_folder, prefix_name+"original_images.png"))

    fig2 = plt.figure(figsize=(15, 13))
    for i in range(min(num_img,3)):
        plt.subplot(min(num_img,3), 1, i + 1)
        plt.imshow(recon_img[-i], cmap=cmap)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(os.path.join(img_folder, prefix_name+"generate_image.png"))
    plt.show()