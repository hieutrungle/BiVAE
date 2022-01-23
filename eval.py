import matplotlib.pyplot as plt
from plot_metrics import plot_metrics
import timeit
import numpy as np
import os
import re
import sys
from skimage import metrics
from generate import generate
import json
import tensorflow as tf
import pickle
from tqdm import tqdm
from utils import utils

def evaluate(model, data, model_path, save_encoding=False, padding=None):

    # Get names of weight_paths
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    weight_paths = sorted([f for f in os.listdir(checkpoint_dir) \
                        if re.search(r'model_[0-9]+.index', f)])
    weight_paths = [f[:-6] for f in weight_paths]
    
    # make eval directory
    eval_dir = os.path.join(model_path, 'eval')
    utils.mkdir_if_not_exist(eval_dir)

    # extract iterations
    iters = [int(re.search(r'[0-9]+', f)[0]) for f in weight_paths]

    # complete names of weight_paths
    weight_paths = [os.path.join(checkpoint_dir, f) for f in weight_paths]
    metrics = {}
    unbatch_data = data.unbatch()
    unbatch_data = np.stack(list(unbatch_data))
    
    with tqdm(iters) as pbar:
        for curr_iter, weight_path in zip(iters, weight_paths):
            
            # load model
            model.load_weights(weight_path)

            # gen_sample = [sample for sample in data.take(1)][0]
            # generate(model, gen_sample, eval_dir)

            decoded_data, z_samples, ftrs = utils.predict(model, data)

            if save_encoding:
                # export compressed data to pickle file
                z_sample_file = f'z_samples_{curr_iter:06d}.pkl'
                with open(os.path.join(eval_dir, z_sample_file),'wb') as f:
                    pickle.dump(z_samples, f)
                ftr_file = f'ftrs_{curr_iter:06d}.pkl'
                with open(os.path.join(eval_dir, ftr_file),'wb') as f:
                    pickle.dump(ftrs, f)

                # Load encoded data
                with open(os.path.join(eval_dir, z_sample_file),'rb') as f:
                    z_samples = pickle.load(f)

            # Process decoded data
            if padding is not None:
                decoded_data = decoded_data.reshape(np.prod(decoded_data.shape))
                decoded_data = decoded_data[:-padding]
            decoded_data = decoded_data.reshape(unbatch_data.shape)
            
            # record metrics
            psnr, ssim, mse = get_metrics(unbatch_data, decoded_data)
            metrics.update({curr_iter: {
                            'weight_path': weight_path, 
                            'mse': mse, 'psnr': psnr, 'ssim': ssim}
                            })
            pbar.update(1)
            pbar.set_description(f"Iter: {curr_iter}, MSE: {mse:.03f}, " +
                                f"PSNR: {psnr:.03f}, SSIM: {ssim:.03f}")

        # save metrics
        metric_fname = os.path.join(eval_dir, f'metrics.txt')
        with open(metric_fname, "w") as f:
            f.write(json.dumps(metrics)+"\n")

        plot_metrics(metric_fname)

def get_metrics(image_true, image_test):

    ssim, psnr, mse = 0, 0, 0
    for i in range(image_true.shape[0]):
        ssim += metrics.structural_similarity(image_true[i], image_test[i], 
                                        multichannel=True, data_range=1)
        psnr += metrics.peak_signal_noise_ratio(image_true[i], image_test[i], 
                                        data_range=1)
        mse += metrics.mean_squared_error(image_true[i], image_test[i])
    ssim /= image_true.shape[0]
    psnr /= image_true.shape[0]
    mse /= image_true.shape[0]
    return psnr, ssim, mse

def get_compression_ratio(model_path):
    compress_filename = os.path.join(model_path, "compress_cloud.f32")

    compress_size = int(os.path.getsize(compress_filename))
    print(f"compress file size: {compress_size} bytes\n")
    model_size = utils.get_folder_size(model_path) - compress_size
    print(f"overhead storage: {model_size} bytes\n")

    input_size = int(os.path.getsize(compress_filename))
    print(f"input size: {input_size} bytes\n")
    print(f"CR without overhead: {input_size/compress_size}")
    print(f"CR with overhead: {input_size/(compress_size + model_size)}")

    return input_size / (compress_size + model_size)

def plot_images(images):
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis('off')