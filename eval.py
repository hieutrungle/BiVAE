import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re
import sys
from skimage import metrics
import reconstruct
import json
import tensorflow as tf
from tqdm import tqdm
import seaborn as sns
from utils import utils
sns.set_theme()

def evaluate(model, iterator, dataio, model_path, save_encoding=False):

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
    
    steps_per_execution = dataio.data_dim[0] // dataio.batch_size

    for curr_iter, weight_path in zip(iters, weight_paths):
        
        # with tqdm(total=steps_per_execution) as pbar:
        # load model
        model.load_weights(weight_path)

        (orig_imgs, recon_imgs) = reconstruct.reconstruct_img(model, iterator, dataio, is_plotting=False)

        # record metrics
        psnr, ssim, mse = get_metrics(orig_imgs, recon_imgs)
        metrics.update({curr_iter: {
                        'weight_path': weight_path, 
                        'mse': mse, 'psnr': psnr, 'ssim': ssim}
                        })
        # pbar.update(steps_per_execution)
        tqdm.write(f"Iter: {curr_iter}, MSE: {mse:.06f}, " +
                    f"PSNR: {psnr:.03f}, SSIM: {ssim:.05f}\n"
                    )

    # save metrics
    metric_fname = os.path.join(eval_dir, f'metrics.txt')
    with open(metric_fname, "w") as f:
        f.write(json.dumps(metrics)+"\n")

    plot_metrics(metric_fname)

def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    return tf.image.decode_image(string, channels=3)


def write_png(filename, image):
    """Saves an image to a PNG file."""
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)

def compress(args):
    """Compresses an image."""
    # Load model and use it to compress the image.
    model = tf.keras.models.load_model(args.model_path)
    x = read_png(args.input_file)
    tensors = model.compress(x)

    # Write a binary file with the shape information and the compressed string.
    packed = utils.PackedTensors()
    packed.pack(tensors)
    with open(args.output_file, "wb") as f:
        f.write(packed.string)

    # If requested, decompress the image and measure performance.
    if args.verbose:
        x_hat = model.decompress(*tensors)

        # Cast to float in order to compute metrics.
        x = tf.cast(x, tf.float32)
        x_hat = tf.cast(x_hat, tf.float32)
        mse = tf.reduce_mean(tf.math.squared_difference(x, x_hat))
        psnr = tf.squeeze(tf.image.psnr(x, x_hat, 255))
        msssim = tf.squeeze(tf.image.ssim_multiscale(x, x_hat, 255))
        msssim_db = -10. * tf.math.log(1 - msssim) / tf.math.log(10.)

        # The actual bits per pixel including entropy coding overhead.
        num_pixels = tf.reduce_prod(tf.shape(x)[:-1])
        bpp = len(packed.string) * 8 / num_pixels

        print(f"Mean squared error: {mse:0.4f}")
        print(f"PSNR (dB): {psnr:0.2f}")
        print(f"Multiscale SSIM: {msssim:0.4f}")
        print(f"Multiscale SSIM (dB): {msssim_db:0.2f}")
        print(f"Bits per pixel: {bpp:0.4f}")


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

def plot_metrics(filename):
    
    with open(filename, 'r') as f:
        data = f.read()
    data = json.loads(data)
    df = pd.DataFrame.from_dict(data).transpose()

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Metrics')

    sns.lineplot(x=df.index.astype(int),y="psnr", data=df, ax=axes[0,0])
    axes[0,0].legend(labels=["PSNR"])
    axes[0,0].set_ylabel("psnr")
    axes[0,0].set_xlabel("epochs")
    
    sns.lineplot(x=df.index.astype(int), y="ssim", data=df, ax=axes[0,1])
    axes[0,1].legend(labels=["SSIM"])
    axes[0,1].set_ylabel("ssim")
    axes[0,1].set_xlabel("epochs")

    sns.lineplot(x=df.index.astype(int), y="mse", data=df, ax=axes[1,0])
    axes[1,0].legend(labels=["MSE"])
    axes[1,0].set_ylabel("mse")
    axes[1,0].set_xlabel("epochs")

    plt.savefig(os.path.join(os.path.dirname(filename), "metrics"))

    plt.show()

if __name__=='__main__':
    eval_dir = "model_output"
    metric_fname = os.path.join(eval_dir, f'metrics.txt')

    plot_metrics(metric_fname)