import matplotlib.pyplot as plt
import os


def generate(model, sample, img_folder, img_name='gen_image.png'):
    recon_imgs, _ = model.predict(sample)

    fig1 = plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(sample[i], cmap="gray")
        plt.axis('off')

    fig2 = plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(recon_imgs[i], cmap="gray")
        plt.axis('off')

    plt.savefig(os.path.join(img_folder, img_name))
    plt.show()

def plot_images(images):
    fig1 = plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap="gray")
        plt.axis('off')