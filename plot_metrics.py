import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme()

def plot_metrics(filename):
    
    with open(filename, 'r') as f:
        data = f.read()
    data = json.loads(data)
    df = pd.DataFrame.from_dict(data).transpose()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle('Metrics')

    sns.lineplot(x=df.index.astype(int),y="psnr", data=df, ax=axes[0])
    axes[0].legend(labels=["psnr"])
    axes[0].set_ylabel("magnitude")
    axes[0].set_xlabel("epochs")
    
    sns.lineplot(x=df.index.astype(int), y="mse", data=df, ax=axes[1])
    sns.lineplot(x=df.index.astype(int), y="ssim", data=df, ax=axes[1])
    axes[1].legend(labels=["mse","ssim"])
    axes[1].set_ylabel("magnitude")
    axes[1].set_xlabel("epochs")
    plt.savefig(os.path.join(os.path.dirname(filename), "ssim"))
    plt.show()

if __name__ == '__main__':
    folder_name = "model_vae_relu_3cnn_0"
    model_dir = os.path.join("model_output", folder_name)
    eval_dir = os.path.join(model_dir, 'eval')
    fname = "metrics.txt"
    fname = os.path.join(eval_dir, fname)
    plot_metrics(fname)