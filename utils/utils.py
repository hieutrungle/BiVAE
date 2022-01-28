import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from typing import *

"""Data Preprocessing Helper Functions"""

def get_data_info(data:np.ndarray):
    print(f"data shape: {data.shape}")
    print(f"maximum value: {data.max()}")
    print(f"minimum value: {data.min()}\n")
    
def draw_hist(data:np.ndarray, title:str=""):
    plt.hist(data, bins=40)
    plt.title(title +"\n")
    plt.ylabel('Count')
    plt.xlabel('Value')
    plt.show()

# Get all file names in directory
def get_filename(parent_dir:str, file_extension:str)->str:
    filenames = []
    for root, dirs, files in os.walk(parent_dir):
        for filename in files:
            if (file_extension in filename):
                filenames.append(os.path.join(parent_dir, filename))
    return filenames

# Convert the input data into a vector of numbers;
def load_data(filename:str)->tf.Tensor:
    with open(filename, "rb") as f:
        data = f.read()
        print(f"Name of the training dataset: {filename}")
        print(f"Length of file: {len(data)} bytes\n")

        if filename[-4:] == ".txt":
            data = data.decode().split('\n')
        # If the input file is a standard file, 
        # there is a chance that the last line could simply be an empty line;
        # if this is the case, then remove the empty line
        if (data[len(data) - 1] == ""):
            data.remove("")
            data = np.array(data)
            data = data.astype('float32')
        else:
        # The input file is a binary file
            data = tf.io.decode_raw(data, tf.float32)
    return data

# Create new directory if not exist
def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Create folders to store training information
def mkdir_storage(model_dir, resume={}):
    if os.path.exists(os.path.join(model_dir, 'summaries')):
        if len(resume) == 0:
        # val = input("The model directory %s exists. Overwrite? (y/n) " % model_dir)
        # print()
        # if val == 'y':
            if os.path.exists(os.path.join(model_dir, 'summaries')):
                shutil.rmtree(os.path.join(model_dir, 'summaries'))
            if os.path.exists(os.path.join(model_dir, 'checkpoints')):
                shutil.rmtree(os.path.join(model_dir, 'checkpoints'))
    
    os.makedirs(model_dir, exist_ok=True)

    summaries_dir = os.path.join(model_dir, 'summaries')
    mkdir_if_not_exist(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    mkdir_if_not_exist(checkpoints_dir)
    return summaries_dir, checkpoints_dir

# Use tensorflow to split data into train + val + test, compatible with tf.data API
def split_train_val_test_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, 
                        shuffle=True, shuffle_size=10000, seed=0):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=seed)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

# Use tensorflow to split data into train + val, compatible with tf.data API
def split_train_val_tf(ds, ds_size, train_size=0.9, shuffle=True, shuffle_size=10000, seed=0):
    assert train_size <= 1, 'Split proportion must be in [0, 1]'
    assert train_size >= 0, 'Split proportion must be in [0, 1]'
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=seed)
    
    train_size = int(train_size * ds_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size)
    
    return train_ds, val_ds

# Use pandas to split data into train + val + test, compatible with pandas DataFrame
def split_train_val_test_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, random_state=0):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=random_state)
    indices_or_sections = [int(train_split * len(df)), int((1 - val_split - test_split) * len(df))]
    
    train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds, test_ds

# Use pandas to split data into train + val, compatible with pandas DataFrame
def split_train_val_pd(df, train_size=0.9, random_state=0):
    assert train_size <= 1, 'Split proportion must be in [0, 1]'
    assert train_size >= 0, 'Split proportion must be in [0, 1]'

    # Specify seed to always have the same split distribution between runs
    df_sample = df.sample(frac=1, random_state=random_state)
    indices_or_sections = [int(train_size * len(df)), len(df)]
    
    train_ds, val_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds, val_ds

def split_train_test_np(df, train_size=0.9, random_state=0):
    assert train_size <= 1, 'Split proportion must be in [0, 1]'
    assert train_size >= 0, 'Split proportion must be in [0, 1]'

    split = np.random.choice(range(df.shape[0]), int(train_size*df.shape[0]))
    train_ds = df[split]
    test_ds =  df[~split]
    
    print(f"train_ds.shape : {train_ds.shape}")
    print(f"test_ds.shape : {test_ds.shape}")
    return train_ds, test_ds

# Find number of row and column to slice the original image into many smaller
# block of given block_size
def find_dim_sblock(data:np.ndarray, block_size:int, verbose:bool=True):
    # Calculate number of blocks in a row (j) and in a column (i)
    num_block_row = int(data.shape[0] / block_size)
    num_block_col = int(data.shape[1] / block_size)

    # Add 1 more batch if mod( data.shape[0], block_size) != 0
    num_block_row = num_block_row if data.shape[0] % block_size == 0 else num_block_row+1
    num_block_col = num_block_col if data.shape[1] % block_size == 0 else num_block_col+1

    if verbose==True:
        print("Number of rows in the data that can be split into smaller blocks:", num_block_row)
        print("Number of columns in the data that can be split into smaller blocks:", num_block_col)

    return num_block_row, num_block_col

def get_folder_size(start_path:str='.')->int:
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    return total_size

def groups_per_scale(num_scales, num_groups_per_scale, is_adaptive, divider=2, minimum_groups=1):
    g = []
    n = num_groups_per_scale
    for s in range(num_scales):
        assert n >= 1
        g.append(n)
        if is_adaptive:
            n = n // divider
            n = max(minimum_groups, n)
    return g

def get_model_arch(arch_type):
    if arch_type == 'res_bnswish':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_dec'] = ['res_bnswish', 'res_bnswish']
        model_arch['up_sampling_dec'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_post'] = ['res_bnswish', 'res_bnswish']
        model_arch['up_sampling_post'] = ['res_bnswish', 'res_bnswish']
        model_arch['ar_nn'] = ['res_bnswish']
    elif arch_type == 'res_mbconv':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_enc'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_dec'] = ['mconv_e6k5g0']
        model_arch['up_sampling_dec'] = ['mconv_e6k5g0']
        model_arch['normal_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['down_sampling_pre'] = ['res_bnswish', 'res_bnswish']
        model_arch['normal_post'] = ['mconv_e3k5g0']
        model_arch['up_sampling_post'] = ['mconv_e3k5g0']
        model_arch['ar_nn'] = ['mconv_e6k5g0']
    elif arch_type == 'res_wnelu':
        model_arch = dict()
        model_arch['normal_enc'] = ['res_wnelu', 'res_elu']
        model_arch['down_sampling_enc'] = ['res_wnelu', 'res_elu']
        model_arch['normal_dec'] = ['mconv_e3k5g0']
        model_arch['up_sampling_dec'] = ['mconv_e3k5g0']
        model_arch['normal_pre'] = ['res_wnelu', 'res_elu']
        model_arch['down_sampling_pre'] = ['res_wnelu', 'res_elu']
        model_arch['normal_post'] = ['mconv_e3k5g0']
        model_arch['up_sampling_post'] = ['mconv_e3k5g0']
        model_arch['ar_nn'] = ['mconv_e3k5g0']
    return model_arch