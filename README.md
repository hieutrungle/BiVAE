# Bidirectional Variational Autoencoder with Inverted Autoregressive Flows:

## 1. Requirements
The model is built in Python 3.9 using Tensorflow 2.7.0. Use the following command to install the requirements:
```
pip install -r requirements.txt
``` 


## Dataset

Dataset for CESM can be downloaded at: https://sdrbench.github.io/

Sample download instruction:

```shell script
wget https://97235036-3749-11e7-bcdc-22000b9a448b.e.globus.org/ds131.2/Data-Reduction-Repo/raw-data/CESM-ATM/SDRBENCH-CESM-ATM-26x1800x3600.tar.gz
tar -xvzf SDRBENCH-CESM-ATM-26x1800x3600.tar.gz -C cesm_data_2
```

## 3. Running the training scripts

<details><summary>MNIST</summary>

```shell script
python main.py --use_se --num_initial_channel 16 --num_process_blocks 2 \
    --num_preprocess_cells 1 --num_postprocess_cells 1 --num_cell_per_group_enc 1 \
    --num_cell_per_group_dec 1 --num_groups_per_scale 1 --num_scales 2 --batch_size 256 \
    --learning_rate 0.001 --learning_rate_min 0.000005 --epochs 100 \
    --model_path ./model_output/mnist_iaf
```

</details>

<details><summary>CESM-Cloud</summary>

```shell script
python main.py --use_se --num_initial_channel 16 --num_process_blocks 3 \
    --num_preprocess_cells 1 --num_postprocess_cells 1 --num_cell_per_group_enc 1 \
    --num_cell_per_group_dec 1 --num_groups_per_scale 1 --num_scales 2 --batch_size 128 \
    --learning_rate 0.001 --learning_rate_min 0.000005 --epochs 200 \
    --model_path ./model_output/cesm_iaf --data_path ./data --dataset cesm \
    --tile_size 64
```

</details>

## 3. Evaluating: ``

## 4. Generating images: ``


# GitHub URL
**[https://github.com/hieutrungle](https://github.com/hieutrungle)**

# License
This program is created by [Hieu Le](https://github.com/hietrungle)