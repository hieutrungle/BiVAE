# Folder:


# Quick start
1. Training: 

## Running the main NVAE training and evaluation scripts

**MNIST:**

<details><summary>MNIST</summary>

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python ../BiVAE/main.py --use_se --num_initial_channel 16 --num_process_blocks 2 \
    --num_preprocess_cells 1 --num_postprocess_cells 1 --num_cell_per_group_enc 1 \
    --num_cell_per_group_dec 1 --num_groups_per_scale 1 --num_scales 2 --batch_size 256 \
    --learning_rate 0.001 --learning_rate_min 0.000005 --epochs 100 \
    --model_path ./model_output/mnist_iaf
```
</details>

<details><summary>CIFAR-10</summary>

**CESM:**

```shell script
export EXPR_ID=UNIQUE_EXPR_ID
export DATA_DIR=PATH_TO_DATA_DIR
export CHECKPOINT_DIR=PATH_TO_CHECKPOINT_DIR
export CODE_DIR=PATH_TO_CODE_DIR
cd $CODE_DIR
python ../BiVAE/main.py --use_se --num_initial_channel 16 --num_process_blocks 3 \
    --num_preprocess_cells 1 --num_postprocess_cells 1 --num_cell_per_group_enc 1 \
    --num_cell_per_group_dec 1 --num_groups_per_scale 1 --num_scales 2 --batch_size 128 \
    --learning_rate 0.001 --learning_rate_min 0.000005 --epochs 200 \
    --model_path ./model_output/cesm_iaf_groups1_scales3 --data_path ../BiVAE/data --dataset cesm \
    --tile_size 64
```
</details>

2. Evaluating: ``

3. Generating images: ``


# GitHub URL
**[https://github.com/hieutrungle](https://github.com/hieutrungle)**

# License
This program is created by [Hieu Le](https://github.com/hietrungle)