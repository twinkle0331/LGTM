# LGTM
This is the official implementation of our conference paper : Tailoring Instructions to Student’s Learning Levels Boosts Knowledge Distillation (ACL 2023).

## Introduction
### Tab of Content
- [Installation](#1)
- [Usage](#2)
  - [Training](#3)
  <!-- - [Evaluation](#4) -->
  - [Inference](#4)

### Installation
1. Clone the repository
    ```sh
    git clone https://github.com/twinkle0331/LGTM.git
    ```
2. Install the dependencies
    ```sh
    conda create -n lgtm python==3.8.0

    conda activate lgtm

    pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

    cd LGTM

    pip install -r requirements.txt
    ```

### Usage
- #### Training
    ```sh
    python run_glue.py \
        --model_name_or_path google/bert_uncased_L-6_H-768_A-12 \
        --teacher_model bert-base-uncased \
        --task_name sst2 \
        --per_device_train_batch_size 32 \
        --per_device_eval_batch_size 32 \
        --learning_rate 5e-05 \
        --t_learning_rate 3e-05 \
        --alpha_kd 1.0 \
        --temperature 1.0 \
        --num_train_epochs 6 \
        --output_dir <out_dir> \
        --eval_steps 40 \
        --do_train \
        --do_eval \
        --train_teacher \
        --init_classifier_to_zero \
        --use_lgtm
    ```
- #### Inference
    ```sh
    python run_glue.py \
        --model_name_or_path <checkpoint_path> \
        --task_name sst2 \
        --per_device_eval_batch_size 32 \
        --output_dir <predict_out_dir> \
        --do_predict
    ```
    You can get the prediction files for each task and submit them into the GLUE test benchmark.

### License
Distributed under the MIT License. See LICENSE for more information.

### Cite 

If you find it helpful, you can cite our paper in your work.
