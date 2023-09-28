# NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF

[**[Paper]**](https://arxiv.org/abs/2307.09112) | [**[Project page]**](https://numcc.github.io/)

![teaser](media/animation.gif "teaser")

This repository contains the official implementation of the paper:

NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF  
Stefan Lionar, Xiangyu Xu, Lin Min, and Gim Hee Lee   
**Accepted at NeurIPS 2023**  


## Installation
Please see [INSTALL.md](INSTALL.md) for information on installation.


## Data
Please see [DATASET.md](DATASET.md) for information on data preparation.

## Pretrained models

To download the pretrained models, run:

```
mkdir pretrained

# CO3D-V2 Repulsive UDF
wget https://numcc.s3.us-west-1.amazonaws.com/udf-ep99.pth -P pretrained

# Hypersim
wget https://numcc.s3.us-west-1.amazonaws.com/numcc_hypersim_550c.pth -P pretrained
```

## Zero-shot Demo

We provide demo for reconstruction from iPhone capture and AI-generated image.

To run demo for reconstruction from iPhone capture, run:

```
python demo_iphone.py
```

For AI-generated image, run:

```
python demo_web.py
```

Output visualization will be generated to `demo/output.html`.


## CO3D-v2 Experiments

To train NU-MCC from scratch, run:
```
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [NUM_GPU] main_numcc.py --exp_name [YOUR_EXPERIMENT_NAME] --accum_iter [32/NUM_GPU]
```

For example, to train with 4 GPUs:

```
PYTHONHASHSEED=0 torchrun --nproc_per_node 4 main_numcc.py --exp_name numcc_udf --accum_iter 8
```

For evaluation/inference:

```
# Standard inference
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [N_GPU] main_numcc.py --run_val --resume [MODEL_PATH] --n_query_udf [BATCH_QUERY_FOR_REPULSIVE]

# High-resolution
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [N_GPU] main_numcc.py --run_val --resume [MODEL_PATH] --n_query_udf [BATCH_QUERY_FOR_REPULSIVE] --hr --xyz_size_hr 224

# Smoothing
PYTHONHASHSEED=[SEED] torchrun --nproc_per_node [N_GPU] main_numcc.py --run_val --resume [MODEL_PATH] --n_query_udf [BATCH_QUERY_FOR_REPULSIVE] --nneigh 12 --nn_seen 12
```

`PYTHONHASHSEED` defines the random seed for seen images. The argument `--n_query_udf` defines the batch query points for the repulsive force. In general, the higher numbers result in more uniform point distribution, but somewhere around 10k points is already good enough. 

An example of a working command with 4 GPUs:

```
PYTHONHASHSEED=0 torchrun --nproc_per_node 4 main_numcc.py --run_val --resume pretrained/udf-ep99.pth --n_query_udf 48000
```

To run visualization, use `--run_viz` flag. The output will be generated to the folder specified in `--exp_name`. Visualization/evaluation from one class can be specified using `--one_class [OBJECT_CLASS]` flag. Point clouds can be exported by activating `--save_pc` flag.


## Hypersim Experiment

To train on Hypersim dataset, run:

```
torchrun --nproc_per_node 4 main_numcc.py --exp_name [EXPERIMENT_NAME] --hypersim_path [DATASET_PATH] --use_hypersim --blr 5e-5 --epochs 50 --train_epoch_len_multiplier 3200 --accum_iter 8 --n_groups 550
```

For evaluation/visualization, use `--run_val` or `--run_viz` flags and specify the model on `--resume [MODEL_PATH]`.

**Note:** Outputs from the data preparation, i.e., `hypersim_gt_train.pt` and `hypersim_gt_val.pt` need to be placed in this repository's home directory.

## Acknowledgement
This codebase is mainly inherited from Meta Platforms' [MCC](https://github.com/facebookresearch/MCC) codebase.



## Citation
If you find our code or paper useful, please consider citing us:

```bibtex
@inproceedings{lionar2023nu,
  title={NU-MCC: Multiview Compressive Coding with Neighborhood Decoder and Repulsive UDF},
  author={Lionar, Stefan and Xu, Xiangyu and Lin, Min and Lee, Gim Hee},
  booktitle={Advances in neural information processing systems},
  year={2023}
}
```

