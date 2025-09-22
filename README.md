# Towards a Certificate of Trust: Task-Aware OOD Detection for Scientific AI

This repository contains implementations of the paper "Towards a Certificate of Trust: Task-Aware OOD Detection for Scientific AI".  

The **GenCFD** and **CNO** models used in this paper are adapted from the repositories:

- [GenCFD](https://github.com/camlab-ethz/GenCFD)  
- [CNO](https://github.com/camlab-ethz/poseidon)  

Some dataloaders are adapted from [Poseidon](https://github.com/camlab-ethz/poseidon) implementation.

While the code is also available in this project, please note that the **original implementations** can be found in their respective repositories.

⚠️ **Note**: The code has only been tested on **GPUs**. Running on CPUs may cause issues.

## 📦 Requirements
- Python 3.8+
- PyTorch + PyTorch Lightning
- [Weights & Biases (wandb)](https://wandb.ai/) account (optional, for logging)

Install dependencies (example):

    pip install -r requirements.txt

## 🔹 Regression Model

### Steps to Train
1. **Set working directory** in `train_regression_pl.py` (default: `trained_models`).  
2. **Configure wandb**: provide your wandb account for logging metrics. Otherwise, disable logging manually (see: `regression/GeneralModule_pl.py` and `GenCFD/model/lightning_wrap/pl_wrapper.py`).  
3. **Create a config file**. Example:

    {
   
        "config": null,
        "device": "cpu",
        "which_model": "cno",
        "tag": "tmp",
        "loss": 1,
        "epochs": 100,
        "warmup_epochs": 0,
        "batch_size": 32,
        "peak_lr": 0.0001,
        "end_lr": 0.00001,
        "which_data": "wave",
        "in_dim": 1,
        "out_dim": 1,
        "N_train": 128,
        "ood_share": 0.0,
        "is_time": true,
        "is_masked": null,
        "max_num_time_steps": 1,
        "time_step_size": 1,
        "fix_input_to_time_step": null,
        "allowed_transitions": [0],
        "s": 128,
        "config_arch": "/architectures_regression/config_cno_very_small_att.json",
        "wandb_project_name": "your_project",
        "wandb_run_name": "_1"
    }

**Variable explanations**:  
- `config`: keep `null`  
- `device`: `"cpu"` or `"cuda"`  
- `which_model`: `"cno"`, `"unet"`, `"basic_vit3"`, or `"fno"`  
- `tag`: string identifier for the model (important for saving)  
- `loss`: loss function index  
- `epochs`: number of epochs  
- `warmup_epochs`: number of warmup epochs  
- `batch_size`: training batch size  
- `peak_lr`: peak learning rate  
- `end_lr`: final learning rate at end of training  
- `which_data`: dataset/experiment (`wave`, `ns_mix`, `ns_pwc`, etc.)  
- `in_dim`, `out_dim`: input and output channel dimensions  
- `N_train`: number of training samples/trajectories  
- `ood_share`: OOD fraction (used in classification)  
- `is_time`: whether time conditioning is used  
- `is_masked`: keep `null` unless using masks  
- `max_num_time_steps`: max number of time steps (set to 7 for time-dependent NS)  
- `time_step_size`: step size in trajectory (set to 2 for NS)  
- `fix_input_to_time_step`: keep `null` unless fixed inputs are required  
- `allowed_transitions`: transitions allowed in all2all strategy (set to `[1,2,3,4,5,6,7]` for NS)  
- `s`: resolution  
- `config_arch`: path to architecture config file (examples in `architectures_regression`)  
- `wandb_project_name`: project name for wandb logging  
- `wandb_run_name`: run tag for wandb  

### Run Training

    python3 train_regression_pl.py --config=/path_to_config_file/

## 🔹 Diffusion Model

### Steps to Train

**Create a config file**. Example:

    {
   
        "config": null,
        "device": "cuda",
        "tag": "tmp",
        "epochs": 200,
        "warmup_epochs": 0,
        "batch_size": 40,
        "peak_lr": 0.0002,
        "end_lr": 0.00001,
        "which_data": "ns_pwc",
        "is_time": true,
        "is_masked": null,
        "max_num_time_steps": 10,
        "time_step_size": 2,
        "fix_input_to_time_step": null,
        "allowed_transitions": [1,2,3,4,5,6,7],
        "which_type": "x&y",
        "sigma": 100.0,
        "in_dim": 2,
        "out_dim": 2,
        "N_train": 1000,
        "ood_share": 0.0,
        "s": 128,
        "is_log_uniform": false,
        "log_uniform_frac": 1.0,
        "is_exploding": true,
        "ema_param": 0.999,
        "skip": true,
        "config_arch": "/configs/architectures/config_unet_base.json",
        "wandb_project_name": "your_project",
        "wandb_run_name": "_1"
    }

**Variable explanations** (in addition to regression ones):  
- `which_type`: `"joint"` (`x&y`) or `"only x"` (conditioning type)  
- `sigma`: max noise level for denoiser  
- `is_log_uniform`: whether to use log-uniform scheme  
- `log_uniform_frac`: scaling factor for log-uniform scheme  
- `is_exploding`: whether to use exploding schedule  
- `ema_param`: exponential moving average decay parameter  
- `skip`: whether the denoiser uses skip connections  

### Run Training

    python3 train_diffusion_pl.py --config=/path_to_config_file/

## 🔹 Inference

To obtain estimated likelihoods (or other diffusion-based certificates), you need a config file. Example:

    {
        "config_regression": "/path_to_regression_model/",
        "config_diffusion": "/path_to_diffusion_model/",
        "which_data": "ns_pwc",
        "tag_data": "3",
        "device": "cuda",
        "N_samples": 123,
        "ood_share": 0.0,
        "batch_size": 8,
        "baseline_avg_grad": null,
        "which_ckpt": null,
        "save_data": false,
        "is_diff": true,
        "is_ar": true,
        "is_time": true,
        "is_masked": null,
        "max_num_time_steps": 7,
        "time_step_size": 2,
        "fix_input_to_time_step": null,
        "allowed_transitions": [7],
        "regression_scheme": [1,1,1,1,1,1,1],
        "dt": 0.1,
        "inference_tag": "1",
        "num_gen": 0
    }

**Variable explanations** (new ones):  
- `config_regression`: path to regression model config  
- `config_diffusion`: path to diffusion model config  
- `tag_data`: tag for OOD testing data  
- `N_samples`: number of test samples  
- `baseline_avg_grad`: if using gradient-based baselines, set to `"time"`, `"space"`, `"time_space"`  
- `which_ckpt`: checkpoint to load (set `null` for default)  
- `save_data`: whether to save generated samples  
- `is_diff`: run diffusion model  
- `is_ar`: whether to run autoregressive evaluation  
- `regression_scheme`: AR scheme used during evaluation  
- `dt`: autoregressive time step size  
- `inference_tag`: identifier for inference run  
- `num_gen`: number of generations (set `0`)  

### Run Inference

    python3 inference.py --config=/path_to_config_file/

## 🔹 Segmentation

The segmentation training follows [this script](https://github.com/s0mnaths/Brain-Tumor-Segmentation/blob/master/notebooks/brain_tumor_segmentation.ipynb).  
The backbone used for segmentation is the **CNO model**, with the final layer being a **binary segmentation head**.

## 🔹 1D Experiments

For running 1D experiments, please use the provided Jupyter notebooks:

- `1d_notebooks/notebook_1d_diffusion.ipynb`  
- `1d_notebooks/notebook_1d.ipynb`

