from dataloader.dataloader import BinaryClassificationBaseline, BrainDataset, MNSIT_Dataset, CIFAR10_Dataset, Wave2d_Dataset, Merra2Dataset
from dataloader.dataloader_poseidon import BrownianBridgeTimeDataset, SinesTestDataset, SinesTimeDataset, SinesEasyTimeDataset, PiecewiseConstantsTimeDataset, GaussiansTimeDataset, ComplicatedShearLayerTimeDataset

from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import json
from pathlib import Path
import netCDF4
import numpy as np
import wandb
import torch
import albumentations as A



def save_data(D, filename="data.json"):
    with open(filename, "w") as f:
        json.dump(D, f, indent=4)

def load_data(filename="data.json"):
    with open(filename, "r") as f:
        return json.load(f)

def save_errors(file_name,
                error,
                rel_error,
                likelihood,
                p = 2):
    
    with netCDF4.Dataset(file_name, "w", format="NETCDF4") as ncfile:
        # Define a dimension (same for all arrays)
        
        if rel_error is not None:
            n_samples = rel_error.shape[0]
        else:
            n_samples = likelihood.shape[0]
        
        ncfile.createDimension("samples", n_samples)
        ncfile.createDimension("p", p)

        # Create variables with the same dimension
        if error is not None:
            error_var = ncfile.createVariable("error", "f4", ("samples",))
            error_rel_var = ncfile.createVariable("error_rel", "f4", ("samples",))
            error_var.description = "Lp testing errors of samples"
            error_rel_var.description = "Relative Lp errors of samples"
            
            error_var[:] = error
            error_rel_var[:] = rel_error
            
            mean_lp = ncfile.createVariable("mean_error", "f4")
            median_lp = ncfile.createVariable("median_error", "f4")
            std_lp = ncfile.createVariable("std_error", "f4")
            mean_lp.assignValue(np.mean(error))
            median_lp.assignValue(np.median(error))
            std_lp.assignValue(np.std(error))

            mean_rel_lp = ncfile.createVariable("mean_rel_error", "f4")
            median_rel_lp = ncfile.createVariable("median_rel_error", "f4")
            std_red_lp = ncfile.createVariable("std_rel_error", "f4")
            mean_rel_lp.assignValue(np.mean(rel_error))
            median_rel_lp.assignValue(np.median(rel_error))
            std_red_lp.assignValue(np.std(rel_error))
        
        if likelihood is not None:
            likelihood_var = ncfile.createVariable("likelihood", "f4", ("samples",))
            likelihood_var.description = "Likelihoods of the samples"

            # Store data in variables
            likelihood_var[:] = likelihood

            mean_likelihood = ncfile.createVariable("mean_likelihood", "f4")
            median_likelihood = ncfile.createVariable("median_likelihood", "f4",)
            std_likelihood = ncfile.createVariable("std_likelihood", "f4")
            mean_likelihood.assignValue(np.mean(likelihood))
            median_likelihood.assignValue(np.median(likelihood))
            std_likelihood.assignValue(np.std(likelihood))

def load_errors(file_name, only_likelihood = False, only_errors = False):
    with netCDF4.Dataset(file_name, "r", format="NETCDF4") as ncfile:
        # Read data from variables
        if not only_likelihood:
            error = np.array(ncfile.variables["error"][:])
            error_rel = np.array(ncfile.variables["error_rel"][:])
        else:
            error = None
            error_rel = None
        
        if not only_errors:
            likelihood =  np.array(ncfile.variables["likelihood"][:])
        else:
            likelihood = None
        
    return error, error_rel, likelihood

def find_files_with_extension(folder_path, extension = "pth", tags = [], is_pl = False):
    potential_paths =  list(Path(folder_path).glob(f"*.{extension}"))
    real_paths = []
    for path in potential_paths:
        is_valid = True
        for tag in tags:
            is_valid = is_valid and (tag in str(path))
        if is_valid:
            real_paths.append(str(path))
    
    if is_pl:
        max_step = 0
        max_path = None
        for path in real_paths:
            step = int(path.split("=")[-1].split(".")[0])
            if step>max_step:
                max_path = path
                max_step = step
        return [max_path]
    else:
        return real_paths



def read_cli_regression(parser):
    """Reads command line arguments."""
    # Existing arguments
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--device", type=str, default = 'cuda')
    
    parser.add_argument("--config_arch", type = str, default = "/configs/architectures_regression/config_cno_small.json")

    parser.add_argument("--which_model", type=str, default = 'cno')
    parser.add_argument("--tag", type=str, default = '')
    parser.add_argument("--loss", type=int, default = 1)

    parser.add_argument("--epochs", type=int, default = 100)
    parser.add_argument("--warmup_epochs", type=int, default = 2)
    parser.add_argument("--batch_size", type=int, default = 20)
    parser.add_argument("--peak_lr", type=float, default = 1e-4)
    parser.add_argument("--end_lr", type=float, default = 1e-6)
    
    parser.add_argument("--is_time", type=bool, default = False)
    parser.add_argument("--is_masked", type=bool, default = False)
    

    parser.add_argument("--which_data", type=str, default = 'wave')
    parser.add_argument("--in_dim", type=int, default = 1)
    parser.add_argument("--out_dim", type=int, default = 1)
    parser.add_argument("--N_train", type=int, default = 1000)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--s", type=int, default = 128)

    parser.add_argument("--cno_layers", type=int, default = 4)
    parser.add_argument("--cno_res", type=int, default = 2)
    parser.add_argument("--cno_res_neck", type=int, default = 6)
    parser.add_argument("--cno_channels", type=int, default = 24)
    parser.add_argument("--cno_lift_dim", type=int, default = 128)
    parser.add_argument("--cno_emb_dim", type=int, default = 128)

    parser.add_argument("--unet_param", type=list, default = [32, 64, 128, 256])

    parser.add_argument("--use_generated", type=str, default = None)

    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb-project-name", type=str, default="project_name", help="Name of the wandb project")

    return parser\

def read_cli_diffusion(parser):
    """Reads command line arguments."""
    # Existing arguments
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--device", type=str, default = 'cuda')
    
    parser.add_argument("--tag", type=str, default = '')

    parser.add_argument("--epochs", type=int, default = 100)
    parser.add_argument("--warmup_epochs", type=int, default = 2)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--peak_lr", type=float, default = 0.005)
    parser.add_argument("--end_lr", type=float, default = 1e-5)
    
    parser.add_argument("--which_data", type=str, default = 'wave')
    parser.add_argument("--which_type", type=str, default = 'xy')
    
    parser.add_argument("--sigma", type=float, default = 25.0)
    parser.add_argument("--in_dim", type=int, default = 1)
    parser.add_argument("--out_dim", type=int, default = 1)
    parser.add_argument("--N_train", type=int, default = 1000)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--s", type=int, default = 128)

    parser.add_argument("--unet_param", type=list, default = [64, 128, 256, 512])

    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb-project-name", type=str, default="project_name", help="Name of the wandb project")

    return parser

def read_config(parser):
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    return parser

def read_cli_diffusion_gencfd(parser):
    """Reads command line arguments."""
    # Existing arguments
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--config_arch", type = str, default = "/configs/architectures/config_unet_small.json")

    parser.add_argument("--device", type=str, default = 'cuda')
    
    parser.add_argument("--tag", type=str, default = '')

    parser.add_argument("--epochs", type=int, default = 100)
    parser.add_argument("--warmup_epochs", type=int, default = 2)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--peak_lr", type=float, default = 0.005)
    parser.add_argument("--end_lr", type=float, default = 1e-5)
    
    parser.add_argument("--which_data", type=str, default = 'wave')
    parser.add_argument("--which_type", type=str, default = 'xy')
    parser.add_argument("--is_time", type=bool, default = False)
    parser.add_argument("--is_masked", type=bool, default = False)

    parser.add_argument("--sigma", type=float, default = 25.0)
    parser.add_argument("--in_dim", type=int, default = 1)
    parser.add_argument("--out_dim", type=int, default = 1)
    parser.add_argument("--N_train", type=int, default = 1000)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--s", type=int, default = 128)
    parser.add_argument("--ema_param", type=float, default = 0.999)

    parser.add_argument("--unet_param", type=list, default = [64, 128, 256, 512])
    parser.add_argument("--is_log_uniform", type=bool, default = False)
    parser.add_argument("--is_exploding", type=bool, default = False)
    parser.add_argument("--log_uniform_frac", type=float, default = 3)

    parser.add_argument("--skip", type=bool, default = False)

    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb-project-name", type=str, default="project_name", help="Name of the wandb project")

    return parser

def read_cli_inference(parser):
    parser.add_argument("--config", type=str, default = None, help="Path to config file or JSON string")
    parser.add_argument("--config_regression", type=str, default = None, help="Path to regression model")
    parser.add_argument("--config_diffusion", type=str, default = None, help="Path to diffusion model")
    parser.add_argument("--which_data", type=str, default = None, help="which testing dataset")
    parser.add_argument("--tag_data", type=str, default = None, help="tag for testing dataset")
    parser.add_argument("--device", type=str, default = "cuda")
    parser.add_argument("--N_samples", type=int, default = 128)
    parser.add_argument("--ood_share", type=float, default = 0.0)
    parser.add_argument("--batch_size", type=int, default = 32)
    parser.add_argument("--save_data", type=bool, default = False)
    parser.add_argument("--inference_tag", type=str, default = "")

    parser.add_argument("--is_log_uniform", type=bool, default = False)
    parser.add_argument("--is_exploding", type=bool, default = False)
    parser.add_argument("--log_uniform_frac", type=float, default = 3)

    return parser

def save_id(run_, filepath):
    print(f"Sweep ID: {run_.sweep_id}")
    print(f"Run ID: {run_.id}")
    with open(filepath + '/ids.txt', 'w') as file:
        file.write(f"Sweep ID: {run_.sweep_id}\n")
        file.write(f"Run ID: {run_.id}")

def get_loader(which_data: str,
            which_type: str,
            N_samples:int,
            batch_size: int,
            ood_tag: int = None,
            ood_share:float = 0.0,
            num_workers:int = 4,
            in_dim: int = None,
            out_dim: int = None,
            use_generated: str = None,
            masked_input: list = None,
            is_time: bool = False,
            max_num_time_steps: int = None,
            time_step_size: int = None,
            fix_input_to_time_step: int = None,
            allowed_transitions: list = None,
            return_loader: bool = True,
            rel_time: bool = True,
            baseline_folder: str = None,
            baseline_only_x: bool = False,
            N_max: int = -1):
    
    if which_type == "train":
        shuffle = True
    else:
        shuffle = False

    if which_data == "mnist":
        dataset = MNIST(root = "/path_to_MNIST/", train=True, transform=transforms.ToTensor(), download=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    elif which_data == "cifar_diff":
        dataset = CIFAR10_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "cifar_class":
        dataset = CIFAR10_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    elif which_data == "mnist_diff":
        dataset = MNSIT_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=True)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "mnist_class":
        dataset = MNSIT_Dataset(which = which_type, N_samples = N_samples, ood_share = ood_share, is_diffusion=False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    elif which_data == "wave":
        dataset = Wave2d_Dataset(which = which_type, N_samples = N_samples, is_time = is_time)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    elif which_data == "wave_ood":
        dataset = Wave2d_Dataset(which = which_type, N_samples = N_samples, is_ood=ood_tag, is_time = is_time)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    elif which_data == "brain":
        
        trasnform = None
        if which_type == "train": 
            transform = A.Compose([
                                A.Resize(width=128, height=128, p=1.0),
                                A.HorizontalFlip(p=0.5)])
        elif which_type == "val":
            transform = A.Compose([
            A.Resize(width=128, height=128, p=1.0),
            A.HorizontalFlip(p=0.5),])

        '''
            Training Brain: /path_to_brats2018_HGG_t1.nc/
        '''
        dataset = BrainDataset("/path_to_brats2018_HGG_t1.nc/", which = which_type, transform=transform, is_ood = False)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    elif which_data == "ns_shear":
        dataset = ComplicatedShearLayerTimeDataset(max_num_time_steps = max_num_time_steps, 
                                                    time_step_size = time_step_size,
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_type,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = N_samples,
                                                    data_path = "",
                                                    time_input = True,
                                                    masked_input = masked_input,
                                                    allowed_transitions = allowed_transitions,
                                                    rel_time = rel_time,
                                                    is_time = is_time)
    elif which_data == "ns_brownian":
        dataset = BrownianBridgeTimeDataset(max_num_time_steps = max_num_time_steps, 
                                                    time_step_size = time_step_size,
                                                    fix_input_to_time_step = fix_input_to_time_step,
                                                    which = which_type,
                                                    resolution = 128,
                                                    in_dist = True,
                                                    num_trajectories = N_samples,
                                                    data_path = "",
                                                    time_input = True,
                                                    masked_input = masked_input,
                                                    allowed_transitions = allowed_transitions,
                                                    rel_time = rel_time,
                                                    is_time = is_time)

    elif which_data == "ns_sin":

        if which_type!="test":
            dataset = SinesTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
        else:
            dataset = SinesTestDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
                                    
    elif which_data == "ns_sin_easy":
        dataset = SinesEasyTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time)

    elif which_data == "ns_pwc":
        dataset = PiecewiseConstantsTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
    elif which_data == "ns_gauss":
        dataset = GaussiansTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)

    elif which_data == "ns_mix1":
        dataset1 = SinesTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
        dataset2 = ComplicatedShearLayerTimeDataset(max_num_time_steps = max_num_time_steps, 
                                    time_step_size = time_step_size,
                                    fix_input_to_time_step = fix_input_to_time_step,
                                    which = which_type,
                                    resolution = 128,
                                    in_dist = True,
                                    num_trajectories = N_samples,
                                    data_path = "",
                                    time_input = True,
                                    masked_input = masked_input,
                                    allowed_transitions = allowed_transitions,
                                    rel_time = rel_time,
                                    is_time = is_time)
        dataset3 = GaussiansTimeDataset(max_num_time_steps = max_num_time_steps, 
                                        time_step_size = time_step_size,
                                        fix_input_to_time_step = fix_input_to_time_step,
                                        which = which_type,
                                        resolution = 128,
                                        in_dist = True,
                                        num_trajectories = N_samples,
                                        data_path = "",
                                        time_input = True,
                                        masked_input = masked_input,
                                        allowed_transitions = allowed_transitions,
                                        rel_time = rel_time,
                                        is_time = is_time)
        dataset = [dataset1, dataset2, dataset3]
        dataset = torch.utils.data.ConcatDataset(dataset)

    elif which_data == "baseline_class":
        dataset = BinaryClassificationBaseline(folder = baseline_folder,
                                                which = which_type,
                                                N_samples = N_samples,
                                                N_max = N_max,
                                                mean = 0.0,
                                                std  = 1.0,
                                                only_x = baseline_only_x)

    elif which_data == "merra2":

        is_ood = False
        is_ood_spatial = None
        if ood_tag == 1:
            is_ood = True
        elif ood_tag == 3:
            is_ood_spatial = 3
        elif ood_tag == 4:
            is_ood_spatial = 4
        
        dataset = Merra2Dataset(which = which_type,
                                N_samples = N_samples,
                                max_allowed_transition = allowed_transitions[-1],
                                is_time = True,
                                is_ood = is_ood,
                                is_ood_spatial = is_ood_spatial,
                                in_dim = 1,
                                out_dim = 1)

    if which_type == "test":
        num_workers = 1
    else:
        num_workers = num_workers
    

    if which_type == "val":
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)  # Shuffle indices once
        shuffled_dataset = Subset(dataset, indices)
        loader = DataLoader(shuffled_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    if return_loader:
        return loader
    else:
        return dataset

def select_variable_condition(input_batch,
                              output_batch,
                              which_type = "x&y",
                              mask = None):

  if which_type == "x":
    variable = input_batch
    condition = None
  elif which_type == "y":
    variable = output_batch
    condition = None
  elif which_type == "x&y":
    variable = torch.cat((input_batch, output_batch), axis = 1)
    condition = None

    if mask is not None:
        mask = torch.cat((mask, mask), dim = 1)
  
  return variable, condition, mask


def print_size(model):
    nparams = 0
    nbytes = 0

    for param in model.parameters():
        nparams += param.numel()
        nbytes += param.data.element_size() * param.numel()

    print(f'Total number of model parameters: {nparams}')

    return nparams