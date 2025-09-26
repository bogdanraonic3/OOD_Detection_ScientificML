import torch
import tqdm
import argparse
import functools
import numpy as np
import os
import time
import copy
from diffusion.likelihood import ode_likelihood
from diffusion.variance_fn import marginal_prob_std_1, diffusion_coeff_1, marginal_prob_std_2, diffusion_coeff_2
from diffusion.model import ScoreNet, EMA
from GenCFD import model

from CNO2d_original_version.CNOModule import CNO
from utils.utils_data import get_loader, load_data, read_cli_inference, find_files_with_extension, save_errors
from visualization.plot import plot_prediction
import matplotlib.pyplot as plt
from diffusion.sampler import Euler_Maruyama_sampler
import matplotlib.pyplot as plt

from GenCFD.model.lightning_wrap.pl_conditional_denoiser import PreconditionedDenoiser_pl
from regression.UNetModule import UNetModel_pl
from regression.FNOModule import FNOModel_pl
from regression.CNOModule_pl import CNOModel_pl
from regression.ViTModulev2 import MultiVit3_pl, MultiVit2_pl, Vit3_pl
import sys
sys.path.append("/path/to/site-packages")  # Replace with the correct path
torch.cuda.empty_cache()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="load parameters for training")
    params = read_cli_inference(parser).parse_args()
    
    if params.config is None:
        config = params
    else:
        config = argparse.Namespace(**load_data(params.config))
    
    if hasattr(config, 'which_ckpt'):
        which_ckpt = config.which_ckpt
    else:
        which_ckpt = None
    

    if which_ckpt is None:
        diff_flag = []
    else:
        diff_flag = ["=" + str(which_ckpt) + "-step"]

    ood_tag = None if config.tag_data is None else int(config.tag_data)
    if which_ckpt is not None and ood_tag is not None:
        ood_tag = str(ood_tag) + "_" + str(which_ckpt) + "ep"

    regression_model_path = str(find_files_with_extension(config.config_regression + "/model", "ckpt", [], is_pl = True)[0])
    regression_config_path = str(find_files_with_extension(config.config_regression, "json", ["param"])[0])
    diffusion_model_path = str(find_files_with_extension(config.config_diffusion + "/model", "ckpt", diff_flag, is_pl = True)[0])
    diffusion_config_path = str(find_files_with_extension(config.config_diffusion, "json", ["param"])[0])

    config_diff = argparse.Namespace(**load_data(diffusion_config_path))
    config_reg  = argparse.Namespace(**load_data(regression_config_path))

    device = config.device
    config_diff_arch = load_data(config_diff.config_arch)
    config_reg_arch = load_data(config_reg.config_arch)

    '''
        Load diffusion model:
    ''' 

    is_exploding = False
    print(is_exploding, "is_exploding", config_diff.sigma, "sigma")
    sigma =  config_diff.sigma

    
    marginal_prob_std_fn = functools.partial(marginal_prob_std_2, sigma_min = 0.001, sigma_max=sigma, device = device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff_2, sigma_min = 0.001, sigma_max=sigma, device = device)
    marginal_prob_std_sample_fn = marginal_prob_std_fn
    diffusion_coeff_sample_fn = diffusion_coeff_fn


    is_time_diff = config_diff.is_time
    which_type = config_diff.which_type
    if which_type == "xy":
        dim = config_diff.in_dim
        dim_cond = config_diff.out_dim 
    elif which_type == "yx":
        dim = config_diff.out_dim
        dim_cond = config_diff.in_dim
    elif which_type == "x":
        dim = config_diff.in_dim
        dim_cond = 0
    elif which_type == "y":
        dim = config_diff.out_dim
        dim_cond = 0
    elif which_type == "x&y":
        dim = config_diff.out_dim + config_diff.in_dim
        dim_cond = 0
    

    diffusion_model = PreconditionedDenoiser_pl(dim = dim, 
                                                dim_cond = dim_cond,
                                                loss_fn = None,
                                                marginal_prob_std_fn = marginal_prob_std_fn,
                                                diffusion_coeff_fn = diffusion_coeff_fn,
                                                config_train = vars(config_diff),
                                                config_arch = config_diff_arch,
                                                is_inference = True
                                                )

    checkpoint = torch.load(diffusion_model_path, map_location = device)
    diffusion_model.load_state_dict(checkpoint["state_dict"])
    diffusion_model = diffusion_model.best_model_ema.to(device)

    config_reg = vars(config_reg)
    config_reg["workdir"] = None
    if config_reg["which_model"] == "cno":
        regression_model = CNOModel_pl(in_dim = config_reg["in_dim"], 
                                        out_dim = config_reg["out_dim"],
                                        loss_fn = None,
                                        config_train = config_reg,
                                        config_arch = config_reg_arch)
    elif config_reg["which_model"] == "unet":
        regression_model = UNetModel_pl(in_dim = config_reg["in_dim"], 
                                        out_dim = config_reg["out_dim"],
                                        loss_fn = None,
                                        config_train = config_reg,
                                        config_arch = config_reg_arch)                               
    elif config_reg["which_model"] == "fno":
        regression_model = FNOModel_pl(in_dim = config_reg["in_dim"], 
                                        out_dim = config_reg["out_dim"],
                                        loss_fn = None,
                                        config_train = config_reg,
                                        config_arch = config_reg_arch)
    elif config_reg["which_model"] == "basic_vit3":
        regression_model = Vit3_pl(in_dim = config_reg["in_dim"], 
                                    out_dim = config_reg["out_dim"],
                                    loss_fn = None,
                                    config_train = config_reg,
                                    config_arch = config_reg_arch)

    checkpoint = torch.load(regression_model_path, map_location = device)
    regression_model.load_state_dict(checkpoint["state_dict"])
    regression_model = regression_model.model.to(device).eval()

    is_time_reg = config_reg["is_time"]
    masked_input = config_reg["is_masked"]
    
    test_loader = get_loader(which_data = config.which_data,
                            which_type = "test",
                            N_samples = config.N_samples,
                            batch_size = config.batch_size,
                            masked_input = masked_input,
                            is_time = is_time_reg,
                            max_num_time_steps = config.max_num_time_steps,
                            time_step_size = config.time_step_size,
                            fix_input_to_time_step = config.fix_input_to_time_step,
                            allowed_transitions = config.allowed_transitions,
                            rel_time = True,
                            ood_tag = ood_tag)
    
    if not os.path.exists(f"{config.config_regression}/errors/{config_diff.tag}"):
        os.makedirs(f"{config.config_regression}/errors/{config_diff.tag}")

    all_bpds = 0.0
    all_items = 0.0

    tqdm_data = tqdm.tqdm(test_loader)
    
    errors_lp = np.zeros(0)
    errors_lp_rel = np.zeros(0)
    
    baseline_avg_grad = config.baseline_avg_grad is not None

    if not baseline_avg_grad:
        likelihoods = np.zeros(0)
    else:
        likelihoods_time = np.zeros(0)
        likelihoods_space = np.zeros(0)
        likelihoods_time_space = np.zeros(0)
        likelihoods_grad_norm = np.zeros(0)


    data_folder = f"{config.config_regression}/errors/{config_diff.tag}/errors_{config.which_data}_{ood_tag}_{config.N_samples}s"
    if not os.path.exists(f"{data_folder}"):
        os.makedirs(f"{data_folder}")

    save_data = config.save_data
    if save_data:
        counter = 0
        if not os.path.exists(f"{config.config_regression}/predictions_{config.which_data}_{ood_tag}"):
            os.makedirs(f"{config.config_regression}/predictions_{config.which_data}_{ood_tag}")
        folder_pred = f"{config.config_regression}/predictions_{config.which_data}_{ood_tag}"
    
    regression_scheme = config.regression_scheme
    dt = config.dt

    p = 1

    cnt_pred = 0

    
    if not baseline_avg_grad:
        epsilon_size = 32
    else:
        epsilon_size = 1
    
    epsilon = torch.randn((epsilon_size, dim, config_reg["s"], config_reg["s"]), device = device).type(torch.float32)
    epsilon = torch.sqrt(torch.prod(torch.tensor(dim, device=device))) * epsilon / torch.norm(epsilon, dim=1, keepdim=True)

    with torch.no_grad():
        for step, batch in enumerate(tqdm_data):
            
            '''
                Unpack:
            '''
            if is_time_reg:
                t_batch, input_batch, output_batch = batch
            else:
                input_batch, output_batch = batch
                t_batch = None
            
            if t_batch is not None and which_type!= "x":
                T = dt*np.sum(np.array(regression_scheme))
                t_batch_diff = T*torch.ones_like(t_batch)
            else:
                t_batch_diff = None

            '''
                Diffusion type:
            '''
            if which_type == "yx":
                condition = input_batch.to(device)
            elif which_type == "x&y":
                condition = None
            if which_type == "x":
                condition = None


            '''
                Sampler (guided) & save
            '''
            if step == 1000:
                guidance_condition = input_batch, (0, input_batch.shape[1])
                samples = Euler_Maruyama_sampler(diffusion_model,
                                                marginal_prob_std_sample_fn,
                                                diffusion_coeff_sample_fn,
                                                condition,
                                                t_batch_diff,
                                                batch_size=config.batch_size,
                                                num_steps=128,
                                                device=device,
                                                dimension = (dim, input_batch.shape[-1], input_batch.shape[-1]),
                                                eps=1e-3,
                                                is_skip = diffusion_model.is_skip,
                                                guidance_condition=guidance_condition)   
                batch_size_plot = min(config.batch_size, 8)
                plot_prediction(batch_size_plot, (1,1), input_batch[:batch_size_plot], output_batch[:batch_size_plot], samples[:batch_size_plot], f"{data_folder}/generated.png", is_cifar = False)
                plt.close()


            '''
                Send to device & make a copy for later use
            '''
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)
            input_batch_copy = copy.deepcopy(input_batch).to(device)
            output_copy = copy.deepcopy(output_batch).detach().cpu().numpy()
            

            '''
                Make predictions (regression) - AR prediction based on regression_scheme and dt
            '''
            input_batch_diff = copy.deepcopy(input_batch).to(device)
            if is_time_reg:
                for i, reg_step in enumerate(regression_scheme):
                    t = dt * reg_step
                    t_batch = t * torch.ones_like(t_batch).to(device)
                    t_batch = t_batch.type(torch.float32)
                    output_pred_batch = regression_model(input_batch, t_batch)
                    input_batch[:,:config_reg["out_dim"]] = output_pred_batch
            
            '''
                Select variables for ode_likelihood based on diffusion type
            '''
            
            if which_type == "yx":
                variable = output_pred_batch
            elif which_type == "x&y":
                variable = torch.cat((input_batch_diff, output_pred_batch), axis = 1)
            if which_type == "x":
                variable = input_batch_diff
                
            
            '''
                Save the predictions if the flag is up
            '''
            if save_data:
                pred_copy = copy.deepcopy(output_pred_batch).detach().cpu().numpy()
                for i in range(pred_copy.shape[0]):
                    np.save(f"{folder_pred}/sample_{cnt_pred}_pred.npy", pred_copy[i])
                    np.save(f"{folder_pred}/sample_{cnt_pred}_out.npy", output_copy[i])
                    np.save(f"{folder_pred}/sample_{cnt_pred}_inp.npy",input_batch_copy.detach().cpu().numpy()[i])
                    cnt_pred+=1
                del pred_copy


            '''
                Concatenate the errors
            '''
            loss_rel_lp = torch.mean(abs(output_pred_batch - output_batch), dim = [1,2,3])/ torch.mean(abs(output_batch), dim = [1,2,3])
            loss_lp = (torch.mean(abs(output_pred_batch - output_batch) ** p, (-3, -2, -1))) ** (1 / p)
            print(loss_rel_lp)
            loss_rel_lp = loss_rel_lp.reshape(-1).detach().cpu().numpy()
            loss_lp = loss_lp.reshape(-1).detach().cpu().numpy()
            errors_lp = np.concatenate((errors_lp, loss_lp), axis = 0)
            errors_lp_rel = np.concatenate((errors_lp_rel, loss_rel_lp), axis = 0)


            guidance_condition = None
            
            if is_time_reg:
                if config.is_ar:
                    regression_scheme_ = regression_scheme
                else:
                    regression_scheme_ = [np.sum(np.array(regression_scheme))]
                
                input_batch= copy.deepcopy(input_batch_diff)
                for i, reg_step in enumerate(regression_scheme_):
                    t = dt * reg_step
                    t_batch = t * torch.ones_like(t_batch).to(device)
                    t_batch = t_batch.type(torch.float32)
                    output_pred_batch = regression_model(input_batch, t_batch)
                    input_batch[:,:config_reg["out_dim"]] = output_pred_batch

                if which_type == "yx":
                    variable = output_pred_batch
                elif which_type == "x&y":
                    variable = torch.cat((input_batch_diff, output_pred_batch), axis = 1)
                if which_type == "x":
                    variable = input_batch_diff

                if config.is_diff:
                    result = ode_likelihood(diffusion_model,
                                            variable,
                                            condition,
                                            marginal_prob_std_fn,
                                            diffusion_coeff_fn,
                                            t_batch = t_batch,
                                            batch_size=epsilon_size,
                                            device='cuda',
                                            eps = 1e-6,
                                            rtol = 1e-6,
                                            atol = 1e-6,
                                            epsilon = epsilon,
                                            reduce_prior = True,
                                            avg_grad = baseline_avg_grad)
                
                if baseline_avg_grad:
                    grads, time_grads, grad_norms =  result
                    grads_avg = torch.mean(torch.abs(grads), axis = (1,2,3)).reshape(-1).detach().cpu().numpy()
                    time_grads_avg = torch.mean(torch.abs(time_grads), axis = (1,2,3)).reshape(-1).detach().cpu().numpy()
                    grad_norms = grad_norms.reshape(-1).detach().cpu().numpy()

                    bpd_time = time_grads_avg
                    bpd_space = grads_avg
                    bpd_time_space = (grads_avg + time_grads_avg)
                    bpd_grad_norm = grad_norms

                    likelihoods_time = np.concatenate((likelihoods_time, bpd_time), axis = 0)
                    likelihoods_space = np.concatenate((likelihoods_space, bpd_space), axis = 0)
                    likelihoods_time_space = np.concatenate((likelihoods_time_space, bpd_time_space), axis = 0)
                    likelihoods_grad_norm = np.concatenate((likelihoods_grad_norm, bpd_grad_norm), axis = 0)

                    median_likelihood = np.median(likelihoods_grad_norm)
                    tqdm_data.set_description("Median bits/dim: {:5f}".format(median_likelihood))

                else:
                    _, prior, delta = result
                    bpd = prior + delta
                    bpd = bpd.reshape(-1).detach().cpu().numpy()
                    likelihoods = np.concatenate((likelihoods, bpd), axis = 0)
                    median_likelihood = np.median(likelihoods)
                    tqdm_data.set_description("Median bits/dim: {:5f}".format(median_likelihood))
                
    print(np.median(errors_lp_rel), "MEDIAN")
    
    inference_tag = config.inference_tag

    if baseline_avg_grad:
        tag_grad_baselines = ["time", "space", "time_space", "grad_norm"]
        data = [likelihoods_time, likelihoods_space, likelihoods_time_space, likelihoods_grad_norm]

    if baseline_avg_grad:
        for i,tag_grad_baseline in enumerate(tag_grad_baselines):
            tag_file = f"grad_baseline_{tag_grad_baseline}"
            likelihoods = data[i]
            save_errors(file_name = f"{data_folder}/errors_{tag_file}_{config_diff.tag}_{config.which_data}_{ood_tag}_{config.N_samples}samples{inference_tag}.nc",
                        error = errors_lp,
                        rel_error = errors_lp_rel,
                        likelihood = likelihoods,
                        p = 1)
    
    else:
        save_errors(file_name = f"{data_folder}/errors_{config_diff.tag}_{config.which_data}_{ood_tag}_{config.N_samples}samples{inference_tag}.nc",
                    error = errors_lp,
                    rel_error = errors_lp_rel,
                    likelihood = likelihoods,
                     p = 1)
