import torch
import copy
import tqdm
from diffusion.lr_schedulers import get_lr_cosine_linear_warmup
import torch.nn as nn
from visualization.plot import plot_prediction
import wandb
import matplotlib.pyplot as plt

def train(model, 
        optimizer, 
        training_loader, 
        val_loader, 
        epochs = 10, 
        peak_lr = 1e-3,
        end_lr = 1e-6,
        warmup_epochs = 2,
        loss =  nn.MSELoss(), 
        dir_path = "",
        tag = "",
        device = "cpu"):

    tqdm_epoch = tqdm.trange(epochs)
    best_model_testing_error = 1000
    dict_wandb = dict()

    for epoch in tqdm_epoch:
        train_mse = 0.0
        model.train()
        lr = get_lr_cosine_linear_warmup(epoch, warmup_epochs, epochs, peak_lr, end_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # Update optimizer LR

        for step, (input_batch, output_batch) in enumerate(training_loader):
            
            input_batch = input_batch.to(device)
            output_batch = output_batch.to(device)

            optimizer.zero_grad()
            output_pred_batch = model(input_batch, None, None)
            loss_f = loss(output_pred_batch, output_batch)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_loader)
        dict_wandb['train/loss'] = train_mse

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            for step, (input_batch, output_batch) in enumerate(val_loader):
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)

                output_pred_batch = model(input_batch, None, None)
                loss_f = torch.mean(torch.norm(output_pred_batch - output_batch, p=2, dim = [1,2,3])/torch.norm(output_batch, p=2, dim = [1,2,3]))

                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(val_loader)

        dict_wandb['train/val_loss'] = test_relative_l2

        if test_relative_l2 < best_model_testing_error:
            best_model_testing_error = test_relative_l2
            best_model = copy.deepcopy(model)
            torch.save(model.state_dict(), f"{dir_path}/ckpt_regression_{tag}.pth")

        dict_wandb['train/best_val_loss'] =  best_model_testing_error
        dict_wandb['train/epoch'] = epoch + 1
        wandb.log(dict_wandb, step = epoch + 1)

        if epoch%50 == 0:
            fig = plot_prediction(4, (1,1), input_batch, output_batch, output_pred_batch, f"{dir_path}/train_plot_ep_{epoch}.png")
            wandb.log({f"fig_train/train_plot_ep_{epoch+1}": wandb.Image(fig)})
            plt.close()

        tqdm_epoch.set_description('Train: {:.5f} Val: {:.5f}'.format(train_mse, test_relative_l2))

