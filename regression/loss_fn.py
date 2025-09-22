import torch
import numpy as np

def relative_lp_loss_fn(out, 
                        pred, 
                        mask = None,
                        reduction = True,
                        p=2):
    """
    Computes Lp loss for different channel groups efficiently without a for-loop.

    Args:
        out (torch.Tensor): Tensor of shape [b, C, s, s] (ground truth).
        pred (torch.Tensor): Tensor of shape [b, C, s, s] (predictions).
        mask (list): List of integers defining the mask of the loss function.
        reduction (bool): Should I reduce it to the scalar?
        p (int): The order of the norm (default is L2 loss).

    Returns:
        torch.Tensor: Scalar tensor representing the total loss.
    """

    assert p in [1,2]

    C = out.shape[1]  # Number of channels

    # Set mask to ones if None (no masking)
    if mask is None:
        mask = torch.ones((out.shape[0], C), device=out.device, dtype=out.dtype)
    
    # Reshape mask to [B, C, 1, 1] for broadcasting across batch and spatial dimensions
    M = torch.sum(mask, dim = [1]).reshape(-1,1)
    mask = mask.view(-1, C, 1, 1)
    
    diff = torch.abs(out - pred) if p == 1 else (out - pred) ** 2
    diff = mask * diff

    if p == 1:
        diff = torch.mean(diff, dim =[2,3])/(torch.mean(torch.abs(out), dim = [2,3]) + 1e-10)
    else:
        diff = torch.mean(diff, dim =[2,3])/(torch.mean(out**2, dim = [2,3]) + 1e-10)

    diff = diff * (C/M)
    if not reduction:
        return torch.mean(diff, dim = [1])
    else:
        return torch.mean(diff)

def relative_lp_loss_separate_fn(out, 
                                pred,
                                separate_dim = None,
                                mask = None,
                                reduction = True,
                                p=2):
    """
    Computes Lp loss for different channel groups efficiently without a for-loop.

    Args:
        out (torch.Tensor): Tensor of shape [b, C, s, s] (ground truth).
        pred (torch.Tensor): Tensor of shape [b, C, s, s] (predictions).
        mask (list): List of integers defining the mask of the loss function.
        reduction (bool): Should I reduce it to the scalar?
        p (int): The order of the norm (default is L2 loss).

    Returns:
        torch.Tensor: Scalar tensor representing the total loss.
    """

    assert p in [1,2]
    assert separate_dim is not None

    C = out.shape[1]  # Number of channels
    # Set mask to ones if None (no masking)
    if mask is None:
        mask = torch.ones((out.shape[0], C), device=out.device, dtype=out.dtype)
    # Reshape mask to [B, C, 1, 1] for broadcasting across batch and spatial dimensions
    M = torch.sum(mask, dim = [1]).reshape(-1,1)
    mask = mask.view(-1, C, 1, 1)
    
    diff = torch.abs(out - pred) if p == 1 else (out - pred) ** 2
    diff = mask * diff
    
    if reduction:
        loss = 0.
        weight = 1./(len(separate_dim) - 1)
        for i in range(len(separate_dim) - 1):
            dim_in = separate_dim[i]
            dim_out = separate_dim[i+1]
            
            loss = loss + weight*torch.mean(diff[:, dim_in:dim_out])/(torch.mean(torch.abs(out[:, dim_in:dim_out])) + 1e-10)
        
        return loss
    else:
        loss = torch.zeros(out.shape[0], device = out.device)
        weight = 1./(len(separate_dim) - 1)
        for i in range(len(separate_dim) - 1):
            dim_in = separate_dim[i]
            dim_out = separate_dim[i+1]
            loss = loss + weight*torch.mean(diff[:, dim_in:dim_out], dim = [1,2,3])/(torch.mean(torch.abs(out[:, dim_in:dim_out]), dim = [1,2,3]) + 1e-10)
        return loss