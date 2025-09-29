import torch

def mae_per_dim(pred, target):
    return (pred - target).abs().mean(dim=0)

def mse_per_dim(pred, target):
    return   torch.sqrt ((pred - target)**2).mean(dim=0)

def denorm(y, mean, std):
    return y * (std + 1e-8) + mean



