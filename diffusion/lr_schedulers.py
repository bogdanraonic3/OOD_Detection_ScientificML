import torch
import numpy as np

# Function to compute learning rate for each epoch
def get_lr_cosine_linear_warmup(epoch, warmup_epochs, total_epochs, peak_lr, end_lr):
    if epoch < warmup_epochs:
        # Linear Warm-up
        return end_lr + (peak_lr - end_lr) * (epoch / warmup_epochs)
    elif epoch <= total_epochs:
        # Cosine Annealing
        cosine_decay = 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        return end_lr + (peak_lr - end_lr) * cosine_decay
    else:
        return end_lr

class CosineLinearWarmupCustomScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, peak_lr, end_lr, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            get_lr_cosine_linear_warmup(self.last_epoch + 1, self.warmup_epochs, self.total_epochs, self.peak_lr, self.end_lr)
            for _ in self.base_lrs
        ]