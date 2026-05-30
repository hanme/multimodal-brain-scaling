import torch
from torch.optim.lr_scheduler import _LRScheduler, StepLR

class CosineAnnealingWithWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with cosine decay and linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for warmup phase
        total_epochs: Total number of epochs for training
        min_lr: Minimum learning rate at the end of training (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # During warmup phase
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        
        # After warmup phase - cosine decay
        progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
        progress = torch.clamp(torch.tensor(progress), 0.0, 1.0)
        
        cosine_decay = 0.5 * (1 + torch.cos(progress * torch.pi))
        
        return [(base_lr - self.min_lr) * cosine_decay.item() + self.min_lr 
                for base_lr in self.base_lrs]
        
class StepWarmupScheduler(_LRScheduler):
    """
    Learning rate scheduler with step decay and linear warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_epochs: Number of epochs for warmup phase
        step_size: Number of epochs between learning rate decays
        gamma: Multiplicative factor of learning rate decay (default: 0.1)
        min_lr: Minimum learning rate (default: 0)
        last_epoch: The index of last epoch (default: -1)
    """
    def __init__(self, optimizer, warmup_epochs, step_size, gamma=0.1, 
                 min_lr=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # During warmup phase
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        
        # After warmup phase - step decay
        steps_after_warmup = (self.last_epoch - self.warmup_epochs) // self.step_size
        decay_factor = self.gamma ** steps_after_warmup
        
        return [max(base_lr * decay_factor, self.min_lr) 
                for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        """
        Called by torch.optim.lr_scheduler for logging/monitoring purposes.
        """
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        
        steps_after_warmup = (self.last_epoch - self.warmup_epochs) // self.step_size
        decay_factor = self.gamma ** steps_after_warmup
        
        return [max(base_lr * decay_factor, self.min_lr) 
                for base_lr in self.base_lrs]


def create_scheduler(optimizer:torch.optim.Optimizer, **kwargs) -> _LRScheduler:
    scheduler = kwargs.get('lr_scheduler', 'cosineannealinglrwarmup')
    
    # If updating interval is evert iteration step, convert epochs to iterations
    warmup_epochs = kwargs.get('lr_warmup_duration', 10)
    total_epochs = kwargs.get('max_epochs', 100)
    lr_scheduler_interval = kwargs.get('lr_scheduler_interval', 'step')
    if lr_scheduler_interval == 'step':
        # iterations_per_epoch = kwargs['iterations_per_epoch']
        iterations_per_epoch = kwargs.get('iterations_per_epoch', None)
        warmup_epochs = warmup_epochs * iterations_per_epoch
        total_epochs = total_epochs * iterations_per_epoch
        
    print(f"iterations_per_epoch {iterations_per_epoch} with warmup_epochs={warmup_epochs} and total_epochs={total_epochs}")
    
    
    match scheduler:
        case 'cosineannealinglrwarmup':

            min_lr = kwargs.get('min_lr', 1e-5)
            return CosineAnnealingWithWarmupScheduler(
                optimizer=optimizer, 
                warmup_epochs=warmup_epochs, 
                total_epochs=total_epochs, 
                min_lr=min_lr
            )
    
        case 'steplr':
            step_size = kwargs.get('lr_step_size', 30)
            gamma = kwargs.get('lr_gamma', 0.1)
            return StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
        
        case 'steplrwarmup':
            step_size = kwargs.get('lr_step_size', 30)
            gamma = kwargs.get('lr_gamma', 0.1)
            min_lr = kwargs.get('min_lr', 1e-5)
            return StepWarmupScheduler(
                optimizer=optimizer, 
                warmup_epochs=warmup_epochs, 
                step_size=step_size, 
                gamma=gamma,
                min_lr=min_lr
            )
        
        case _:
            raise NotImplementedError(f"Scheduler {scheduler} not implemented yet")