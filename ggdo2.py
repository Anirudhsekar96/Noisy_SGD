import torch
import math
from torch.optim import Optimizer
class GGDO(Optimizer):
    '''
        Implements the Gaussian Gradient Distruption Optimization
    '''
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=5e-4,eps=1e-6, noise=0.1):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= noise:
            raise ValueError("Invalid noise value: {}".format(noise))
        
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, noise=noise)
        super(GGDO, self).__init__(params, defaults)
            
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Gaussian Gradients does not support sparse gradients')
                state = self.state[p]

                #if weight_decay != 0:
                #    grad.add_(weight_decay, p.data)

                # State initialization
                if len(state) == 0:
                    # Intialize mean and variance to zero
                    state['mean'] = torch.zeros_like(p.data)
                    state['variance'] = torch.zeros_like(p.data)
                    state['std'] = torch.zeros_like(p.data)
                    state['step'] = 0
                    
                
                mean = state['mean'] # Works now
                var = state['variance']
                std = state['std']
                
                state['step'] += 1
                
                # Getting mean,std at previous step
                old_mean = mean.clone()
                old_std = std.clone()
                
                
                # Calculating gradients
                new_updt = torch.normal(mean=old_mean, std=old_std)
                updt = grad.add(group['noise'],new_updt)
                if weight_decay != 0:
                    updt.add_(weight_decay, p.data)

                # Updating mean
                mean = mean.mul(group['momentum']).add(updt)
                
                part_var1 = grad.add(-old_mean)
                part_var2 = grad.add(-mean)
                
                new_std = torch.pow(old_std,2).mul(group['momentum']).addcmul(1,part_var1,part_var2).add(group['eps'])                
                new_std = torch.pow(torch.abs_(new_std),1/2)
                std.add_(-1,std).add_(new_std)
                
		
                
                p.data.add_(-group['lr'],updt)
                
        
        return loss
