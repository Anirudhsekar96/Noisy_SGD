import torch
import math
from torch.optim import Optimizer
class SGLD(Optimizer):
    '''
        Implements the SGLD
    '''
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=5e-4,eps=1e-6, noise=0.3, gamma=0.55):
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
        if not 0.0 <= gamma:
            raise ValueError("Invalid noise value: {}".format(gamma))
        
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps, noise=noise, gamma=gamma)
        super(SGLD, self).__init__(params, defaults)
            
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
                    raise RuntimeError('SGLD does not support sparse gradients')
                state = self.state[p]

                if weight_decay != 0:
                    grad.add_(weight_decay, p.data)

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                state['step'] += 1

                # std = (nu/(1+t)^gamma)^(1/2)
                

                #std = torch.pow(torch.div(group['noise'],torch.pow(state['step'].add(1),group['gamma'])),(0.5))
                
                std = torch.div(group['lr'],(state['step']**(group['gamma'])))
                nr = torch.normal(mean=0, std = std)
                updt = grad.add(nr)
                p.data.add_(-group['lr'],updt)
                
        
        return loss
