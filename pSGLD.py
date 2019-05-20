import torch
import math
from torch.optim import Optimizer
class pSGLD(Optimizer):
    '''
        Implements the pSGLD
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
        super(pSGLD, self).__init__(params, defaults)
            
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
                    raise RuntimeError('SGLD with RMSPROP preconditioner doesnot work with sparse gradients')
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    
                
                square_avg = state['square_avg']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(group['momentum']).addcmul_(1 - group['momentum'], grad, grad)
                #torch.normal(mean=old_mean, std=old_std)
                
                avg = square_avg.sqrt().add_(group['eps'])
                sgld_updt = torch.normal(mean=0,std=avg)
                
                # RMSPROP Update rule
                p.data.addcdiv_(-group['lr'], grad, avg).add_(sgld_updt)
        
        return loss
