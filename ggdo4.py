import torch
import math
from torch.optim import Optimizer
class GGDO(Optimizer):
    '''
        Implements the Gaussian Gradient Distruption Optimization
    '''
    def __init__(self, params, lr=1e-2, momentum=0.9, weight_decay=5e-4,eps=1e-6):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= eps:
            raise ValueError("Invalid eps value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        
        
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Gaussian Gradients does not support sparse gradients')
                state = self.state[p]

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
                #print(torch.max(variance))
                
                state['step'] += 1
                
                #var = state['mean']
                old_mean = mean.clone()
                mean.mul_(group['momentum']).add_(grad)
                
                
                old_std = std.clone()
                part_var1 = grad.add(-old_mean)
                part_var2 = grad.add(-mean)
                
                new_std = torch.pow(old_std,2).mul(group['momentum']).addcmul(1,part_var1,part_var2).add(group['eps'])                
                new_std = torch.pow(torch.abs_(new_std),1/2)
                
                #new_std = torch.clamp(new_std,0,10**(-state['step']/100))
                
                std.add_(-std).add_(new_std)
                
                #torch.clamp(std,10**(-state['step']))
                #print(torch.max(std))
                #updt = grad
                updt = torch.normal(mean=mean, std=new_std)

                if group['weight_decay'] != 0:
                    updt.add_(group['weight_decay'], p.data)
                #updt = torch.normal(mean=mean, std=variance.sqrt())
                p.data.add_(-group['lr'],updt)
                
        
        # Find the weighted mean and variance of past gradients
        # Clamp the variance using a upper bound,
        # Sample from Normal distribution using mean and variance of the previous gradients
        
        # New mean = mean + weight* (x-mean)/(sum of weights)
        # New variance = variance/(sum of weights minus the new one) + weight * (x - mean) * (x - new mean) /(sum of weights)
        
        
        return loss


# Mean formula
#Mean = mean + w*(x - mean)/ (sum of w)

# Variance fomula
# Variance = variance/(sum of old w) + w*(x-mean)*(x-Mean)/(sum of w) 
    
#Getting the normal distribution
#mean= torch.tensor(0,dtype=torch.float32)
#std= torch.tensor(1,dtype=torch.float32)
#torch.normal(mean=mean, std=std)
