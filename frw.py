# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable


class FRW(nn.Module):
    """ Implements the FRW (Feature ReWeighting) layer from the paper 'Deep Person 
        Re-Identification with Improved Embedding and Efficient Training'. It's output
        is the element-wise multiplication between the input and the weights.
    """
    def __init__(self, feature_dim):
        super(FRW, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(feature_dim))
        self.reset_parameters()
        self.feature_dim = feature_dim
        
    def reset_parameters(self):
        self.weight.data.fill_(1)
        
    def forward(self, x):
        return torch.mul(x, self.weight)
    
    def __repr__(self):
        return self.__class__.__name__ + '(feature_dim=' + str(self.feature_dim) + ')'
    
    
    
class L2_penalty_with_constant(nn.Module):
    """ Implements L2 penalty that pushes the squared norm of the weights close 
        to a constant value of 2*C instead of zero. 
    """
    def __init__(self, C, loss_weight=1):
        super(L2_penalty_with_constant, self).__init__()
        assert isinstance(C, (int, float)), "C must be of type 'int' or 'float'"
        self.register_buffer('C', torch.Tensor([C]))
        self.loss_weight = loss_weight
        
    def forward(self, x, target=None):
        
        if target is None:
            target = Variable(torch.zeros(1), requires_grad=False)
        if x.is_cuda:
            target = target.cuda()
        C = Variable(self.C, requires_grad=False)
        
        return self.loss_weight * (0.5 * (x - target).pow(2).sum() - C).pow(2)
    
    
    
if __name__ == "__main__":
    from torch.autograd import Variable
    import numpy as np
    from argparse import ArgumentParser
    
    
    parser = ArgumentParser()
    parser.add_argument('test', choices=['frw', 'l2'], help='run tests for "frw" or "l2"')
    parser.add_argument('--C', default=200, type=float, help='Value of C when testing L2_penalty_with_constant')
    args = parser.parse_args()
    
    
    def test_frw():
        frw = FRW(10)
        frw.weight.data = torch.arange(0, 10)
        
        inp = Variable(2*torch.ones(10))
        out = frw(inp)
        
        print('FRW weights:')
        print(frw.weight.data)
        print('Input:')
        print(inp)
        print('Output:')
        print(out)
        
        assert torch.equal(out.data, torch.arange(0, 20, 2))
        
    
    def test_l2():
        A = Variable(torch.randn(10, 20), requires_grad=True)
        l2_with_C = L2_penalty_with_constant(args.C)
        print('Initial A tensor:\n', A)
        print('Initial A squared norm: ', A.pow(2).sum().data.numpy())
        print('Constraint C value: ', l2_with_C.C)
        
        i = 0
        cost = np.Inf
        last_cost = np.Inf
        lr = 0.01
        while cost > 1e-6 and i < 100000:
            if A.grad is not None:
                A.grad.data.fill_(0)
            loss = l2_with_C(A)
            if i%20 == 0:
                print('Step {}, loss: {}'.format(i, loss.data.numpy()))
            loss.backward()
            cost = loss.data.numpy()
            if last_cost - cost < 0.1: 
                lr *= 0.8
                
            torch.nn.utils.clip_grad_norm([A], 500)
            A.data -= lr * A.grad.data
            i += 1
            last_cost = cost
        
        print('\nFinal loss: ', loss.data.numpy())
        print('Final A:\n', A)
        A_norm = A.pow(2).sum().data.numpy()
        print('Final A squared norm: ', A_norm)
        assert np.allclose(0.5 * A_norm, args.C)
        
        
    if args.test == 'frw':
        test_frw()
    elif args.test == 'l2':
        test_l2()
    
        
        

    
    