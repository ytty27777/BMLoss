#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.distributions.beta import Beta
from torch import Tensor

eps=1e-8
resolution = 100

def linspace(start: Tensor, stop: Tensor, num: int):
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps*(stop - start)[None]
    return out

def mu_std_to_alp_beta(mu,std):
    v = ( mu*(1-mu) )/ (std) -1
    alp = mu * v
    beta = ( 1-mu ) * v
    return alp, beta

def beta_cdf(beta_dist, start, stop, npts):
    x = linspace(start, stop, npts).T
    return torch.trapz(beta_dist.log_prob(x).exp(),x)

def Belief_mismatch_loss(pred_alp, pred_bet, true_alp, true_bet, alpha_number=5):
    # init stage
    global eps
    global resolution
    data_len = len(pred_alp)
    device = pred_alp.device
    zeros_tansor = (torch.zeros(data_len)+eps).to(device)
    ones_tansor = (torch.ones(data_len)-eps).to(device)
    
    pred_dist = Beta(pred_alp, pred_bet)
    true_dist = Beta(true_alp, true_bet)
    x = linspace(zeros_tansor, ones_tansor, resolution).T.to(device)
    true_pdf = true_dist.log_prob(x).exp()
    max_in_true, max_idx_in_true = true_pdf.max(dim=1)
    
    x_max_in_true_pdf = x[torch.arange(x.size(0)), max_idx_in_true]
    x_st_to_max = linspace(zeros_tansor, x_max_in_true_pdf, resolution).T
    x_max_to_ed = linspace( x_max_in_true_pdf, ones_tansor, resolution).T
    pdf_head = true_dist.log_prob(x_st_to_max).exp() # pdf before max value
    pdf_tail = true_dist.log_prob(x_max_to_ed).exp() # pdf after max value
    # not take the first and last, but middle
    alpha_h = linspace(zeros_tansor, max_in_true, alpha_number+2).T
    bmr_sum = 0
    for i in range(1, alpha_number+1):
        alpha_like = alpha_h[:,i].repeat(resolution,1).T # alpha value
        _, min_pdf_st_to_max_idx = torch.abs(pdf_head-alpha_like).min(dim=1) # closest x index in pdf before max value
        _, min_pdf_max_to_ed_idx = torch.abs(pdf_tail-alpha_like).min(dim=1) # closest x index in pdf after max value
        x1 = x_st_to_max[torch.arange(x_st_to_max.size(0)), min_pdf_st_to_max_idx] # closest x in pdf before max value
        x2 = x_max_to_ed[torch.arange(x_max_to_ed.size(0)), min_pdf_max_to_ed_idx] # closest x in pdf after max value
        ture_region = beta_cdf(true_dist, x1, x2, resolution)
        pred_region = beta_cdf(pred_dist, x1, x2, resolution)
        bmr_sum += pred_region/ture_region
        
    bmr_sum = bmr_sum/alpha_number
    bmr_loss = torch.abs(bmr_sum-1)
    return bmr_loss.mean()


#%%
if __name__ == '__main__':
    pred_alp = torch.rand((64,1))*10+1
    pred_bet = torch.rand((64,1))*10+1
    true_alp = torch.rand((64,1))*10+1
    true_bet = torch.rand((64,1))*10+1
    
    BMLloss = Belief_mismatch_loss(pred_alp, pred_bet, true_alp, true_bet, alpha_number=5)
    print('BMLloss', BMLloss)
