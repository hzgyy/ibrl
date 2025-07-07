# import argparse
# import os
# import time
# import common_utils
# import train_bc
# import train_rl
import rich.traceback
import torch
import numpy as np
from env.robosuite_wrapper import PixelRobosuite
from common_utils import ibrl_utils as utils
import torch.nn.functional as F

def eval_kl(agent1,agent2,obs_batch,is_dict=0,stddev=0.1):
    if is_dict:
        pi1 = agent1.actor.torch_forward(obs_batch,stddev)
        pi2 = agent2.actor.torch_forward(obs_batch,stddev)
        #action1, log_pi1 = agent1.actor.act_with_prob(obs_batch,stddev)
        #log_pi2 = agent2.actor.logp(obs_batch,action1,stddev)
    else:
        pi1 = agent1.actor.torch_forward({'state':obs_batch},stddev)
        pi2 = agent2.actor.torch_forward({'state':obs_batch},stddev)
        # action1, log_pi1 = agent1.actor.act_with_prob({'state':obs_batch},stddev)
        # log_pi2 = agent2.actor.logp({'state':obs_batch},action1,stddev)
    #kl = (torch.exp(log_pi1)*(log_pi1-log_pi2)).mean()
    kl = torch.distributions.kl_divergence(pi1, pi2).mean()
    return kl.item()

def q_kl_divergence(off_qs, off_ori_qs, tau=1.0):
    """
    Compute D_KL(softmax(Q_ori / tau) || softmax(Q / tau)) per sample in batch.
    
    Args:
        off_qs: Tensor of shape [B, A] - current Q values.
        off_ori_qs: Tensor of shape [B, A] - reference Q values.
        tau: Temperature for softmax.
        
    Returns:
        Scalar KL divergence (averaged over batch).
    """
    # Compute soft policies
    logp = F.log_softmax(off_ori_qs / tau, dim=1)     # log π_old(a|s)
    logq = F.log_softmax(off_qs / tau, dim=1)         # log π_new(a|s)
    p = logp.exp()                                    # π_old(a|s)

    # KL divergence: sum_a p(a) * (logp(a) - logq(a))
    kl = torch.sum(p * (logp - logq), dim=1)          # shape: [B]
    return kl.mean()                                  # scalar

