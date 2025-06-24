import argparse
import os
import time
import common_utils
import train_bc
import train_rl
import rich.traceback
import torch
import numpy as np
from env.robosuite_wrapper import PixelRobosuite
from common_utils import ibrl_utils as utils

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