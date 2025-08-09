import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class cqlAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig):
        self._num_batch_step = 1
        self.bc_start_steps = 0
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)

    def compute_actor_loss(self, batch, stddev: float):
        obs = batch.obs
        #get action
        dist = self.actor.torch_forward(obs,stddev)
        action,log_prob = self._get_actions_and_log_prob(dist=dist)
        #get predicted q values for all state,action pairs
        if isinstance(self.critic, Critic):
            pred_qs = torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
        else:
            pred_qs = self.critic.forward(obs["state"], action).min(-1)[0]
        #use bc if we are in the beginning of training
        baseline = dist.log_prob(batch.action["action"]) if\
            self._num_batch_step < self.bc_start_steps else pred_qs
        policy_loss = (self.cfg.entropy_weight*log_prob-baseline).mean()
        return policy_loss
    
    def compute_critic_loss(
        self,
        batch,
        stddev: float,
    ) -> Tuple[torch.Tensor, dict]:
        metrics = {}
        B,A = batch.action["action"].shape
        N = self.cfg.num_random_actions
        obs = batch.obs
        action = batch.action
        next_obs = batch.next_obs
        reward = batch.reward
        discount = batch.bootstrap
        # get predicted q-values from taken actions
        if isinstance(self.critic, Critic):
            q_preds = self.critic.forward(obs["feat"], obs["prop"], action["action"])
        else:
            q_preds = self.critic.forward(obs["state"], action["action"]) #shape (B,Nc)

        #sample actions at the current and next steps
        curr_dist = self.actor.torch_forward(obs,stddev)
        next_dist = self.actor.torch_forward(next_obs,stddev)
        next_actions,next_log_prob = self._get_actions_and_log_prob(dist=next_dist)

        with torch.no_grad():
            if isinstance(self.critic, Critic):
                target_qs = torch.min(*self.critic_target.forward(next_obs["feat"], next_obs["prop"], next_actions))
            else:
                target_qs = self.critic_target(next_obs["state"], next_actions).min(-1)[0]
            target_qs = target_qs-self.cfg.entropy_weight * next_log_prob
            # calculate q target values
            q_target = (reward + discount*target_qs).unsqueeze(-1) #shape(B,1)
        metrics["train/critic_qt"] = q_target.mean().item()
        #calculate cql stuff
        cql_random_actions = torch.FloatTensor(N*B,A).uniform_(-1.,1.).to("cuda:0")
        cql_random_log_prob = np.log(0.5**A)
        cql_curr_actions, cql_curr_log_prob = self._get_actions_and_log_prob(dist=curr_dist, sample_shape=(N,))     # shape (N, B, A) and (N, B)
        cql_next_actions, cql_next_log_prob = self._get_actions_and_log_prob(dist=next_dist, sample_shape=(N,))     # shape (N, B, A) and (N, B)
        cql_curr_actions = cql_curr_actions.permute(1,0,2).reshape(N*B,A).detach()   #shape (B,N,A) -> (BN,A)
        cql_next_actions = cql_next_actions.permute(1,0,2).reshape(N*B,A).detach()   #shape (B,N,A) -> (BN,A)
        cql_curr_log_prob = cql_curr_log_prob.permute(1, 0).unsqueeze(-1).detach()                                # shape (B, N,1)
        cql_next_log_prob = cql_next_log_prob.permute(1, 0).unsqueeze(-1).detach()                                # shape (B, N,1)
        expand_obs = obs["state"].repeat_interleave(N,dim=0) #shape (B*N,N_obs)
        
        q_rand = self.critic(expand_obs,cql_random_actions).reshape(B,N,-1) #shape (B,N,Nc)
        q_curr = self.critic(expand_obs,cql_curr_actions).reshape(B,N,-1)   #shape (B,N,Nc)
        q_next = self.critic(expand_obs,cql_next_actions).reshape(B,N,-1)   #shape (B,N,Nc)

        q_cat = torch.cat([
                q_rand - cql_random_log_prob,
                q_next - cql_next_log_prob,
                q_curr - cql_curr_log_prob,
            ], dim=1)           # shape (B, 3 * N,Nc)
        assert True, f'{q_preds.shape,q_target.shape,torch.logsumexp(q_cat,dim=1).shape}'
        #calculate loss
        td_loss = nn.functional.mse_loss(q_preds,q_target.expand(-1,q_preds.shape[-1])) #shape scalar
        cql_loss = self.cfg.min_q_weight*(torch.logsumexp(q_cat,dim=1).mean() - q_preds.mean()) - self.cfg.target_q_gap #shape scalar
        critic_loss = td_loss + self.cfg.cql_weight * cql_loss
        metrics["train/td_loss"] = td_loss.detach().cpu().item()
        metrics["train/critic_loss"] = critic_loss.detach().cpu().item()
        return critic_loss, metrics

    def _get_actions_and_log_prob(self,dist,sample_shape=torch.Size()):
        actions = dist.rsample(sample_shape=sample_shape)
        log_prob = dist.log_prob(actions).sum(-1)
        return actions,log_prob
    
    # def _get_qs_from_actions(self,obs, actions):
    #     """
    #     Helper function for grabbing Q values given a single state and multiple (N) sampled actions.
    #     Returns:
    #         tensor: (B, N) corresponding Q values
    #     """
    #     # Get the number of sampled actions
    #     # N, B, _ = actions.shape

    #     # Repeat obs and goals in the batch dimension
    #     # obs_dict_stacked = ObsUtils.repeat_and_stack_observation(obs_dict, N)
    #     # goal_dict_stacked = ObsUtils.repeat_and_stack_observation(goal_dict, N)
    #     # obs_stacked = obs.repeat_interleave(N,dim=0)

    #     # Pass the obs and (flattened) actions through to get the Q values
    #     qs = self.critic(obs,actions)

    #     # Unflatten output
    #     qs = qs.reshape(B, N, qs.shape[-1])

    #     return qs

def print_cuda_memory(tag=""):
    print(f"[{tag}] CUDA allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB | "
          f"reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")