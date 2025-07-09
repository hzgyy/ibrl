import numpy as np
from typing import Optional
from typing import Tuple
import time

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class pessorlAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: str, cfg: QAgentConfig,wms_path:str):
        self._num_batch_step = 1
        self.bc_start_steps = 0
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
        self.lambda_dual = torch.tensor([1.0], requires_grad=False).to("cuda:0")
        self.wms = EnsembleWorldModel(obs_shape[0],obs_shape[0]//3,action_dim)
        self.wm_opt = torch.optim.Adam(self.wms.parameters(),lr=1e-4)
        self.state_dim = obs_shape[0]//3
        self.wms.load_state_dict(torch.load(wms_path,weights_only=True))
        self.wms.eval()

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
        # start = time.time()
        metrics = {}
        B,A = batch.action["action"].shape
        S = batch.obs["state"].shape[-1]
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
        # cql_time = time.time()
        # print(f"cql take:{cql_time-start}!!!")
        # sample random states
        NS = 256
        pess_tau = 5.0
        state_mean = obs["state"].mean(dim=0)
        state_std = obs["state"].var(dim=0)
        rand_states = torch.normal(mean=state_mean.expand(NS,S),std = state_std.expand(NS,S)*10) #shape (Ns,s)
        # calculate uncertainty
        with torch.no_grad():
            rand_states_actions = self.get_action(obs={"state":rand_states},eval_mode=False,stddev=0.01,use_target=False,clip=None)  #shape (NS,A)
            rand_mus,_ = self.wms(rand_states,rand_states_actions)    #shape (E,NS,S)
        rand_states_vs = self.critic(rand_states,rand_states_actions)  #shape (NS,Nc)
        rand_states_v = rand_states_vs.mean(dim=1)   #shape (NS,1)
        # rand_states_var = rand_states_vs.var(dim=-1).detach()    #shape (NS,1)
        # uncertainty = rand_states_var/(rand_states_var.sum())   #shape (NS,1)
        # metrics["train/rand_states_v"] = rand_states_v.mean().cpu().item()
        rand_mus_mean = rand_mus.mean(dim=0,keepdim=True)    #shape (1,NS,S)
        vars = torch.norm(rand_mus-rand_mus_mean,p=2,dim=-1).mean(dim=0)    #shape (NS)
        assert vars.shape==(NS,),f'{vars.shape}'
        uncertainty = vars/(vars.sum())
        # calculate
        curr_actions = curr_dist.sample().detach()
        batch_states_v_mean = self.critic(obs["state"],curr_actions).mean()  #scalar
        # assert False,f'{uncertainty.shape,rand_states_vs.shape}'
        # pessorl loss
        pess_loss = torch.log((uncertainty*torch.exp(rand_states_v)).sum())-batch_states_v_mean-pess_tau
        #update lambda
        with torch.no_grad():
            self.lambda_dual += pess_loss* 0.01
            self.lambda_dual.clamp_(min=0.0)
        metrics["train/lambda"] = self.lambda_dual
        # pess_time = time.time()
        # print(f"pess take:{pess_time-cql_time}!!!")
        #calculate loss
        td_loss = nn.functional.mse_loss(q_preds,q_target.expand(-1,q_preds.shape[-1])) #shape scalar
        cql_loss = self.cfg.min_q_weight*(torch.logsumexp(q_cat,dim=1).mean() - q_preds.mean()) - self.cfg.target_q_gap #shape scalar
        critic_loss = td_loss + self.cfg.cql_weight * cql_loss + self.lambda_dual*pess_loss
        # critic_loss = td_loss + self.cfg.cql_weight * cql_loss
        metrics["train/td_loss"] = td_loss.detach().cpu().item()
        metrics["train/critic_loss"] = critic_loss.detach().cpu().item()
        metrics["train/pess_loss"] = pess_loss.detach().cpu().item()
        # print(f'pess loss:{pess_loss,self.lambda_dual}')
        return critic_loss, metrics

    def warmup(self,batch):
        obs:torch.Tensor = batch.obs["state"]
        action: torch.Tensor = batch.action["action"]
        n_obs:torch.Tensor = batch.next_obs["state"]
        delta_mus,stds = self.wms.forward_k(obs,action)   #shape[2,B,state_dim]
        expand_obs = obs[:,0:self.state_dim].unsqueeze(0).expand(2,-1,-1)   #shape (2,B,state_dim)
        dist = torch.distributions.Normal(expand_obs+delta_mus,stds)
        log_prob = dist.log_prob(n_obs[:,0:self.state_dim].unsqueeze(0).expand(2,-1,-1))    #shape (2,B,state_dim)
        loss = -log_prob.sum(dim=-1).mean()
        self.wm_opt.zero_grad()
        loss.backward()
        self.wm_opt.step()
        return loss.detach().cpu().item()


    def _get_actions_and_log_prob(self,dist,sample_shape=torch.Size()):
        actions = dist.rsample(sample_shape=sample_shape)
        log_prob = dist.log_prob(actions).sum(-1)
        return actions,log_prob


class EnsembleWorldModel(nn.Module):
    def __init__(self, instate_dim, state_dim, action_dim, ensemble_size=5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.state_dim = state_dim

        hidden_dim = 400
        self.trunk = nn.Sequential(
            nn.Linear(instate_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        ).to("cuda:0")

        # 多个输出 head，分别预测 (mu, log_std)，每个 head 输出维度 = 2 * state_dim
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, 2 * state_dim)
            for _ in range(ensemble_size)
        ]).to("cuda:0")

    def forward(self, state, action):
        """
        Input:
            state: [B, S]
            action: [B, A]
        Output:
            mus:  [E, B, state_dim]
            stds: [E, B, state_dim]
        """
        x = torch.cat([state, action], dim=-1)  # [B, S+A]
        features = self.trunk(x)                # [B, hidden]

        mus, stds = [], []
        for head in self.heads:
            out = head(features)               # [B, 2 * state_dim]
            mu, log_std = torch.chunk(out, 2, dim=-1)
            log_std = torch.clamp(log_std, -5.0, 2.0)
            std = torch.exp(log_std)
            mus.append(mu)
            stds.append(std)

        # Stack into [E, B, state_dim]
        mus = torch.stack(mus, dim=0)
        stds = torch.stack(stds, dim=0)

        return mus, stds
    
    def forward_k(self,state,action):
        mus,stds = self.forward(state,action)
        indices = np.random.choice(5, 2, replace=False)
        selected_mus = mus[indices,:,:]
        selected_stds = stds[indices,:,:]
        return selected_mus,selected_stds