import numpy as np
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from rl.q_agent import QAgent,QAgentConfig
from rl.critic import Critic, CriticConfig

class ibrlAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
    
    def get_action(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
        eps_greedy = 1.0,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            assert not actor.training

        assert len(self.bc_policies) == 1
        bc_policy = self.bc_policies[0]
        bc_action = bc_policy.act(obs, cpu=False)

        rl_dist: utils.TruncatedNormal = actor(obs, stddev)
        if eval_mode:
            rl_action = rl_dist.mean
        else:
            rl_action = rl_dist.sample(clip)

        rl_bc_actions = torch.stack([rl_action, bc_action], dim=1)
        bsize, num_action, _ = rl_bc_actions.size()

        # get q(a)
        # feat -> [batch, num_patch, patch_dim] -> [batch, num_action(2), num_patch, patch_dim]
        flat_actions = rl_bc_actions.flatten(0, 1)
        if isinstance(self.critic_target, Critic):
            flat_qfeats = obs["feat"].unsqueeze(1).repeat(1, num_action, 1, 1).flatten(0, 1)
            flat_props = obs["prop"].unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            q1, q2 = self.critic_target.forward(flat_qfeats, flat_props, flat_actions)
            qa: torch.Tensor = torch.min(q1, q2).view(bsize, num_action)
        else:
            state = obs["state"]
            flat_state = state.unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            qa: torch.Tensor = self.critic_target.forward_k(flat_state, flat_actions)
            qa = qa.min(-1)[0].view(bsize, num_action)

        # best_action_idx: [batch]
        greedy_action_idx: torch.Tensor = qa.argmax(1)
        greedy_action = rl_bc_actions[range(bsize), greedy_action_idx]
        # actions: [batch, action_dim]

        if eval_mode or eps_greedy == 1:
            action = greedy_action
            selected_action_idx = greedy_action_idx
        else:
            eps = torch.rand((bsize, 1), device=qa.device)
            use_greedy = (eps < eps_greedy).float()
            rand_action_idx = torch.randint(0, num_action, (bsize,))
            rand_action = rl_bc_actions[range(bsize), rand_action_idx]
            assert rand_action.size() == greedy_action.size()
            action = rand_action * (1 - use_greedy) + greedy_action * use_greedy
            selected_action_idx = rand_action_idx * (1 - use_greedy) + greedy_action_idx * use_greedy

            if self.stats is not None:
                self.stats["actor/greedy"].append(use_greedy.sum(), bsize)

        if self.stats is not None:
            use_bc = (selected_action_idx >= 1).float()
            if use_target:
                use_bc = use_bc.mean().item()
                self.stats["actor/bootstrap_bc"].append(use_bc)
            else:
                use_bc = use_bc.sum().item()
                if eval_mode:
                    self.stats["actor/bc_eval"].append(use_bc, bsize)
                else:
                    assert bsize == 1, f"bsize should be 1, but got {bsize}"
                    rl_action_norm = rl_action.squeeze()[:6].norm().item()
                    bc_action_norm = bc_action.squeeze()[:6].norm().item()
                    self.stats["actor/anorm_rl"].append(rl_action_norm)
                    self.stats["actor/anorm_bc"].append(bc_action_norm)
                    self.stats["actor/bc_train"].append(use_bc, bsize)
                    # self.stats["actor/greedy_index"].append(greedy_action_idx.mean())
                    self.stats["actor/greedy_index"].append(greedy_action_idx)

        return action
    

class ibrlsoftAgent(QAgent):
    def __init__(self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig):
        super().__init__(use_state, obs_shape, prop_shape, action_dim, rl_camera, cfg)
    
    def get_action(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
        eps_greedy = 1.0,
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        if eval_mode:
            assert not actor.training

        assert len(self.bc_policies) == 1
        bc_policy = self.bc_policies[0]
        bc_action = bc_policy.act(obs, cpu=False)

        rl_dist: utils.TruncatedNormal = actor(obs, stddev)
        if eval_mode:
            rl_action = rl_dist.mean
        else:
            rl_action = rl_dist.sample(clip)

        rl_bc_actions = torch.stack([rl_action, bc_action], dim=1)

        # cat along the num action dim
        # actions: [bsize, n_rl_actions + n_bc_actions * n_bc, action_dim]
        bsize, num_action, _ = rl_bc_actions.size()

        flat_actions = rl_bc_actions.flatten(0, 1)
        if isinstance(self.critic_target, Critic):
            flat_qfeats = obs["feat"].unsqueeze(1).repeat(1, num_action, 1, 1).flatten(0, 1)
            flat_props = obs["prop"].unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            q1, q2 = self.critic_target.forward(flat_qfeats, flat_props, flat_actions)
            qa: torch.Tensor = torch.min(q1, q2).view(bsize, num_action)
        else:
            state = obs["state"]
            flat_state = state.unsqueeze(1).repeat(1, num_action, 1).flatten(0, 1)
            qa: torch.Tensor = self.critic_target.forward_k(flat_state, flat_actions)
            qa = qa.min(-1)[0].view(bsize, num_action)

        # decide which action to take
        p_center = torch.nn.functional.softmax(qa * self.cfg.soft_ibrl_beta, dim=1)
        center_idx = p_center.multinomial(1)
        if (not use_target) and self.stats is not None and (not eval_mode):
            assert bsize == 1,f'{bsize}'
            self.stats["actor/p_max"].append(p_center.max().item())

        # center_idx: [batchsize, 1]
        action = rl_action * (1 - center_idx) + bc_action * center_idx

        if self.stats is not None:
            use_bc = center_idx.sum().item()
            if use_target:
                # must be called from update_critic
                self.stats["actor/bootstrap_bc"].append(use_bc, bsize)
            else:
                if eval_mode:
                    self.stats["actor/bc_eval"].append(use_bc, bsize)
                    #calculate the difference in rl and bc policy
                    act_diff = (rl_action-bc_action).unsqueeze(0)
                    act_diff = torch.linalg.vector_norm(act_diff).item()
                    self.stats["actor/diff_act_eval"].append(act_diff,bsize)
                else:
                    assert bsize == 1
                    bc_logp = rl_dist.log_prob(bc_action).sum(-1).item()
                    self.stats["actor/bc_logp_rl"].append(bc_logp,bsize)
                    self.stats["actor/bc_act"].append(use_bc, bsize)
                    self.stats["actor/qrl-qbc"].append((qa[0][0] - qa[0][1]).item())
        return action