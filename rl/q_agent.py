from typing import Optional
from dataclasses import dataclass, field
import copy
from contextlib import contextmanager
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import common_utils
from common_utils import ibrl_utils as utils
from networks.encoder import VitEncoder, VitEncoderConfig
from networks.encoder import ResNetEncoder, ResNetEncoderConfig, DrQEncoder
from networks.encoder import ResNet96Encoder, ResNet96EncoderConfig
from rl.critic import Critic, CriticConfig
from rl.actor import Actor, ActorConfig, SpatialEmb
from rl.actor import FcActor, FcActorConfig
from rl.critic import MultiFcQ, MultiFcQConfig


@dataclass
class QAgentConfig:
    device: str = "cuda"
    lr: float = 1e-4
    critic_target_tau: float = 0.01
    stddev_clip: float = 0.3
    # encoder
    use_prop: int = 0
    enc_type: str = "vit"
    vit: VitEncoderConfig = field(default_factory=lambda: VitEncoderConfig())
    resnet: ResNetEncoderConfig = field(default_factory=lambda: ResNetEncoderConfig())
    resnet96: ResNet96EncoderConfig = field(default_factory=lambda: ResNet96EncoderConfig())
    # critic & actor
    critic: CriticConfig = field(default_factory=lambda: CriticConfig())
    actor: ActorConfig = field(default_factory=lambda: ActorConfig())
    state_critic: MultiFcQConfig = field(default_factory=lambda: MultiFcQConfig())
    state_actor: FcActorConfig = field(default_factory=lambda: FcActorConfig())
    # algo
    act_method: str = "rl"  # "rl/ibrl/ibrl_soft"
    bootstrap_method: str = ""  # "rl/ibrl/ibrl_soft"
    # ibrl
    ibrl_eps_greedy: float = 1
    soft_ibrl_beta: float = 10
    # bc loss regularization
    bc_loss_coef: float = 0.1
    bc_loss_dynamic: int = 0  # dynamically scale bc loss weight
    #awac
    awac_beta:float = 1.0
    #sac like
    entropy_weight: float = 1.0
    #cql
    num_random_actions: int = 4
    cql_weight: float = 1.0
    min_q_weight: float = 1.0
    target_q_gap: float = 5.0
    #dgn
    shutoff_threshold: float = 0.5
    shutoff_epochs: int = 10
    dgn_update_freq: int= 1000
    dgn_batch_size: int = 128
    dgn_epoch_per_update: int = 2
    #vib
    vib_beta: float = 0.1
    repr_dim: int = 128
    def __post_init__(self):
        if self.bootstrap_method == "":
            self.bootstrap_method = self.act_method


class QAgent(nn.Module,ABC):
    def __init__(
        self, use_state, obs_shape, prop_shape, action_dim, rl_camera: list[str], cfg: QAgentConfig
    ):
        super().__init__()
        self.use_state = use_state
        self.rl_camera = rl_camera
        self.cfg = cfg
        assert len(self.rl_camera)==2,f'num of rl camera wrong'
        assert "agentview" in self.rl_camera and "robot0_eye_in_hand" in self.rl_camera,f'rl camera wrong:{self.rl_camera}'
        self.rl_camera = ["robot0_eye_in_hand","agentview"]
        if use_state:
            self.critic = MultiFcQ(obs_shape, action_dim, cfg.state_critic)
            self.actor = FcActor(obs_shape, action_dim, cfg.state_actor)
        else:
            self.agentview_encoder = self._build_encoders(obs_shape)
            self.handview_encoder = self._build_encoders(obs_shape)
            repr_dim = self.agentview_encoder.repr_dim
            patch_repr_dim = self.agentview_encoder.patch_repr_dim
            print("encoder output dim: ", repr_dim)
            print("patch output dim: ", patch_repr_dim)

            assert len(prop_shape) == 1
            prop_dim = prop_shape[0] if cfg.use_prop else 0
            policy_repr_dim = cfg.repr_dim*2
            self.agentview_compressor = SpatialEmb(num_patch=repr_dim // patch_repr_dim,
                                     patch_dim=patch_repr_dim,
                                     prop_dim=0,
                                     proj_dim=2*cfg.repr_dim,
                                     dropout=cfg.actor.dropout) #output B
            self.handview_compressor = SpatialEmb(num_patch=repr_dim // patch_repr_dim,
                                     patch_dim=patch_repr_dim,
                                     prop_dim=0,
                                     proj_dim=cfg.repr_dim,
                                     dropout=cfg.actor.dropout) #output B
            # create critics & actor
            self.critic = Critic(
                repr_dim=policy_repr_dim,
                patch_repr_dim=patch_repr_dim,
                prop_dim=prop_dim,
                action_dim=action_dim,
                cfg=self.cfg.critic,
            )
            self.actor = Actor(policy_repr_dim, patch_repr_dim, prop_dim, action_dim, cfg.actor)

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)

        if not self.use_state:
            print(common_utils.wrap_ruler(f"encoder weights"))
            common_utils.count_parameters(self.handview_encoder)
            common_utils.count_parameters(self.handview_compressor)
            common_utils.count_parameters(self.agentview_encoder)
            common_utils.count_parameters(self.agentview_compressor)
            print(self.agentview_compressor)
            print(self.handview_compressor)
            print(self.handview_encoder)
            print(self.agentview_encoder)

        print(common_utils.wrap_ruler("critic weights"))
        print(self.critic)
        common_utils.count_parameters(self.critic)

        print(common_utils.wrap_ruler("actor weights"))
        print(self.actor)
        common_utils.count_parameters(self.actor)

        # optimizers
        if not self.use_state:
            self.agentview_encoder_opt = torch.optim.Adam(list(self.agentview_encoder.parameters())+list(self.agentview_compressor.parameters()),
                                                 lr=self.cfg.lr)
            self.handview_encoder_opt = torch.optim.Adam(list(self.handview_encoder.parameters())+list(self.handview_compressor.parameters()),
                                                 lr=self.cfg.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr)

        # data augmentation
        self.aug = common_utils.RandomShiftsAug(pad=4)

        self.bc_policies: list[nn.Module] = []
        # to log rl vs bc during evaluation
        self.stats: Optional[common_utils.MultiCounter] = None

        self.critic_target.train(False)
        self.train(True)
        self.to(self.cfg.device)

    def _build_encoders(self, obs_shape):
        if self.cfg.enc_type == "vit":
            return VitEncoder(obs_shape, self.cfg.vit).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet":
            return ResNetEncoder(obs_shape, self.cfg.resnet).to(self.cfg.device)
        elif self.cfg.enc_type == "resnet96":
            return ResNet96Encoder(obs_shape, self.cfg.resnet96).to(self.cfg.device)
        elif self.cfg.enc_type == "drq":
            return DrQEncoder(obs_shape).to(self.cfg.device)
        else:
            assert False, f"Unknown encoder type {self.cfg.enc_type}."

    def add_bc_policy(self, bc_policy):
        bc_policy.train(False)
        self.bc_policies.append(bc_policy)

    def set_stats(self, stats):
        self.stats = stats

    def train(self, training=True):
        self.training = training
        if not self.use_state:
            self.handview_encoder.train(training)
            self.handview_compressor.train(training)
            self.agentview_encoder.train(training)
            self.agentview_compressor.train(training)
        self.actor.train(training)
        self.critic.train(training)

        assert not self.critic_target.training
        for bc_policy in self.bc_policies:
            assert not bc_policy.training

    @contextmanager
    def override_act_method(self, override_method: str):
        original_method = self.cfg.act_method
        assert original_method != override_method

        self.cfg.act_method = override_method
        yield

        self.cfg.act_method = original_method
        return

    def _encode(self, obs: dict[str, torch.Tensor], augment: bool,need_kl:bool=False) -> torch.Tensor:
        """This function encodes the observation into feature tensor."""
        #encode handview
        hand_data = obs["robot0_eye_in_hand"].float()
        batch_size = hand_data.shape[0]
        if augment:
            hand_data = self.aug(hand_data)   #B,96,96
        hand_feat = self.handview_encoder.forward(hand_data, flatten=False)   #B,144,256
        hand_feat = self.handview_compressor.forward(hand_feat,None)   # B, cfg.repr_dim
        # synthesize distributions
        agent_data = obs["agentview"].float()
        if augment:
            agent_data = self.aug(agent_data)   #B,96,96
        agent_feat = self.agentview_encoder.forward(agent_data, flatten=False)   #B,144,256
        agent_feat = self.agentview_compressor.forward(agent_feat,None)   # B, cfg.repr_dim
        # print(agent_feat[:,:self.cfg.repr_dim])
        agent_feat_dis = torch.distributions.Normal(agent_feat[:,:self.cfg.repr_dim],
                                                    torch.exp(agent_feat[:,self.cfg.repr_dim:])) #B, distri
        sampled_agent_feat = agent_feat_dis.rsample()   #B,cfg.repr_dim
        assert hand_feat.shape == sampled_agent_feat.shape,f'two view shape wrong:{hand_feat.shape}vs{agent_feat.shape}'
        # if self.cfg.use_prop:
        #     prop_feat = obs["prop"]
        #     assert hand_feat.shape[0] == prop_feat.shape[0],f'prop and feature shape wrong:{hand_feat.shape}vs{prop_feat.shape}'
        #     feat_list = [hand_feat,sampled_agent_feat,prop_feat]
        # else:
        #     feat_list = [hand_feat,sampled_agent_feat]
        result = torch.cat([hand_feat,sampled_agent_feat],dim=-1)  #B,2*cfg.repr_dim
        assert result.shape == (batch_size,2*self.cfg.repr_dim),f'result shape wrong:{result.shape}'
        if need_kl:
            prior = torch.distributions.Normal(0, 1)
            kl = torch.distributions.kl.kl_divergence(agent_feat_dis,prior).sum(-1) #B,1
            assert kl.shape == (batch_size,),f'kl shape wrong:{kl.shape}'
            return result,kl
        else:
            return result

    def _maybe_unsqueeze_(self, obs):
        should_unsqueeze = False
        if self.use_state:
            if obs["state"].dim() == 1:
                should_unsqueeze = True
        else:
            for camera in self.rl_camera:
                if obs[camera].dim() == 3:
                    should_unsqueeze = True

        if should_unsqueeze:
            for k, v in obs.items():
                obs[k] = v.unsqueeze(0) #create a batch whose size is 1
        return should_unsqueeze

    def act(
        self, obs: dict[str, torch.Tensor], *, eval_mode=False, stddev=0.0, cpu=True
    ) -> torch.Tensor:
        """This function takes tensor and returns actions in tensor"""
        assert not self.training
        assert not self.actor.training
        unsqueezed = self._maybe_unsqueeze_(obs)

        if not self.use_state:
            assert "feat" not in obs
            obs["feat"] = self._encode(obs, augment=False)
        action = self.get_action(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=None,
            use_target=False,
            eps_greedy=self.cfg.ibrl_eps_greedy
        )

        if unsqueezed:
            action = action.squeeze(0)

        action = action.detach()
        if cpu:
            action = action.cpu()
        return action

    def get_action(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
        eps_greedy = 1.0
    ) -> torch.Tensor:
        return self._act_default(
            obs=obs,
            eval_mode=eval_mode,
            stddev=stddev,
            clip=clip,
            use_target=use_target,
            eps_greedy=eps_greedy,
        )
    
    def _act_default(
        self,
        *,
        obs: dict[str, torch.Tensor],
        eval_mode: bool,
        stddev: float,
        clip: Optional[float],
        use_target: bool,
        eps_greedy = 1.0
    ) -> torch.Tensor:
        actor = self.actor_target if use_target else self.actor
        dist = actor.forward(obs, stddev)
        if eval_mode:
            assert not self.training

        if eval_mode:
            action = dist.mean.detach()
        else:
            action = dist.sample(clip=clip)

        return action

    def compute_critic_loss(
        self,
        obs: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        kl: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        # obs: dict[str, torch.Tensor] = batch.obs
        # reply: dict[str, torch.Tensor] = batch.action
        # reward: torch.Tensor = batch.reward
        # discount: torch.Tensor = batch.bootstrap
        # next_obs: dict[str, torch.Tensor] = batch.next_obs
        #assert False,f"{obs.keys(),next_obs.keys()}"
        with torch.no_grad():
            next_action = self.get_action(
                    obs=next_obs,
                    eval_mode=False,
                    stddev=stddev,
                    clip=self.cfg.stddev_clip,
                    use_target=True,
                )
        if isinstance(self.critic_target, Critic):
            target_q1, target_q2 = self.critic_target.forward(
                next_obs["feat"], next_obs["prop"], next_action
            )
            target_q = torch.min(target_q1, target_q2)
        else:
            target_q = self.critic_target.forward_k(next_obs["state"], next_action).min(-1)[0]

        target_q = (reward + (discount * target_q)).detach()

        action = reply["action"]
        loss_fn = nn.functional.mse_loss
        if isinstance(self.critic, Critic):
            q1, q2 = self.critic.forward(obs["feat"], obs["prop"], action)
            critic_loss = loss_fn(q1, target_q) + loss_fn(q2, target_q) + self.cfg.vib_beta*kl.mean()
        else:
            # change to forward k， in order to update q nets on different batch
            #qs: torch.Tensor = self.critic.forward(obs["state"], action)
            qs: torch.Tensor = self.critic.forward_k(obs["state"], action)
            critic_loss = nn.functional.mse_loss(
                qs, target_q.unsqueeze(1).repeat(1, qs.size(1)), reduction="none"
            )
            critic_loss = critic_loss.sum(1).mean(0)
        
        metrics = {}
        metrics["train/critic_qt"] = target_q.mean().item()
        metrics["train/critic_loss"] = critic_loss.item()

        return critic_loss,metrics

    def update_critic(
        self,
        obs: dict[str, torch.Tensor],
        reply: dict[str, torch.Tensor],
        reward: torch.Tensor,
        discount: torch.Tensor,
        next_obs: dict[str, torch.Tensor],
        stddev: float,
        kl:torch.Tensor,
    ):
        critic_loss, metrics = self.compute_critic_loss(obs,reply,reward,discount,next_obs,stddev,kl)
        if not self.use_state:
            self.agentview_encoder_opt.zero_grad(set_to_none=True)
            self.handview_encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)

        critic_loss.backward(retain_graph=True)

        if not self.use_state:
            self.agentview_encoder_opt.step()
            self.handview_encoder_opt.step()
        self.critic_opt.step()
        return metrics

    #def compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
    def compute_actor_loss(self, obs: dict[str, torch.Tensor], stddev: float):
        #obs = batch.obs
        if not self.use_state:
            assert "feat" in obs, "safety check"

        action: torch.Tensor = self._act_default(
            obs=obs,
            eval_mode=False,
            stddev=stddev,
            clip=self.cfg.stddev_clip,
            use_target=False,
        )
        if isinstance(self.critic, Critic):
            q = torch.min(*self.critic.forward(obs["feat"], obs["prop"], action))
        else:
            q: torch.Tensor = self.critic(obs["state"], action).min(-1)[0]
        actor_loss = -q.mean()
        return actor_loss        

    def update_actor(
        self, 
        obs: dict[str, torch.Tensor], 
        #batch,
        stddev: float,
        bc_batch,
        ref_agent: "QAgent",
    ):
        metrics = {}
        actor_loss = self.compute_actor_loss(obs, stddev)
        metrics["train/actor_loss"] = actor_loss.detach().item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        return metrics

    def update(
        self,
        batch,
        stddev,
        update_actor,
        bc_batch=None,
        ref_agent: Optional["QAgent"] = None,
    ):
        obs: dict[str, torch.Tensor] = batch.obs
        reply: dict[str, torch.Tensor] = batch.action
        reward: torch.Tensor = batch.reward
        discount: torch.Tensor = batch.bootstrap
        next_obs: dict[str, torch.Tensor] = batch.next_obs

        if not self.use_state:
            obs["feat"],kl = self._encode(obs, augment=True,need_kl=True)
            with torch.no_grad():
                next_obs["feat"] = self._encode(next_obs, augment=True)
        
        metrics = {}
        metrics["data/batch_R"] = reward.mean().item()
        critic_metric = self.update_critic(obs,reply,reward,discount,next_obs,stddev,kl)
        utils.soft_update_params(self.critic, self.critic_target, self.cfg.critic_target_tau)
        metrics.update(critic_metric)

        if not update_actor:
            return metrics

        # NOTE: actor loss does not backprop into the encoder
        if not self.use_state:
            obs["feat"] = obs["feat"].detach()

        # if bc_batch is None:
        #     actor_metric = self.update_actor(obs, stddev)
        # else:
        #     assert ref_agent is not None
        #     actor_metric = self.update_actor_rft(obs, stddev, bc_batch, ref_agent)
        actor_metric = self.update_actor(obs,stddev,bc_batch,ref_agent)

        utils.soft_update_params(self.actor, self.actor_target, self.cfg.critic_target_tau)
        metrics.update(actor_metric)

        return metrics

    def pretrain_actor_with_bc(self, batch):
        """pretrain actor and encoder with bc"""
        loss = self._compute_actor_bc_loss(batch, backprop_encoder=True)

        if not self.use_state:
            self.encoder_opt.zero_grad(set_to_none=True)

        self.actor_opt.zero_grad(set_to_none=True)
        loss.backward()

        if not self.use_state:
            self.encoder_opt.step()
        self.actor_opt.step()

        return {"pretrain/loss": loss.item()}
