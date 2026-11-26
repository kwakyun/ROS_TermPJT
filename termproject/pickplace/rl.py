"""RLPD-style stabilized SAC components for Pick-and-Place.

Provides:
- LayerNormCritic / LayerNormSACPolicy: prevents OOD Q-value explosion on mixed offline+online batches.
- DemoMixReplayBuffer: keeps offline demos separate and mixes them at an annealed ratio.
- StableSAC: clamps ent_coef and clips actor/critic gradients without altering SAC.__init__ signature
  so saved policy zips remain loadable by standard SAC.load() in evaluate.py and policy_runner.py.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import ContinuousCritic, SACPolicy


class LayerNormCritic(ContinuousCritic):
    """ContinuousCritic with LayerNorm inserted after each hidden linear layer."""

    def __init__(
        self,
        observation_space,
        action_space,
        net_arch: list[int],
        features_extractor,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            observation_space,
            action_space,
            net_arch,
            features_extractor,
            features_dim,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )
        action_dim = int(np.prod(action_space.shape))
        self.q_networks = []
        for idx in range(self.n_critics):
            q_net_list = create_mlp(
                features_dim + action_dim,
                1,
                net_arch,
                activation_fn,
                post_linear_modules=[nn.LayerNorm],
            )
            q_net = nn.Sequential(*q_net_list)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)


class LayerNormSACPolicy(SACPolicy):
    """SACPolicy using LayerNormCritic for offline-to-online stability."""

    def make_critic(self, features_extractor=None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return LayerNormCritic(**critic_kwargs).to(self.device)


class DemoMixReplayBuffer(ReplayBuffer):
    """ReplayBuffer that mixes a controlled ratio of offline demonstration transitions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.demo_obs: Optional[np.ndarray] = None
        self.demo_next_obs: Optional[np.ndarray] = None
        self.demo_actions: Optional[np.ndarray] = None
        self.demo_rewards: Optional[np.ndarray] = None
        self.demo_dones: Optional[np.ndarray] = None
        self.demo_ratio: float = 0.0

    def set_demos(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        demo_ratio: float = 0.25,
    ) -> None:
        self.demo_obs = np.asarray(obs, dtype=np.float32)
        self.demo_next_obs = np.asarray(next_obs, dtype=np.float32)
        self.demo_actions = np.asarray(actions, dtype=np.float32)
        self.demo_rewards = np.asarray(rewards, dtype=np.float32)
        self.demo_dones = np.asarray(dones, dtype=np.float32)
        self.set_demo_ratio(demo_ratio)

    def set_demo_ratio(self, ratio: float) -> None:
        self.demo_ratio = float(np.clip(ratio, 0.0, 0.9))

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        if self.demo_obs is None or len(self.demo_obs) == 0 or self.demo_ratio <= 0.0:
            return super().sample(batch_size, env=env)
        n_demo = min(batch_size - 1, max(1, int(round(batch_size * self.demo_ratio))))
        n_online = batch_size - n_demo
        online = super().sample(n_online, env=env)
        idx = np.random.randint(0, len(self.demo_obs), size=n_demo)
        d_obs = self.to_torch(self.demo_obs[idx])
        d_act = self.to_torch(self.demo_actions[idx])
        d_next = self.to_torch(self.demo_next_obs[idx])
        d_done = self.to_torch(self.demo_dones[idx].reshape(-1, 1))
        d_rew = self.to_torch(self.demo_rewards[idx].reshape(-1, 1))
        return ReplayBufferSamples(
            observations=th.cat([online.observations, d_obs], dim=0),
            actions=th.cat([online.actions, d_act], dim=0),
            next_observations=th.cat([online.next_observations, d_next], dim=0),
            dones=th.cat([online.dones, d_done], dim=0),
            rewards=th.cat([online.rewards, d_rew], dim=0),
            discounts=None,
        )


class StableSAC(SAC):
    """SAC with entropy coefficient clamping and actor/critic gradient clipping."""

    min_ent_coef: float = 0.005
    max_ent_coef: float = 0.20
    max_grad_norm: float = 10.0

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            if self.use_sde:
                self.actor.reset_noise()

            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                assert isinstance(self.target_entropy, float)
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                with th.no_grad():
                    self.log_ent_coef.data.clamp_(
                        math.log(self.min_ent_coef),
                        math.log(self.max_ent_coef),
                    )

            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            assert isinstance(critic_loss, th.Tensor)
            critic_losses.append(critic_loss.item())

            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            th.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.critic.optimizer.step()

            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            th.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor.optimizer.step()

            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
