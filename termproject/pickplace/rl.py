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
from gymnasium import spaces
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import create_mlp
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.utils import explained_variance, polyak_update
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


class AnchoredPPO(PPO):
    """PPO with BC anchor loss, dampened negative advantages, and bounded action std."""

    adv_neg_scale: float = 0.25
    bc_anchor_start: float = 0.50
    bc_anchor_end: float = 0.10
    min_std: float = 0.03
    max_std: float = 0.10
    demo_obs_norm: Optional[th.Tensor] = None
    demo_actions: Optional[th.Tensor] = None

    def set_bc_anchor(
        self,
        obs_norm: th.Tensor,
        actions: th.Tensor,
        coef_start: float = 0.50,
        coef_end: float = 0.10,
    ) -> None:
        self.demo_obs_norm = obs_norm.detach().to(self.device)
        self.demo_actions = actions.detach().to(self.device)
        self.bc_anchor_start = float(coef_start)
        self.bc_anchor_end = float(coef_end)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, bc_losses = [], [], []
        clip_fractions = []
        bc_coef = self.bc_anchor_end + (self.bc_anchor_start - self.bc_anchor_end) * float(
            self._current_progress_remaining
        )
        actor_trainable = any(p.requires_grad for p in self.policy.action_net.parameters())

        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if self.adv_neg_scale < 1.0:
                    advantages = th.where(advantages > 0, advantages, self.adv_neg_scale * advantages)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                if actor_trainable and self.demo_obs_norm is not None and self.demo_actions is not None and bc_coef > 0.0:
                    d_idx = th.randint(0, len(self.demo_obs_norm), (self.batch_size,), device=self.device)
                    pred_demo = self.policy.get_distribution(self.demo_obs_norm[d_idx]).mode()
                    bc_loss = F.mse_loss(pred_demo, self.demo_actions[d_idx])
                    bc_losses.append(bc_loss.item())
                    loss = loss + bc_coef * bc_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if hasattr(self.policy, "log_std"):
                    with th.no_grad():
                        self.policy.log_std.data.clamp_(math.log(self.min_std), math.log(self.max_std))

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/bc_anchor_coef", bc_coef)
        if bc_losses:
            self.logger.record("train/bc_anchor_loss", np.mean(bc_losses))
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
