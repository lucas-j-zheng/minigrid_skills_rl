from typing import Any, Optional, Sequence
import torch
import numpy as np
import torch.nn.functional as F
from pfrl.agents import DoubleDQN
from pfrl.utils.batch_states import batch_states
from collections import deque
import random


class MaskedReplayBuffer:
    """Replay buffer that stores action masks with transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def append(self, state, action, reward, next_state, is_state_terminal,
                mask=None, next_mask=None, **kwargs):
         self.buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'is_state_terminal': is_state_terminal,
            'mask': mask if mask is not None else None,
            'next_mask': next_mask if next_mask is not None else None,
        })

    def sample(self, n):
        assert len(self.buffer) >= n
        return random.sample(self.buffer, n)

    def __len__(self):
        return len(self.buffer)

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = value


class MaskedDoubleDQN(DoubleDQN):
    """DoubleDQN with action masking - masks applied during action selection and backprop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_last_mask = None
        self.batch_next_mask = None

    def act(self, obs, mask=None):
        """Select action with optional masking."""
        return self.batch_act([obs], [mask])[0]

    def batch_act(
        self,
        batch_obs: Sequence[Any],
        batch_mask: Optional[Sequence[Any]] = None
    ) -> Sequence[Any]:
        """Select actions for batch with optional masking."""
        with torch.no_grad():
            batch_xs = batch_states(batch_obs, self.device, self.phi)
            batch_qout = self.model(batch_xs)

            if batch_mask is not None:
                mask_tensor = torch.tensor(
                    np.array(batch_mask), dtype=torch.bool, device=self.device
                )
                q_values = batch_qout.q_values.clone()
                q_values[~mask_tensor] = float('-inf')
                batch_action = q_values.argmax(dim=1).cpu().numpy()

                if self.training:
                    batch_action = [
                        self.explorer.select_action(
                            self.t,
                            greedy_action_func=lambda idx=i: batch_action[idx],
                            action_value=None
                        )
                        for i in range(len(batch_obs))
                    ]

                    # Validate actions against mask
                    for i, action in enumerate(batch_action):
                        valid_actions = np.where(batch_mask[i])[0]
                        if len(valid_actions) == 0:
                            raise ValueError(
                                f"No valid actions available at timestep {self.t}, "
                                f"observation {i}, mask: {batch_mask[i]}"
                            )
                        if action not in valid_actions:
                            # Action was masked or invalid - sample randomly from valid actions
                            batch_action[i] = np.random.choice(valid_actions)
                    # Keep as list when training to match base class behavior
                else:
                    # Convert to list when not training to match expected return type
                    batch_action = batch_action.tolist()
            else:
                if self.training:
                    batch_action = self.explorer.select_action(
                        self.t,
                        greedy_action_func=batch_qout.greedy_actions.cpu().numpy,
                        action_value=batch_qout
                    )
                else:
                    batch_action = batch_qout.greedy_actions.cpu().numpy().tolist()

        if self.training:
            self.batch_last_obs = list(batch_obs)
            self.batch_last_action = list(batch_action)
            self.batch_last_mask = list(batch_mask) if batch_mask is not None else None

        return batch_action

    def _batch_select_greedy_action(self, batch_obs, batch_mask=None):
        with torch.no_grad():
            batch_xs = batch_states(batch_obs, self.device, self.phi)
            batch_qout = self.model(batch_xs)

            if batch_mask is not None:
                mask_tensor = torch.tensor(
                    np.array(batch_mask), dtype=torch.bool, device=self.device
                )
                q_values = batch_qout.q_values.clone()
                q_values[~mask_tensor] = float('-inf')
                batch_action = q_values.argmax(dim=1).cpu().numpy()
            else:
                batch_action = batch_qout.greedy_actions.cpu().numpy()

        return batch_action

    def batch_observe(self, batch_obs, batch_reward, batch_done, batch_reset, batch_next_mask=None):
        self.batch_next_mask = list(batch_next_mask) if batch_next_mask is not None else None
        super().batch_observe(batch_obs, batch_reward, batch_done, batch_reset)

    def _batch_observe_train(self, batch_obs, batch_reward, batch_done, batch_reset):
        """Override the training observation to support masked replay buffer."""
        if isinstance(self.replay_buffer, MaskedReplayBuffer):
            for i, _ in enumerate(batch_obs):
                self.t += 1
                self._cumulative_steps += 1
                # Update the target network
                if self.t % self.target_update_interval == 0:
                    self.sync_target_network()

                if self.batch_last_obs[i] is not None:
                    assert self.batch_last_action[i] is not None
                    mask = (self.batch_last_mask[i]
                            if self.batch_last_mask is not None else None)
                    next_mask = (self.batch_next_mask[i]
                                 if self.batch_next_mask is not None else None)

                    self.replay_buffer.append(
                        state=self.batch_last_obs[i],
                        action=self.batch_last_action[i],
                        reward=batch_reward[i],
                        next_state=batch_obs[i],
                        is_state_terminal=batch_done[i],
                        mask=mask,
                        next_mask=next_mask,
                    )

                    if batch_reset[i] or batch_done[i]:
                        self.batch_last_obs[i] = None
                        self.batch_last_action[i] = None

                update_condition = (
                    len(self.replay_buffer) >= self.replay_start_size and
                    self.t % self.update_interval == 0
                )
                if update_condition:
                    self._masked_update()
        else:
            # Fall back to parent implementation for non-masked replay buffer
            super()._batch_observe_train(
                batch_obs, batch_reward, batch_done, batch_reset
            )
            
    def _masked_update(self):
        """DQN update with masked target Q-values."""
        transitions = self.replay_buffer.sample(self.minibatch_size)

        batch_state = [t['state'] for t in transitions]
        batch_action = np.array([t['action'] for t in transitions])
        batch_reward = np.array([t['reward'] for t in transitions])
        batch_next_state = [t['next_state'] for t in transitions]
        batch_terminal = np.array([t['is_state_terminal'] for t in transitions])
        batch_next_mask = [t['next_mask'] for t in transitions]

        batch_state_tensor = batch_states(batch_state, self.device, self.phi)
        batch_next_state_tensor = batch_states(batch_next_state, self.device, self.phi)
        batch_action_tensor = torch.tensor(batch_action, dtype=torch.int64, device=self.device)
        batch_reward_tensor = torch.tensor(batch_reward, dtype=torch.float32, device=self.device)
        batch_terminal_tensor = torch.tensor(batch_terminal, dtype=torch.float32, device=self.device)

        current_qout = self.model(batch_state_tensor)
        current_q = current_qout.q_values.gather(1, batch_action_tensor.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_qout_online = self.model(batch_next_state_tensor)
            next_qout_target = self.target_model(batch_next_state_tensor)

            next_q_online = next_qout_online.q_values.clone()
            next_q_target = next_qout_target.q_values.clone()

            if any(mask is not None for mask in batch_next_mask):
                for i, mask in enumerate(batch_next_mask):
                    if mask is not None:
                        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device)
                        next_q_online[i][~mask_tensor] = float('-inf')
                        next_q_target[i][~mask_tensor] = float('-inf')

            next_action = next_q_online.argmax(dim=1)
            next_q = next_q_target.gather(1, next_action.unsqueeze(1)).squeeze(1)
            target_q = batch_reward_tensor + self.gamma * next_q * (1 - batch_terminal_tensor)

        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        # Target network syncing is handled in _batch_observe_train, not here
        # This avoids redundant syncing on the same timestep

        # These attributes are defined in the parent DQN class
        if hasattr(self, 'q_record'):
            self.q_record.extend(current_q.detach().cpu().numpy().ravel())  # type: ignore
        if hasattr(self, 'loss_record'):
            self.loss_record.append(loss.item())  # type: ignore


def make_masked_dqn_agent(q_func, num_actions, phi, lr=2.5e-4, gamma=0.99, buffer_size=50_000,
                          replay_start_size=1000, update_interval=4, target_update_interval=1000,
                          start_epsilon=1.0, final_epsilon=0.01, final_exploration_steps=10_000,
                          batch_size=32, gpu=-1):
    from pfrl import explorers
    optimizer = torch.optim.Adam(q_func.parameters(), lr=lr)
    replay_buffer = MaskedReplayBuffer(capacity=buffer_size)
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=final_epsilon,
        decay_steps=final_exploration_steps,
        random_action_func=lambda: np.random.randint(num_actions),
    )

    agent = MaskedDoubleDQN(
        q_func, optimizer, replay_buffer,
        gamma=gamma, explorer=explorer,
        replay_start_size=replay_start_size,
        update_interval=update_interval,
        target_update_interval=target_update_interval,
        phi=phi, minibatch_size=batch_size, gpu=gpu,
    )
    return agent
