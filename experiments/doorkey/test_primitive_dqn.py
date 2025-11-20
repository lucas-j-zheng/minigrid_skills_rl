"""
Test script for training DQN with primitive MiniGrid actions (no skills).
Uses 7 primitive actions: left, right, forward, pickup, drop, toggle, done.
"""
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque

import pfrl
from pfrl import agents, explorers, replay_buffers
from pfrl.q_functions import DistributionalDuelingDQN
from pfrl import nn as pnn


def phi(obs):
    """Preprocess obs: normalize, HWC→CHW."""
    x = obs["image"] if isinstance(obs, dict) else obs
    x = np.asarray(x, dtype=np.float32)
    if x.max() > 1.1:
        x = x / 255.0
    if x.ndim == 3 and x.shape[-1] in [1, 3]:
        x = np.transpose(x, (2, 0, 1))
    return x


def make_agent(n_actions, input_channels, lr=1e-4, gamma=0.99,
               buffer_size=50_000, replay_start_size=1000,
               update_interval=4, target_update_interval=1000,
               start_epsilon=1.0, final_epsilon=0.01,
               final_exploration_steps=10_000, batch_size=32, gpu=-1):
    """Create a Rainbow DQN agent for primitive actions."""

    # Distributional DQN
    n_atoms = 51
    v_min, v_max = -10, 10

    q_func = DistributionalDuelingDQN(
        n_actions, n_atoms, v_min, v_max,
        n_input_channels=input_channels
    )

    # Add noisy networks
    pnn.to_factorized_noisy(q_func, sigma_scale=0.5)

    optimizer = torch.optim.Adam(q_func.parameters(), lr=lr, eps=1.5e-4)

    # Prioritized replay buffer
    replay_buffer = replay_buffers.PrioritizedReplayBuffer(
        capacity=buffer_size,
        alpha=0.5,
        beta0=0.4,
        betasteps=final_exploration_steps,
        num_steps=3,
    )

    # Epsilon-greedy explorer
    explorer = explorers.LinearDecayEpsilonGreedy(
        start_epsilon=start_epsilon,
        end_epsilon=final_epsilon,
        decay_steps=final_exploration_steps,
        random_action_func=lambda: np.random.randint(n_actions)
    )

    agent = agents.CategoricalDoubleDQN(
        q_func,
        optimizer,
        replay_buffer,
        gamma=gamma,
        explorer=explorer,
        minibatch_size=batch_size,
        replay_start_size=replay_start_size,
        update_interval=update_interval,
        target_update_interval=target_update_interval,
        phi=phi,
        gpu=gpu,
    )

    return agent


def train_primitive_dqn(env_name="MiniGrid-DoorKey-5x5-v0",
                        total_steps=50_000,
                        eval_interval=2000,
                        eval_n_episodes=5,
                        save_dir="results/primitive_dqn",
                        seed=42,
                        use_gpu=False):
    """Train DQN with primitive MiniGrid actions."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    # Create environment with RGB observations
    env = gym.make(env_name, render_mode="rgb_array")
    eval_env = gym.make(env_name, render_mode="rgb_array")

    # Get observation shape
    obs, _ = env.reset(seed=seed)
    obs_processed = phi(obs)
    input_channels = obs_processed.shape[0]
    n_actions = env.action_space.n  # 7 primitive actions

    print(f"Environment: {env_name}")
    print(f"Observation shape: {obs_processed.shape}")
    print(f"Number of actions: {n_actions}")
    print(f"Actions: left, right, forward, pickup, drop, toggle, done")

    # Create agent
    gpu = 0 if (use_gpu and torch.cuda.is_available()) else -1
    agent = make_agent(
        n_actions=n_actions,
        input_channels=input_channels,
        gpu=gpu,
    )

    if gpu >= 0:
        print("Using GPU")
    else:
        print("Using CPU")

    # Training loop
    episode_rewards = []
    episode_lengths = []
    current_episode_reward = 0
    current_episode_length = 0

    obs, _ = env.reset(seed=seed)
    step = 0

    print(f"\nTraining for {total_steps} steps...")

    while step < total_steps:
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        current_episode_reward += reward
        current_episode_length += 1
        step += 1

        agent.observe(next_obs, reward, done, False)
        obs = next_obs

        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

            if len(episode_rewards) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_length = np.mean(episode_lengths[-20:])
                print(f"Step {step}/{total_steps} | "
                      f"Episodes: {len(episode_rewards)} | "
                      f"Avg Reward: {avg_reward:.3f} | "
                      f"Avg Length: {avg_length:.1f}")

            current_episode_reward = 0
            current_episode_length = 0
            obs, _ = env.reset()

        # Evaluation
        if step % eval_interval == 0:
            eval_rewards = []
            for _ in range(eval_n_episodes):
                eval_obs, _ = eval_env.reset()
                eval_done = False
                eval_reward = 0
                while not eval_done:
                    with agent.eval_mode():
                        eval_action = agent.act(eval_obs)
                    eval_obs, r, term, trunc, _ = eval_env.step(eval_action)
                    eval_reward += r
                    eval_done = term or trunc
                eval_rewards.append(eval_reward)

            mean_eval = np.mean(eval_rewards)
            print(f"  [Eval] Step {step} | Mean Reward: {mean_eval:.3f}")

    # Save final model
    agent.save(os.path.join(save_dir, "final_agent"))
    print(f"\nTraining complete! Model saved to {save_dir}")

    return agent, episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN with primitive MiniGrid actions")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--save_dir", type=str, default="results/primitive_dqn")

    args = parser.parse_args()

    train_primitive_dqn(
        env_name=args.env,
        total_steps=args.steps,
        seed=args.seed,
        use_gpu=args.gpu,
        save_dir=args.save_dir,
    )
