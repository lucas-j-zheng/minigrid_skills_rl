import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt

import pfrl

from .skills import SkillEnv
from .skills_dqn_network import SkillQNetwork
from .masked_dqn import MaskedDoubleDQN, make_masked_dqn_agent

# Import visualization utilities
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
from experiments.value_visualization import log_value_function_periodically


def phi(obs):
    """Preprocess obs: normalize, HWC→CHW."""
    x = obs["image"] if isinstance(obs, dict) else obs
    x = np.asarray(x, dtype=np.float32)
    if x.max() > 1.1:
        x = x / 255.0
    if x.ndim == 3 and x.shape[-1] in [1, 3]:
        x = np.transpose(x, (2, 0, 1))
    return x


def train_dqn(env_name="MiniGrid-DoorKey-5x5-v0", total_steps=20_000, lr=2.5e-4, gamma=0.99,
              buffer_size=50_000, batch_size=32, replay_start_size=1000, update_interval=4,
              target_update_interval=1000, start_epsilon=1.0, final_epsilon=0.01,
              final_exploration_steps=10_000, eval_interval=1000, eval_n_episodes=10,
              save_dir="results/dqn_skills", seed=42, use_gpu=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(save_dir, exist_ok=True)

    base_env = gym.make(env_name, render_mode="rgb_array")
    env = SkillEnv(base_env, option_reward=1.0, max_skill_horizon=200)
    eval_base_env = gym.make(env_name, render_mode="rgb_array")
    eval_env = SkillEnv(eval_base_env, option_reward=1.0, max_skill_horizon=200)

    obs, _ = env.reset(seed=seed)
    obs_processed = phi(obs)
    input_channels = obs_processed.shape[0]

    print(f"Observation shape: {obs_processed.shape}")
    print(f"Number of skills: {len(env.skills)}")

    q_func = SkillQNetwork(num_skills=len(env.skills), input_channels=input_channels)

    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
        q_func = q_func.to(device)
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    agent = make_masked_dqn_agent(
        q_func=q_func, num_actions=len(env.skills), phi=phi, lr=lr, gamma=gamma,
        buffer_size=buffer_size, replay_start_size=replay_start_size,
        update_interval=update_interval, target_update_interval=target_update_interval,
        start_epsilon=start_epsilon, final_epsilon=final_epsilon,
        final_exploration_steps=final_exploration_steps, batch_size=batch_size,
        gpu=0 if (use_gpu and torch.cuda.is_available()) else -1,
    )

    episode_rewards = []
    episode_lengths = []
    eval_rewards = []
    current_episode_reward = 0
    current_episode_length = 0

    obs, info = env.reset(seed=seed)
    step = 0

    print(f"\nTraining for {total_steps} steps...")
    print(f"Buffer: {type(agent.replay_buffer).__name__}, start: {replay_start_size}")
    print(f"Epsilon: {start_epsilon} → {final_epsilon} over {final_exploration_steps} steps")

    while step < total_steps:
        action_mask = env.get_action_mask()
        action = agent.act(obs, mask=action_mask)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_mask = env.get_action_mask() if not done else None

        current_episode_reward += reward
        current_episode_length += 1
        step += 1

        agent.batch_observe([next_obs], [reward], [done], [False], batch_next_mask=[next_mask])
        obs = next_obs

        if done:
            episode_rewards.append(current_episode_reward)
            episode_lengths.append(current_episode_length)

            if len(episode_rewards) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_length = np.mean(episode_lengths[-10:])
                epsilon = agent.explorer.epsilon
                print(
                    f"Step {step}/{total_steps} | Episodes: {len(episode_rewards)} | "
                    f"Avg Reward (last 10): {avg_reward:.2f} | "
                    f"Avg Length: {avg_length:.1f} | "
                    f"Epsilon: {epsilon:.3f}"
                )

            # Log value function visualizations every 100 episodes
            # Pass action mask function for skill-based agent
            log_value_function_periodically(
                agent=agent,
                env_name=env_name,
                episode_num=len(episode_rewards),
                log_dir=os.path.join(save_dir, "value_plots"),
                log_interval=100,
                get_action_mask_fn=lambda e: e.get_action_mask(),
                agent_type="skill"
            )

            obs, info = env.reset(seed=seed)
            current_episode_reward = 0
            current_episode_length = 0

        # Evaluation
        if step % eval_interval == 0 and step > 0:
            eval_reward = evaluate(agent, eval_env, n_episodes=eval_n_episodes, seed=seed)
            eval_rewards.append((step, eval_reward))
            print(f"Evaluation at step {step}: {eval_reward:.2f}")

    # Final evaluation
    final_eval_reward = evaluate(agent, eval_env, n_episodes=100, seed=seed)
    print(f"\nFinal evaluation over 100 episodes: {final_eval_reward:.2f}")

    # Save agent
    agent.save(os.path.join(save_dir, "agent"))
    print(f"Agent saved to {save_dir}/agent")

    # Save training curves
    plot_training_curves(episode_rewards, episode_lengths, eval_rewards, save_dir)

    # Save metrics
    np.save(os.path.join(save_dir, "episode_rewards.npy"), episode_rewards)
    np.save(os.path.join(save_dir, "episode_lengths.npy"), episode_lengths)
    np.save(os.path.join(save_dir, "eval_rewards.npy"), eval_rewards)

    return agent, episode_rewards, episode_lengths, eval_rewards


def evaluate(agent, env, n_episodes=10, seed=42):
    """Evaluate agent for n_episodes with action masking."""
    rewards = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0
        done = False

        while not done:
            # Get valid action mask
            action_mask = env.get_action_mask()
            # Select action with masking
            action = agent.act(obs, mask=action_mask)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    return np.mean(rewards)


def plot_training_curves(episode_rewards, episode_lengths, eval_rewards, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Episode rewards
    axes[0].plot(episode_rewards, alpha=0.3)
    if len(episode_rewards) > 50:
        window = min(50, len(episode_rewards) // 10)
        smoothed = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(smoothed, linewidth=2)
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Return")
    axes[0].set_title("Training Returns")
    axes[0].grid(True, alpha=0.3)

    # Episode lengths
    axes[1].plot(episode_lengths, alpha=0.3)
    if len(episode_lengths) > 50:
        window = min(50, len(episode_lengths) // 10)
        smoothed = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
        axes[1].plot(smoothed, linewidth=2)
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Length")
    axes[1].set_title("Episode Lengths")
    axes[1].grid(True, alpha=0.3)

    # Evaluation rewards
    if len(eval_rewards) > 0:
        steps, rewards = zip(*eval_rewards)
        axes[2].plot(steps, rewards, marker="o")
        axes[2].set_xlabel("Training Step")
        axes[2].set_ylabel("Avg Evaluation Return")
        axes[2].set_title("Evaluation Performance")
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="results/dqn_skills")
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()

    train_dqn(
        env_name=args.env,
        total_steps=args.steps,
        lr=args.lr,
        seed=args.seed,
        save_dir=args.save_dir,
        use_gpu=args.use_gpu,
    )
