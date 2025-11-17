import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from collections import defaultdict
import matplotlib.pyplot as plt

import pfrl
from pfrl.agents import DoubleDQN

from .skills import SkillEnv
from .skills_dqn_network import SkillQNetwork
from .masked_dqn import MaskedDoubleDQN


def phi(obs):
    """
    Preprocessing function for observations.
    Same as in training script.
    """
    if isinstance(obs, dict):
        x = obs["image"]
    else:
        x = obs

    x = np.asarray(x, dtype=np.float32)

    if x.max() > 1.1:
        x = x / 255.0

    if x.ndim == 3 and x.shape[-1] in [1, 3]:
        x = np.transpose(x, (2, 0, 1))

    return x


def evaluate_dqn(
    agent_path,
    env_name="MiniGrid-DoorKey-5x5-v0",
    num_episodes=100,
    render=False,
    seed=42,
    save_dir=None,
):
    """
    Evaluate a trained DQN agent.

    Args:
        agent_path: Path to saved agent directory
        env_name: Gymnasium environment ID
        num_episodes: Number of episodes to evaluate
        render: Whether to render episodes
        seed: Random seed
        save_dir: Directory to save results (if None, uses agent_path)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load agent
    if not os.path.exists(agent_path):
        raise ValueError(f"Agent path does not exist: {agent_path}")

    # Create environment first to get observation shape and number of skills
    render_mode = "human" if render else "rgb_array"
    base_env = gym.make(env_name, render_mode=render_mode)
    env = SkillEnv(base_env, option_reward=1.0, max_skill_horizon=200)

    # Get observation shape
    obs, _ = env.reset(seed=seed)
    obs_processed = phi(obs)
    input_channels = obs_processed.shape[0]
    num_skills = len(env.skills)

    print(f"Loading agent from {agent_path}...")
    print(f"Observation shape: {obs_processed.shape}")
    print(f"Number of skills: {num_skills}")

    # Recreate Q-network architecture
    q_func = SkillQNetwork(num_skills=num_skills, input_channels=input_channels)

    # Recreate agent with same architecture (no GPU for evaluation)
    from .masked_dqn import make_masked_dqn_agent
    agent = make_masked_dqn_agent(
        q_func=q_func,
        num_actions=num_skills,
        phi=phi,
        lr=2.5e-4,  # Learning rate doesn't matter for evaluation
        gamma=0.99,
        buffer_size=1000,  # Small buffer for evaluation
        replay_start_size=100,
        update_interval=4,
        target_update_interval=1000,
        start_epsilon=0.0,  # No exploration during evaluation
        final_epsilon=0.0,
        final_exploration_steps=1,
        batch_size=32,
        gpu=-1,  # CPU only for evaluation
    )

    # Load the saved agent state
    agent.load(agent_path)
    agent.training = False  # Set to evaluation mode

    if save_dir is None:
        save_dir = os.path.dirname(agent_path)

    # Metrics
    episode_rewards = []
    episode_lengths = []
    skill_counts = defaultdict(int)
    skill_success_counts = defaultdict(int)
    skill_sequences = []

    skill_names = [skill.name for skill in env.skills]
    print(f"Evaluating on {num_episodes} episodes...")
    print(f"Available skills: {skill_names}")

    # Evaluation loop
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0
        episode_length = 0
        done = False
        episode_skills = []

        while not done:
            # Get valid action mask
            action_mask = env.get_action_mask()

            # Select action (skill) with masking
            if hasattr(agent, 'act') and 'mask' in agent.act.__code__.co_varnames:
                # MaskedDoubleDQN - supports masking
                action = agent.act(obs, mask=action_mask)
            else:
                # Regular DoubleDQN - no masking support
                action = agent.act(obs)

            episode_skills.append(action)
            skill_counts[action] += 1

            # Debug output
            skill_name = skill_names[action]
            valid_skills = [skill_names[i] for i, valid in enumerate(action_mask) if valid]
            if render:
                print(f"Step {episode_length}: Selected '{skill_name}' | Valid skills: {valid_skills}")

            # Execute skill
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            # Track skill success
            if info.get("skill_terminated", False):
                skill_success_counts[action] += 1
                if render:
                    print(f"  -> Skill '{skill_name}' completed! Reward: {reward}")
            elif info.get("blocked", False):
                if render:
                    print(f"  -> Skill '{skill_name}' was blocked (can't start)")
            else:
                if render:
                    print(f"  -> Skill '{skill_name}' did not complete")

            if render:
                env.render()
                import time
                time.sleep(0.5)  # Slow down to see what's happening

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        skill_sequences.append(episode_skills)

        if (ep + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episodes {ep - 8}-{ep + 1}: Avg Reward = {avg_reward:.2f}")

    # Compute statistics
    results = {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "success_rate": np.mean([r > 0 for r in episode_rewards]),
        "skill_counts": dict(skill_counts),
        "skill_success_counts": dict(skill_success_counts),
        "skill_sequences": skill_sequences,
    }

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Mean Reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
    print(f"Mean Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Success Rate: {results['success_rate']:.1%}")
    print()

    print("Skill Usage:")
    total_skills = sum(skill_counts.values())
    for skill_idx in range(len(skill_names)):
        count = skill_counts.get(skill_idx, 0)
        success = skill_success_counts.get(skill_idx, 0)
        success_rate = success / count if count > 0 else 0
        print(
            f"  {skill_names[skill_idx]:12s} - Used: {count:4d} ({count / total_skills:5.1%}) | "
            f"Success: {success:4d} ({success_rate:5.1%})"
        )
    print("=" * 60)

    # Plot results
    plot_evaluation_results(results, skill_names, save_dir)

    # Save results
    save_path = os.path.join(save_dir, "eval_results.npz")
    np.savez(
        save_path,
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        skill_counts=np.array(list(skill_counts.values())),
        mean_reward=results["mean_reward"],
        std_reward=results["std_reward"],
    )
    print(f"\nResults saved to {save_path}")

    return results


def plot_evaluation_results(results, skill_names, save_dir):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Episode rewards histogram
    axes[0, 0].hist(results["episode_rewards"], bins=20, edgecolor="black", alpha=0.7)
    axes[0, 0].axvline(
        results["mean_reward"], color="red", linestyle="--", label=f'Mean: {results["mean_reward"]:.2f}'
    )
    axes[0, 0].set_xlabel("Episode Reward")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("Distribution of Episode Rewards")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths histogram
    axes[0, 1].hist(results["episode_lengths"], bins=20, edgecolor="black", alpha=0.7, color="orange")
    axes[0, 1].axvline(
        results["mean_length"],
        color="red",
        linestyle="--",
        label=f'Mean: {results["mean_length"]:.1f}',
    )
    axes[0, 1].set_xlabel("Episode Length")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title("Distribution of Episode Lengths")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Skill usage counts
    skill_counts = results["skill_counts"]
    skill_indices = sorted(skill_counts.keys())
    counts = [skill_counts[i] for i in skill_indices]
    skill_labels = [skill_names[i] for i in skill_indices]

    axes[1, 0].bar(range(len(skill_indices)), counts, tick_label=skill_labels)
    axes[1, 0].set_xlabel("Skill")
    axes[1, 0].set_ylabel("Usage Count")
    axes[1, 0].set_title("Skill Usage Frequency")
    axes[1, 0].tick_params(axis="x", rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Skill success rates
    skill_success_counts = results["skill_success_counts"]
    success_rates = []
    for i in skill_indices:
        count = skill_counts.get(i, 0)
        success = skill_success_counts.get(i, 0)
        success_rates.append(success / count if count > 0 else 0)

    axes[1, 1].bar(range(len(skill_indices)), success_rates, tick_label=skill_labels, color="green")
    axes[1, 1].set_xlabel("Skill")
    axes[1, 1].set_ylabel("Success Rate")
    axes[1, 1].set_title("Skill Success Rates")
    axes[1, 1].tick_params(axis="x", rotation=45)
    axes[1, 1].set_ylim([0, 1.1])
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = os.path.join(save_dir, "eval_plots.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Evaluation plots saved to {save_path}")


def analyze_skill_sequences(agent_path, env_name="MiniGrid-DoorKey-5x5-v0", num_episodes=10, seed=42):
    """
    Analyze skill sequences used by the agent.
    Useful for understanding learned strategies.
    """
    results = evaluate_dqn(agent_path, env_name, num_episodes, render=False, seed=seed)

    # Load skill names
    base_env = gym.make(env_name, render_mode="rgb_array")
    env = SkillEnv(base_env)
    skill_names = [skill.name for skill in env.skills]

    print("\n" + "=" * 60)
    print("SKILL SEQUENCE ANALYSIS")
    print("=" * 60)

    sequences = results["skill_sequences"]
    for ep_idx, seq in enumerate(sequences[:min(10, num_episodes)]):
        seq_names = [skill_names[s] for s in seq]
        reward = results["episode_rewards"][ep_idx]
        print(f"Episode {ep_idx + 1} (Reward: {reward:.1f}): {' -> '.join(seq_names)}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_path", type=str, help="Path to saved agent directory")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-5x5-v0")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--analyze_sequences", action="store_true")
    args = parser.parse_args()

    if args.analyze_sequences:
        analyze_skill_sequences(args.agent_path, args.env, num_episodes=10, seed=args.seed)
    else:
        evaluate_dqn(args.agent_path, args.env, args.episodes, args.render, args.seed)
