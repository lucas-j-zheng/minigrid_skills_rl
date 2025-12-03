"""
Visualization utilities for value functions in MiniGrid environments.
Works with both primitive action agents and skill-based agents.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
from typing import Optional, Callable
import os

from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
from PIL import Image


def resize_obs(obs: np.ndarray, target_size: int) -> np.ndarray:
    """Resize observation to target size to match training format."""
    if obs.max() > 1.1:
        img = Image.fromarray(obs.astype(np.uint8))
    else:
        img = Image.fromarray((obs * 255).astype(np.uint8))
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32)


def get_state_value(agent, obs, action_mask=None):
    """
    Get the value of a state by taking max Q-value over available actions.

    Args:
        agent: PFRL agent with Q-function
        obs: Observation (image)
        action_mask: Optional boolean mask for valid actions (for skill agents)

    Returns:
        float: State value (max Q-value)
    """
    with agent.eval_mode():
        # Get Q-values for all actions
        obs_batch = agent.batch_states([obs], agent.device, agent.phi)

        with torch.no_grad():
            q_values = agent.model(obs_batch).q_values.cpu().numpy()[0]

        # Apply action mask if provided (for skill-based agents)
        if action_mask is not None:
            q_values = np.where(action_mask, q_values, -np.inf)

        # Return max Q-value as state value
        return float(np.max(q_values)) if not np.all(np.isinf(q_values)) else 0.0


def create_state_condition(env, unwrapped, agent_pos, has_key=False, door_open=False,
                           key_pos=None, door_pos=None, key_obj=None, door_obj=None):
    """
    Set up environment to specific state condition WITHOUT resetting.

    Args:
        env: The wrapped environment (with RGBImgObsWrapper) for generating observations
        unwrapped: Unwrapped MiniGrid environment for state manipulation
        agent_pos: (x, y) position for agent
        has_key: Whether agent should be holding the key
        door_open: Whether the door should be open
        key_pos: Position of the key in original layout
        door_pos: Position of the door in original layout
        key_obj: Reference to the key object
        door_obj: Reference to the door object

    Returns:
        obs: Observation from this state (RGB image matching training format)
        success: Whether state setup was successful
    """
    try:
        # Set agent position
        unwrapped.agent_pos = np.array(agent_pos)

        # Handle key state
        if has_key and key_obj is not None:
            # Remove key from grid and put in agent's hand
            if key_pos is not None:
                unwrapped.grid.set(*key_pos, None)
            unwrapped.carrying = key_obj
        else:
            # Put key back on grid if it exists
            if key_obj is not None and key_pos is not None:
                unwrapped.grid.set(*key_pos, key_obj)
            unwrapped.carrying = None

        # Handle door state
        if door_obj is not None:
            door_obj.is_open = door_open
            if door_open:
                door_obj.is_locked = False
            else:
                door_obj.is_locked = True

        # Generate observation through the wrapped env to get RGB format
        # This matches the training observation format
        obs = env.observation(unwrapped.gen_obs())

        return obs, True

    except Exception as e:
        print(f"Warning: Failed to create state condition: {e}")
        return None, False


def visualize_value_function(
    agent,
    env_name: str,
    save_path: str,
    seed: int = 42,
    get_action_mask_fn: Optional[Callable] = None,
    title_prefix: str = "",
    use_skill_env: bool = False,
    obs_size: Optional[int] = None
):
    """
    Create value function heatmaps for three conditions:
    1. Agent doesn't have key
    2. Agent has key
    3. Door is open

    Args:
        agent: Trained PFRL agent
        env_name: MiniGrid environment name
        save_path: Path to save the plot
        seed: Random seed
        get_action_mask_fn: Optional function to get action masks (for skill agents)
        title_prefix: Prefix for plot title (e.g., "Episode 1000")
        use_skill_env: If True, wrap env in SkillEnv for action mask compatibility
        obs_size: Optional observation size to resize to (e.g., 84 for 16x16 envs)
    """
    import gymnasium as gym
    import minigrid

    # Create environment with RGB wrappers to match training observation format
    base_env = gym.make(env_name, render_mode="rgb_array")
    base_env = RGBImgObsWrapper(base_env)  # Full top-down RGB view
    rgb_env = ImgObsWrapper(base_env)  # Extract image from obs dict

    if use_skill_env and get_action_mask_fn is not None:
        # Import here to avoid circular imports
        from experiments.advanced_doorkey.core.skills import SkillEnv
        env = SkillEnv(rgb_env, option_reward=1.0, max_skill_horizon=200)
    else:
        env = rgb_env

    # Reset ONCE with seed to get consistent layout
    env.reset(seed=seed)
    unwrapped = env.unwrapped if hasattr(env, 'unwrapped') else env.env.unwrapped

    width, height = unwrapped.width, unwrapped.height

    # Find key and door positions/objects ONCE
    key_pos = None
    door_pos = None
    key_obj = None
    door_obj = None
    goal_pos = None

    for x in range(width):
        for y in range(height):
            cell = unwrapped.grid.get(x, y)
            if cell is not None:
                if cell.type == 'key':
                    key_pos = (x, y)
                    key_obj = cell
                elif cell.type == 'door':
                    door_pos = (x, y)
                    door_obj = cell
                elif cell.type == 'goal':
                    goal_pos = (x, y)

    # Initialize value grids for three conditions
    conditions = [
        ("No Key", False, False),
        ("Has Key", True, False),
        ("Door Open", False, True),
    ]

    value_grids = []

    for condition_name, has_key, door_open in conditions:
        value_grid = np.zeros((width, height))

        # For each position in the grid
        for x in range(width):
            for y in range(height):
                # Skip walls
                cell = unwrapped.grid.get(x, y)
                if cell is not None and cell.type == 'wall':
                    value_grid[x, y] = np.nan
                    continue

                # Create state with agent at this position (no reset!)
                obs, success = create_state_condition(
                    rgb_env, unwrapped, (x, y),
                    has_key=has_key, door_open=door_open,
                    key_pos=key_pos, door_pos=door_pos,
                    key_obj=key_obj, door_obj=door_obj
                )

                if success and obs is not None:
                    # Resize observation if needed to match training format
                    if obs_size is not None:
                        obs = resize_obs(obs, obs_size)

                    # Get action mask if function provided (for skill agents)
                    action_mask = None
                    if get_action_mask_fn is not None:
                        # Sync SkillEnv state if needed
                        if hasattr(env, '_find_objs'):
                            env._find_objs()
                        action_mask = get_action_mask_fn(env)

                    # Get state value
                    value = get_state_value(agent, obs, action_mask)
                    value_grid[x, y] = value
                else:
                    value_grid[x, y] = np.nan

        value_grids.append((condition_name, value_grid))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Find global min/max for consistent color scale
    all_values = []
    for _, grid in value_grids:
        valid_values = grid[~np.isnan(grid)]
        if len(valid_values) > 0:
            all_values.extend(valid_values)

    if len(all_values) > 0:
        vmin, vmax = np.min(all_values), np.max(all_values)
    else:
        vmin, vmax = 0, 1

    # Plot each condition
    for idx, (condition_name, value_grid) in enumerate(value_grids):
        ax = axes[idx]

        # Plot heatmap (transpose for correct orientation)
        im = ax.imshow(
            value_grid.T,
            origin='lower',
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            aspect='equal'
        )

        ax.set_title(f"{condition_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")

        # Add grid
        ax.set_xticks(np.arange(width))
        ax.set_yticks(np.arange(height))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)

        # Add colorbar
        plt.colorbar(im, ax=ax, label='State Value')

    # Overall title
    if title_prefix:
        fig.suptitle(f"{title_prefix} - Value Function Heatmaps",
                     fontsize=14, fontweight='bold', y=1.02)
    else:
        fig.suptitle("Value Function Heatmaps",
                     fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    env.close()

    print(f"Value function visualization saved to {save_path}")


def log_value_function_periodically(
    agent,
    env_name: str,
    episode_num: int,
    log_dir: str,
    log_interval: int = 100,
    get_action_mask_fn: Optional[Callable] = None,
    agent_type: str = "primitive",
    obs_size: Optional[int] = None
):
    """
    Log value function visualization every N episodes.

    Args:
        agent: Trained agent
        env_name: Environment name
        episode_num: Current episode number
        log_dir: Directory to save plots
        log_interval: Log every N episodes
        get_action_mask_fn: Optional function for action masks (skill agents)
        agent_type: "primitive" or "skill" for labeling
        obs_size: Optional observation size to resize to (e.g., 84 for 16x16 envs)
    """
    if episode_num % log_interval == 0:
        save_path = os.path.join(
            log_dir,
            f"value_function_{agent_type}_ep{episode_num:06d}.png"
        )
        visualize_value_function(
            agent=agent,
            env_name=env_name,
            save_path=save_path,
            get_action_mask_fn=get_action_mask_fn,
            title_prefix=f"Episode {episode_num}",
            use_skill_env=(agent_type == "skill"),
            obs_size=obs_size
        )
