"""Debug script to understand the DoorKey environment layout and skill execution."""
import gymnasium as gym
import numpy as np
from experiments.advanced_doorkey.core.skills import SkillEnv

# Create environment
base_env = gym.make('MiniGrid-DoorKey-5x5-v0', render_mode='human')
env = SkillEnv(base_env, step_penalty=0.01)

# Reset and show layout
obs, info = env.reset(seed=42)

print("=" * 60)
print("ENVIRONMENT LAYOUT")
print("=" * 60)
print(f"Grid size: {env.env.unwrapped.width} x {env.env.unwrapped.height}")
print(f"Agent position: {env.get_current_position()}")
print(f"Agent direction: {env.env.unwrapped.agent_dir}")
print()

# Find all objects
print("Objects:")
for x in range(env.env.unwrapped.width):
    for y in range(env.env.unwrapped.height):
        cell = env.env.unwrapped.grid.get(x, y)
        if cell:
            extra = ""
            if hasattr(cell, 'is_open'):
                extra = f" (open={cell.is_open})"
            if hasattr(cell, 'is_locked'):
                extra += f" (locked={cell.is_locked})"
            print(f"  {cell.type:10s} at ({x}, {y}){extra}")

print()
print(f"Key object: {env.objects['key']}")
print(f"Door object: {env.objects['door']}")
print(f"Goal object: {env.objects['goal']}")

print()
print("=" * 60)
print("TESTING SKILLS ONE BY ONE")
print("=" * 60)

skill_names = [s.name for s in env.skills]
print(f"Available skills: {skill_names}")
print()

# Test each skill and see what happens
for skill_idx in range(len(env.skills)):
    # Reset environment
    obs, info = env.reset(seed=42)
    skill = env.skills[skill_idx]

    print(f"\nTesting skill: {skill.name}")
    print(f"  Can start: {skill.can_start(env)}")

    if skill.can_start(env):
        print(f"  Agent pos before: {env.get_current_position()}")
        obs, reward, terminated, truncated, info = env.step(skill_idx)
        print(f"  Agent pos after: {env.get_current_position()}")
        print(f"  Reward: {reward}")
        print(f"  Terminated (goal reached): {terminated}")
        print(f"  Skill completed: {info.get('skill_terminated', False)}")

        if terminated and not info.get('skill_terminated', False):
            print(f"  *** ACCIDENTALLY REACHED GOAL DURING {skill.name}! ***")

print()
print("=" * 60)
print("RUNNING OPTIMAL SEQUENCE")
print("=" * 60)

obs, info = env.reset(seed=42)
print(f"Initial position: {env.get_current_position()}")
print(f"Goal position: {env.objects['goal']}")
print()

sequence = [0, 1, 3]  # get_key, open_door, go_to_goal
for step, skill_idx in enumerate(sequence):
    skill_name = skill_names[skill_idx]
    print(f"Step {step}: {skill_name}")
    print(f"  Position before: {env.get_current_position()}")

    obs, reward, terminated, truncated, info = env.step(skill_idx)

    print(f"  Position after: {env.get_current_position()}")
    print(f"  Reward: {reward:.3f}")
    print(f"  Terminated: {terminated}")
    print(f"  Skill completed: {info.get('skill_terminated', False)}")

    if terminated:
        print(f"\n  Episode ended at step {step + 1}!")
        break

print()
print("=" * 60)
print("TESTING MULTIPLE SEEDS")
print("=" * 60)

# Test seed 0 in detail
test_seed = 0
obs, info = env.reset(seed=test_seed)

print(f"Seed {test_seed} layout:")
print(f"  Agent position: {env.get_current_position()}")
print(f"  Key position: {env.objects['key'].position if env.objects['key'] else 'None'}")
print(f"  Door position: {env.objects['door'].position if env.objects['door'] else 'None'}")
print(f"  Goal position: {env.objects['goal']}")

# Check door state
door_obj = env.get_door_obj()
if door_obj:
    print(f"  Door is_open: {door_obj.is_open}")
    print(f"  Door is_locked: {getattr(door_obj, 'is_locked', 'N/A')}")

# Print full grid
print("\nFull grid for seed 0:")
for y in range(env.env.unwrapped.height):
    row = ""
    for x in range(env.env.unwrapped.width):
        cell = env.env.unwrapped.grid.get(x, y)
        pos = (x, y)
        if pos == tuple(env.env.unwrapped.agent_pos):
            row += " A "
        elif cell is None:
            row += " . "
        elif cell.type == "wall":
            row += " W "
        elif cell.type == "door":
            row += " D "
        elif cell.type == "key":
            row += " K "
        elif cell.type == "goal":
            row += " G "
        else:
            row += " ? "
    print(row)

print()
print("Trying open_door skill...")
if env.skills[1].can_start(env):
    obs, reward, terminated, truncated, info = env.step(1)
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    print(f"  Truncated: {truncated}")
    print(f"  Agent final pos: {env.get_current_position()}")
    print(f"  env._env_done: {env._env_done}")
    print(f"  env._timestep (primitive steps): {env._timestep}")

    if terminated:
        print(f"  *** Goal reached during open_door! ***")
        print(f"  But wait - was the door even opened? Door is_open: {env.get_door_obj().is_open if env.get_door_obj() else 'N/A'}")

        # Check if goal position matches agent position
        print(f"  Goal pos: {env.objects['goal']}, Agent pos: {env.get_current_position()}")
        if env.objects['goal'] == env.get_current_position():
            print(f"  Agent IS at goal position!")
        else:
            print(f"  Agent is NOT at goal position - something else terminated the episode!")

print()
print("=" * 60)
env.close()
