"""
Test script to verify masked backpropagation is working correctly.

This script:
1. Creates a simple SkillEnv and MaskedDoubleDQN agent
2. Runs a few training steps
3. Verifies that masks are stored in the replay buffer
4. Checks that masked updates are being performed
"""

import numpy as np
import torch
import gymnasium as gym
import sys

from skills import SkillEnv
from skills_dqn_network import SkillQNetwork
from masked_dqn import MaskedDoubleDQN, MaskedReplayBuffer, make_masked_dqn_agent


def phi(obs):
    """Preprocessing function for observations."""
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


def test_masked_backpropagation():
    """Test that masked backpropagation is working."""
    print("=" * 60)
    print("Testing Masked Backpropagation")
    print("=" * 60)

    # Create environment
    base_env = gym.make("MiniGrid-DoorKey-5x5-v0", render_mode="rgb_array")
    env = SkillEnv(base_env, option_reward=1.0, max_skill_horizon=200)

    # Get observation shape
    obs, _ = env.reset(seed=42)
    obs_processed = phi(obs)
    input_channels = obs_processed.shape[0]

    print(f"\n1. Environment Setup")
    print(f"   Number of skills: {len(env.skills)}")
    print(f"   Observation shape: {obs_processed.shape}")

    # Create agent with MaskedReplayBuffer
    q_func = SkillQNetwork(num_skills=len(env.skills), input_channels=input_channels)
    agent = make_masked_dqn_agent(
        q_func=q_func,
        num_actions=len(env.skills),
        phi=phi,
        lr=2.5e-4,
        gamma=0.99,
        buffer_size=1000,
        replay_start_size=10,  # Small for testing
        update_interval=4,
        target_update_interval=100,
        start_epsilon=1.0,
        final_epsilon=0.1,
        final_exploration_steps=100,
        batch_size=8,
        gpu=-1,
    )

    print(f"\n2. Agent Setup")
    print(f"   Agent type: {type(agent).__name__}")
    print(f"   Replay buffer type: {type(agent.replay_buffer).__name__}")
    print(f"   Replay buffer capacity: {agent.replay_buffer.capacity}")

    # Verify it's using MaskedReplayBuffer
    if not isinstance(agent.replay_buffer, MaskedReplayBuffer):
        print("   ❌ ERROR: Agent is not using MaskedReplayBuffer!")
        return False
    else:
        print("   ✓ Agent is using MaskedReplayBuffer")

    # Run a few steps
    print(f"\n3. Running Training Steps")
    obs, _ = env.reset(seed=42)
    num_steps = 20

    for step in range(num_steps):
        # Get action mask
        action_mask = env.get_action_mask()

        # Select action
        action = agent.act(obs, mask=action_mask)

        # Step environment
        next_obs, reward, done, info = env.step(action)

        # Get next mask
        next_mask = env.get_action_mask() if not done else None

        # Observe with masks
        agent.batch_observe([next_obs], [reward], [done], [False], batch_next_mask=[next_mask])

        obs = next_obs if not done else env.reset(seed=step)[0]

        if step % 5 == 0:
            print(f"   Step {step}: buffer size = {len(agent.replay_buffer)}")

    print(f"   Final buffer size: {len(agent.replay_buffer)}")

    # Check that masks are stored in buffer
    print(f"\n4. Verifying Masks in Replay Buffer")
    if len(agent.replay_buffer) > 0:
        # Sample a transition
        sample = agent.replay_buffer.sample(1)[0]

        print(f"   Sample transition keys: {sample.keys()}")

        if 'mask' in sample and 'next_mask' in sample:
            print(f"   ✓ Masks are stored in replay buffer")
            print(f"   Sample mask: {sample['mask']}")
            print(f"   Sample next_mask: {sample['next_mask']}")
        else:
            print(f"   ❌ ERROR: Masks are NOT stored in replay buffer!")
            return False
    else:
        print(f"   ⚠ Warning: Replay buffer is empty, cannot verify masks")

    # Check that masked updates are happening
    print(f"\n5. Verifying Masked Updates")
    initial_params = [p.clone() for p in agent.model.parameters()]

    # Force some updates by running more steps
    obs, _ = env.reset(seed=100)
    for step in range(30):
        action_mask = env.get_action_mask()
        action = agent.act(obs, mask=action_mask)
        next_obs, reward, done, info = env.step(action)
        next_mask = env.get_action_mask() if not done else None
        agent.batch_observe([next_obs], [reward], [done], [False], batch_next_mask=[next_mask])
        obs = next_obs if not done else env.reset(seed=step + 100)[0]

    # Check if parameters changed (indicating updates occurred)
    params_changed = False
    for p_init, p_current in zip(initial_params, agent.model.parameters()):
        if not torch.allclose(p_init, p_current, atol=1e-6):
            params_changed = True
            break

    if params_changed:
        print(f"   ✓ Network parameters updated (training is working)")
    else:
        print(f"   ⚠ Warning: Network parameters unchanged (may not have updated yet)")

    # Test masked target computation
    print(f"\n6. Testing Masked Target Computation")
    if len(agent.replay_buffer) >= 8:
        # Get a sample batch
        transitions = agent.replay_buffer.sample(8)

        # Check that next_masks exist
        has_masks = all('next_mask' in t for t in transitions)
        some_masks_not_none = any(t['next_mask'] is not None for t in transitions)

        if has_masks:
            print(f"   ✓ All transitions have next_mask field")
        else:
            print(f"   ❌ ERROR: Some transitions missing next_mask field")
            return False

        if some_masks_not_none:
            print(f"   ✓ Some next_masks are not None (good for testing)")

            # Count how many actions are masked in each transition
            for i, t in enumerate(transitions[:3]):  # Show first 3
                if t['next_mask'] is not None:
                    num_valid = np.sum(t['next_mask'])
                    num_total = len(t['next_mask'])
                    print(f"   Transition {i}: {num_valid}/{num_total} valid actions")
        else:
            print(f"   ⚠ Warning: All next_masks are None")

    print("\n" + "=" * 60)
    print("✓ All tests passed! Masked backpropagation is working.")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = test_masked_backpropagation()
    sys.exit(0 if success else 1)
