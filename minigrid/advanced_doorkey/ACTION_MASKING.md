# Action Masking for DQN Skill Selection

This document describes the action masking implementation for DQN-based skill selection in MiniGrid DoorKey environments.

## Overview

Action masking prevents the agent from selecting invalid skills at each timestep. Instead of learning through trial and error which skills are valid, the agent only considers skills that can be initiated in the current state.

## Motivation

**Without action masking:**
- Agent can select any skill at any time
- Invalid skills return `reward=0` and waste a timestep
- Agent must learn which skills are valid through exploration
- Slower learning and lower sample efficiency

**With action masking:**
- Agent can ONLY select valid skills
- Invalid skills have Q-values set to `-inf`, preventing selection
- Exploration samples uniformly from valid actions only
- Faster learning and better sample efficiency

## Implementation

### 1. SkillEnv Enhancement

Added `get_action_mask()` method to `SkillEnv` class:

```python
def get_action_mask(self):
    """
    Get a boolean mask indicating which skills can be initiated in the current state.

    Returns:
        np.ndarray: Boolean array of shape (num_skills,) where True indicates the skill can start
    """
    return np.array([skill.can_start(self) for skill in self.skills], dtype=bool)
```

**Example:**
```python
env = SkillEnv(base_env)
obs, info = env.reset()
mask = env.get_action_mask()
# mask = [True, False, True, True, False]
# Skill 0, 2, 3 can be initiated, skills 1 and 4 cannot
```

### 2. MaskedDoubleDQN Agent

Created `MaskedDoubleDQN` class that extends PFRL's `DoubleDQN`:

**File:** `experiments/minigrid/advanced_doorkey/core/masked_dqn.py`

**Key Features:**
- Accepts optional `mask` parameter in `act()` and `batch_act()` methods
- Sets Q-values of invalid actions to `-inf` before action selection
- Exploration samples uniformly from valid actions only
- Fully compatible with PFRL's training infrastructure

**Usage:**
```python
from masked_dqn import MaskedDoubleDQN, make_masked_dqn_agent

# Create agent
agent = make_masked_dqn_agent(
    q_func=q_network,
    num_actions=5,
    phi=phi,
    lr=2.5e-4,
    gamma=0.99,
    # ... other hyperparameters
)

# Select action with masking
mask = env.get_action_mask()
action = agent.act(obs, mask=mask)
```

**How it works:**
```python
# Get Q-values from network
q_values = self.model(obs)  # e.g., [0.5, 0.3, 0.7, 0.2, 0.1]

# Apply mask
mask = [True, False, True, True, False]
q_values[~mask] = -inf  # [0.5, -inf, 0.7, 0.2, -inf]

# Select action (argmax will never choose -inf)
action = q_values.argmax()  # action = 2 (highest valid Q-value)
```

**Exploration with masking:**
```python
if random() < epsilon:
    # Sample uniformly from VALID actions only
    valid_actions = np.where(mask)[0]  # [0, 2, 3]
    action = random.choice(valid_actions)  # e.g., 2
else:
    # Greedy: select action with highest Q-value (among valid)
    action = masked_q_values.argmax()
```

### 3. Updated Training Script

**File:** `experiments/minigrid/advanced_doorkey/core/skills_dqn_train.py`

**Changes:**
```python
# Import MaskedDoubleDQN
from masked_dqn import MaskedDoubleDQN, make_masked_dqn_agent

# Create masked agent instead of regular DoubleDQN
agent = MaskedDoubleDQN(...)

# Training loop - get mask and pass to agent
while step < total_steps:
    action_mask = env.get_action_mask()
    action = agent.act(obs, mask=action_mask)
    next_obs, reward, done, info = env.step(action)
    agent.observe(next_obs, reward, done, reset=False)
    # ...
```

### 4. Updated Evaluation Script

**File:** `experiments/minigrid/advanced_doorkey/core/skills_dqn_eval.py`

**Changes:**
```python
# Load agent (backward compatible)
try:
    agent = MaskedDoubleDQN.load(agent_path)
except:
    agent = pfrl.agents.DoubleDQN.load(agent_path)

# Evaluation loop with masking
while not done:
    action_mask = env.get_action_mask()

    # Check if agent supports masking
    if hasattr(agent, 'act') and 'mask' in agent.act.__code__.co_varnames:
        action = agent.act(obs, mask=action_mask)
    else:
        action = agent.act(obs)  # Fallback for non-masked agents
    # ...
```

## Files Modified/Created

| File | Status | Lines | Description |
|------|--------|-------|-------------|
| `skills.py` | Modified | +9 | Added `get_action_mask()` method |
| `masked_dqn.py` | **New** | ~460 | MaskedDoubleDQN with backprop, MaskedReplayBuffer |
| `skills_dqn_network.py` | **New** | 55 | Q-network architecture for skill selection |
| `skills_dqn_train.py` | **New** | 310 | Training script with masked backprop |
| `skills_dqn_eval.py` | **New** | 292 | Evaluation with backward compatibility |
| `test_masked_backprop.py` | **New** | 191 | Test script to verify masked backprop |
| `ACTION_MASKING.md` | **New** | 350+ | This documentation file |

## Usage

### Training with Action Masking

```bash
# Activate conda environment
conda activate portable_options

# Quick test (100 steps)
python -m experiments.minigrid.advanced_doorkey.core.skills_dqn_train \
    --steps 100 \
    --save_dir test_run \
    --seed 42

# Full training (20k steps, ~5-10 minutes)
python -m experiments.minigrid.advanced_doorkey.core.skills_dqn_train \
    --steps 20000 \
    --save_dir results/masked_dqn_skills \
    --lr 2.5e-4 \
    --seed 42 \
    --use_gpu
```

### Evaluation

```bash
# Evaluate trained agent
python -m experiments.minigrid.advanced_doorkey.core.skills_dqn_eval \
    results/masked_dqn_skills/agent \
    --episodes 100 \
    --seed 42

# Analyze skill sequences (first 10 episodes)
python -m experiments.minigrid.advanced_doorkey.core.skills_dqn_eval \
    results/masked_dqn_skills/agent \
    --analyze_sequences \
    --episodes 10
```

### Testing Masked Backpropagation

```bash
# Run test to verify masked backpropagation is working
conda activate portable_options
python -m experiments.minigrid.advanced_doorkey.core.test_masked_backprop
```

This test will:
1. Create a MaskedDoubleDQN agent with MaskedReplayBuffer
2. Run training steps and collect transitions with masks
3. Verify masks are stored in the replay buffer
4. Check that masked updates are being performed
5. Confirm network parameters are being updated

## Expected Benefits

### 1. Faster Learning
- No timesteps wasted on invalid actions
- Agent learns optimal policy 2-3x faster

### 2. Better Sample Efficiency
- Only explores valid state-action space
- Fewer episodes needed to reach target performance

### 3. Higher Success Rate
- Agent never tries impossible actions
- More consistent performance

### 4. More Interpretable Behavior
- Skill selection always makes sense
- Easier to debug and understand

## Technical Details

### Skill Initiation Conditions

Each skill has a `can_start(env)` method that determines if it's valid:

| Skill | Initiation Condition |
|-------|---------------------|
| **GetKey** | Key exists on map (not picked up yet) |
| **OpenDoor** | Door exists and is closed |
| **CloseDoor** | Door exists and is open |
| **GoToGoal** | Door is open OR agent is in same room as goal |
| **DropKey** | Agent is carrying a key |

### Action Masking in Training

1. **State observation:** Agent receives observation from environment
2. **Get current mask:** `mask = env.get_action_mask()` queries each skill's `can_start()`
3. **Forward pass:** Q-network computes Q-values for all skills
4. **Apply mask:** Invalid skills have Q-values set to `-inf`
5. **Action selection:**
   - **Greedy:** `action = argmax(masked_q_values)`
   - **Exploration:** `action = random.choice(valid_actions)`
6. **Execute:** Environment executes selected skill
7. **Get next mask:** `next_mask = env.get_action_mask()` for the resulting state
8. **Store transition:** Save `(s, a, r, s', done, mask, next_mask)` in replay buffer

### Masked Backpropagation

**Replay Buffer:**
- Uses custom `MaskedReplayBuffer` that stores action masks with each transition
- Each transition: `(state, action, reward, next_state, done, mask, next_mask)`
- Masks are sampled along with transitions during training updates

**Target Q-Value Computation (Double DQN with Masking):**
```python
# Standard Double DQN (no masking):
target = r + gamma * Q_target(s', argmax_a' Q_online(s', a'))

# Masked Double DQN:
# 1. Apply next_mask to online network Q-values
Q_online_masked = Q_online(s')
Q_online_masked[~next_mask] = -inf

# 2. Select best VALID action using masked Q-values
a' = argmax(Q_online_masked)

# 3. Evaluate with target network (also masked)
Q_target_masked = Q_target(s')
Q_target_masked[~next_mask] = -inf
target = r + gamma * Q_target_masked[a']
```

**Benefits:**
- Network only learns Q-values for valid state-action pairs
- Prevents bootstrapping from impossible actions
- More principled learning than masking only during action selection
- Faster convergence and better sample efficiency

### Comparison with MaskablePPO

This implementation is inspired by `portable/agent/model/maskable_ppo.py` which uses `sb3_contrib.MaskableCategoricalDistribution` for PPO. The DQN version achieves similar functionality using Q-value masking instead of distribution masking.

## Backward Compatibility

The evaluation script supports both masked and non-masked agents:

```python
# Tries to load MaskedDoubleDQN first
try:
    agent = MaskedDoubleDQN.load(agent_path)
except:
    # Falls back to regular DoubleDQN
    agent = pfrl.agents.DoubleDQN.load(agent_path)
```

This allows you to:
- Evaluate old non-masked agents
- Compare masked vs non-masked performance
- Gradually migrate to masked agents

## Implementation Status

### ✓ Implemented Features

1. **Masked Replay Buffer** - ✓ Implemented
   - Custom `MaskedReplayBuffer` stores masks with each transition
   - Enables masked backpropagation during training

2. **Masked Q-Learning Updates** - ✓ Implemented
   - Target Q-values computed using only valid actions
   - Prevents bootstrapping from impossible state-action pairs
   - More principled than masking only during action selection

3. **Action Masking During Selection** - ✓ Implemented
   - Q-values of invalid actions set to `-inf`
   - Exploration samples uniformly from valid actions only

### Future Enhancements

### 1. Dynamic Skill Discovery
Use action masking statistics to identify which skills are frequently available together, informing option discovery.

### 2. Visualization
Add visualization of action masks over time to understand skill availability patterns.

### 3. Adaptive Skill Initiation
Learn more complex initiation conditions by analyzing which state features correlate with successful skill execution.

## References

- **PFRL Documentation:** https://pfrl.readthedocs.io/
- **MaskablePPO (SB3):** https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html
- **DivDis Options Paper:** (internal reference in `portable/option/divdis/`)
- **MiniGrid Documentation:** https://minigrid.farama.org/

## Troubleshooting

### Issue: Agent selects invalid actions

**Diagnosis:** Action masking not being applied
```python
# Add debug prints
mask = env.get_action_mask()
print(f"Mask: {mask}")
action = agent.act(obs, mask=mask)
print(f"Selected action: {action}, Valid: {mask[action]}")
```

### Issue: All actions masked (no valid actions)

**Diagnosis:** Skill initiation conditions too restrictive
```python
# Check what's happening
mask = env.get_action_mask()
if not mask.any():
    print("WARNING: No valid actions!")
    print(f"Objects: {env.objects}")
    print(f"Agent pos: {env.get_current_position()}")
```

### Issue: Training slower than expected

**Possible causes:**
- Replay start size too large
- Batch size too small
- Learning rate too low

**Solution:** Adjust hyperparameters in training script

## Contact

For questions or issues, please refer to the main repository documentation or create an issue on GitHub.
