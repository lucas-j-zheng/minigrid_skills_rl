# DQN Training Flow

## What skills_dqn_train.py Does

Trains a DQN agent to select skills in MiniGrid DoorKey environment.

## Setup

```python
# Create environments
env = SkillEnv(base_env, option_reward=1.0, max_skill_horizon=200)
eval_env = SkillEnv(eval_base_env)

# Create Q-network (CNN: image → 5 Q-values)
q_func = SkillQNetwork(num_skills=5, input_channels=3)

# Create agent
agent = make_masked_dqn_agent(
    q_func=q_func,
    num_actions=5,
    phi=phi,  # Preprocess: normalize, HWC→CHW
    lr=2.5e-4,
    gamma=0.99,
    buffer_size=50_000,
    replay_start_size=1000,     # Start training after 1k samples
    update_interval=4,          # Update every 4 steps
    target_update_interval=1000, # Sync target network every 1k steps
    batch_size=32,
)
```

## Training Loop

```python
obs = env.reset()
step = 0

while step < total_steps:
    # 1. Get action mask
    mask = env.get_action_mask()  # [True, True, False, False, False]

    # 2. Select action
    action = agent.act(obs, mask=mask)
    # - Forward pass: obs → Q-values
    # - Apply mask: q[~mask] = -inf
    # - Epsilon-greedy: explore (random valid) or exploit (argmax)

    # 3. Execute skill
    next_obs, reward, done, info = env.step(action)
    # Skill runs 10-200 primitive steps until completion

    # 4. Get next mask
    next_mask = env.get_action_mask() if not done else None

    # 5. Store transition + train
    agent.batch_observe(
        [next_obs], [reward], [done], [False],
        batch_next_mask=[next_mask]
    )
    # - Stores in MaskedReplayBuffer with both masks
    # - If buffer >= 1000 and step % 4 == 0:
    #     * Sample batch of 32
    #     * Compute masked target Q-values
    #     * Backprop, update Q-network

    obs = next_obs
    step += 1

    if done:
        obs = env.reset()
```

## DQN Update (inside agent.batch_observe)

```python
# Sample batch
transitions = replay_buffer.sample(32)

# Compute current Q
current_q = Q_network(states)[actions]

# Compute masked targets (Double DQN)
with torch.no_grad():
    # Online network selects
    next_q_online = Q_network(next_states)
    for i, mask in enumerate(next_masks):
        next_q_online[i][~mask] = -inf  # Mask invalid
    best_actions = next_q_online.argmax(dim=1)

    # Target network evaluates
    next_q_target = Q_target(next_states)
    for i, mask in enumerate(next_masks):
        next_q_target[i][~mask] = -inf  # Also mask
    next_q = next_q_target[best_actions]

    target_q = rewards + gamma * next_q * (1 - dones)

# Update
loss = mse_loss(current_q, target_q)
loss.backward()
optimizer.step()

# Sync target network (every 1000 steps)
if step % 1000 == 0:
    Q_target = copy(Q_network)
```

## Hyperparameters

```python
total_steps = 20_000
replay_start_size = 1_000
update_interval = 4
target_update_interval = 1_000
batch_size = 32
buffer_size = 50_000
epsilon: 1.0 → 0.01 over 10k steps
gamma = 0.99
lr = 2.5e-4
```

## Example Run

```
Agent at (1,1), key at (3,2), door closed

Step 1:
  mask = [T, T, F, F, F]  # Can GetKey or OpenDoor
  q = [0.45, 0.62, 0.31, 0.58, 0.23]
  masked_q = [0.45, 0.62, -inf, -inf, -inf]
  action = 1 (OpenDoor)

  OpenDoor executes (15 primitive steps)
  reward = 1.0
  next_mask = [T, F, T, T, F]  # OpenDoor now invalid

Step 2:
  mask = [T, F, T, T, F]
  q = [0.58, 0.32, 0.41, 0.76, 0.29]
  masked_q = [0.58, -inf, 0.41, 0.76, -inf]
  action = 3 (GoToGoal)

  GoToGoal executes
  reward = 1.0
  done = True

Episode return: 2.0
```

## Key Differences from Standard DQN

Standard DQN (primitive actions):
- Action space: forward, left, right, pickup, drop
- ~500 decisions per episode
- Each action = 1 step

This (skill-based):
- Action space: GetKey, OpenDoor, CloseDoor, GoToGoal, DropKey
- ~5-10 decisions per episode
- Each action = 10-200 steps
- Action masking prevents invalid skills
- Masked backpropagation ensures valid targets
