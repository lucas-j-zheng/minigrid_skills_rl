# Complex DoorKey Task

## Task Description

A more challenging variant of the DoorKey environment where the agent must complete a specific sequence of actions to receive reward.

## Success Criteria

Reward (+1.0) is given **only** when ALL conditions are met:

1. **Key is dropped in the first room** (the room where agent starts)
2. **Door is closed**
3. **Agent is at the goal position**

## Required Skill Sequence

```
┌─────────────────────────────────────────────────────────┐
│  FIRST ROOM          │ DOOR │      SECOND ROOM         │
│                      │      │                          │
│  [Agent Start]       │      │                          │
│  [Key]               │      │           [Goal]         │
│                      │      │                          │
└─────────────────────────────────────────────────────────┘
```

### Key Constraint

**Episode only terminates when agent reaches goal WITH preconditions met.** Agent can step on goal without preconditions - the episode continues, allowing the agent to step off and complete the task. This enables exploration and learning without premature termination.

### Optimal Sequence:

| Step | Skill | State After |
|------|-------|-------------|
| 1 | `GetKey` | Agent holding key |
| 2 | `OpenDoor` | Door open, agent at door |
| 3 | `move_*` | Navigate through door |
| 4 | `move_*` | Navigate back to first room |
| 5 | `DropKey` | Key on floor in first room |
| 6 | `move_*` | Navigate to door |
| 7 | `CloseDoor` | Door closed (agent in second room) |
| 8 | `GoToGoal` | Agent at goal → reward! |

**Note:** `GoToGoal` should only be used as the FINAL step when preconditions are met.

## Why This Task is Hard

1. **Delayed reward**: Agent must perform ~8 skills before receiving any reward
2. **State requirements**: Must track key location and door state
3. **Precise sequencing**: Must complete preconditions before goal gives reward
4. **Credit assignment**: Long sequence between actions and reward makes learning difficult

## Implementation

### New Reward Mode: `"complex_doorkey"`

Add to `SkillEnv.REWARD_MODES`:

```python
REWARD_MODES = ("goal", "goal_closed_door", "complex_doorkey")
```

### Reward Logic

```python
if self.reward_mode == "complex_doorkey":
    # Check all three conditions
    door = self.get_door_obj()
    door_closed = door is not None and not door.is_open

    key_in_first_room = (
        self.objects["key"] is not None and
        self._is_in_first_room(self.objects["key"].position)
    )

    agent_at_goal = (
        self.objects["goal"] is not None and
        tuple(self.get_current_position()) == tuple(self.objects["goal"])
    )

    if door_closed and key_in_first_room and agent_at_goal:
        r += self.option_reward
```

### Helper Method Needed

```python
def _is_in_first_room(self, position: Tuple[int, int]) -> bool:
    """Check if position is in the first room (left of door)."""
    if self.objects["door"] is None:
        return True  # No door means single room
    door_x = self.objects["door"].position[0]
    return position[0] < door_x
```

## Training Considerations

- **Curriculum learning**: Start with simpler reward modes, gradually increase complexity
- **Intrinsic motivation**: Consider adding exploration bonuses
- **Longer episodes**: Increase `max_episode_steps` (recommend 200+)
- **Shaped rewards**: Optional intermediate rewards for:
  - Picking up key (+0.1)
  - Opening door (+0.1)
  - Dropping key in first room (+0.2)
  - Closing door (+0.2)

## Skills Required

| Index | Skill | Used For |
|-------|-------|----------|
| 0 | `get_key` | Pick up key |
| 1 | `open_door` | Open door with key |
| 2 | `close_door` | Close door from either side |
| 3 | `go_to_goal` | Navigate to goal |
| 4 | `drop_key` | Drop key in first room |
| 5-7 | `move_up_1/2/3` | Navigate back to first room |
| 8-10 | `move_right_1/2/3` | Navigate back to first room |
| 11-13 | `move_down_1/2/3` | Navigate back to first room |
| 14-16 | `move_left_1/2/3` | Navigate back to first room |
