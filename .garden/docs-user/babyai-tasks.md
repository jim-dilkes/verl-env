# BabyAI Tasks Reference

BabyAI is a platform for studying grounded language acquisition in grid-world environments. The agent receives natural language instructions (e.g., "go to the red ball") and must execute them through a sequence of actions.

## Actions

```
turn left    - Rotate 90 degrees counter-clockwise
turn right   - Rotate 90 degrees clockwise
go forward   - Move one tile in facing direction
pick up      - Pick up object directly in front
drop         - Drop held object in front
toggle       - Interact with object in front (open doors, etc.)
```

## Task Naming Convention

BabyAI task names follow patterns:
- `S{X}` = room/grid size X
- `N{Y}` = Y distractor objects
- `R{Z}` = Z rooms
- `Dist` = distractor objects present
- `Loc` = location may be specified in instruction
- `Local` = single room (no doors)

Example: `GoToLocalS5N2` = Go to object in a 5x5 single room with 2 distractors.

## Task Categories

### Navigation (GoTo)

| Task | Description | Complexity |
|------|-------------|------------|
| **GoToLocal** | Go to object in single room, no doors | Easiest |
| **GoToObj** | Go to object, may require opening doors | Medium |
| **GoToObjMaze** | Go to object in a maze with multiple rooms | Hard |

Mission format: `"go to the {color} {type}"`

### Manipulation (Pickup, PutNext)

| Task | Description | Complexity |
|------|-------------|------------|
| **PickupLoc** | Pick up object (location may be specified) | Easy |
| **PickupDist** | Pick up object with distractors present | Easy |
| **PutNextLocal** | Pick up object A, place next to object B | Medium |

Mission format: `"pick up the {color} {type}"` or `"put the {color} {type} next to the {color} {type}"`

### Door/Key Tasks (Unlock)

| Task | Description | Complexity |
|------|-------------|------------|
| **UnlockLocal** | Get key, unlock and open door (single room) | Medium |
| **UnlockLocalDist** | Same as UnlockLocal with distractors | Medium |
| **UnlockPickup** | Unlock door, enter room, pick up box | Hard |
| **BlockedUnlockPickup** | Move blocking ball, unlock door, pick up box | Hard |
| **UnlockToUnlock** | Unlock door to get key to unlock another door | Hardest |

Mission format: `"open the door"` or `"pick up the {color} box"`

## Skill Progression

Tasks build on each other in complexity:

```
1. GoToLocal          - Navigation only
2. Pickup             - Navigation + pick up
3. PutNext            - Navigation + pick up + place
4. UnlockLocal        - Navigation + key + door
5. UnlockPickup       - Key + door + pickup
6. BlockedUnlockPickup - Unblock + key + door + pickup
7. UnlockToUnlock     - Chained key dependencies
```

## Evaluation Suite (BAI_evals)

| Task | max_steps |
|------|-----------|
| `BabyAI-GoToLocalS5N2-v0` | 20 |
| `BabyAI-GoToObjMazeS4R2-v0` | 60 |
| `BabyAI-PickupLoc-v0` | 20 |
| `BabyAI-PickupDist-v0` | 20 |
| `BabyAI-PutNextLocalS5N3-v0` | 20 |
| `BabyAI-UnlockLocalDist-v0` | 20 |
| `BabyAI-UnlockPickup-v0` | 20 |
| `BabyAI-BlockedUnlockPickup-v0` | 40 |
| `BabyAI-UnlockToUnlock-v0` | 60 |

## References

- [MiniGrid BabyAI Environments](https://minigrid.farama.org/environments/babyai/index.html)
- [BabyAI Paper (ICLR 2019)](https://openreview.net/pdf?id=rJeXCo0cYX)
