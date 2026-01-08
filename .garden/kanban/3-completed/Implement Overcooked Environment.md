We want to add overcooked as an training environment. Ideally this will ultimately be used to post-train models that can actually interact with a human participant. We are particularly interested in how humans could use natural language to specify to an LLM based agent how they want it to behave: which duties it should take on, how it should approach those duties, any norms or preferences it should follow.

We see this as a form of generalisation - the mode must be able to use its world knowledge and reasoning to faithfully follow those instructions using the fundamental skills it learned during post-training.

There are two version of the env. We will build on top of the V2. It is currently only available as a GPU accelerated Jax implementation.

- Original
	- implementation - https://github.com/HumanCompatibleAI/overcooked_ai
	- Jaxmarl implementation - https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked
	- paper https://arxiv.org/abs/1910.05789
- V2
	- implementation with added complexity (layouts, entities, scenario editor) - https://github.com/FLAIROx/JaxMARL/tree/main/jaxmarl/environments/overcooked_v2
	- paper https://arxiv.org/abs/2503.17821

Initially, we will just train the model to learn the environment as a single agent system for simplicity. But we should keep the door open to training the agent in a multi agent environment, either with a non-llm based agent, another copy of itself, or another static llm based agent.

We should also consider if there are extensions to a V3 we could make that would help us test out hypothesis that llms generalise learned skills in order to cooperate with human users.

---

## Implementation Notes

### Status: In Progress

**Branch:** `feat/overcooked_environment`

### Architecture Decisions

1. **JaxMARL V2 with JAX CPU backend** - GPU causes CUDA context issues with async parallel training in verl. CPU-only avoids this.

2. **Partner agent configurable** - Supports `noop`, `random`, or `controlled` partner policies. Default: `noop` (partner stays still).

3. **Single-agent interface** - LLM controls `agent_0`, partner follows configured policy. Multi-agent architecture preserved for future.

### Files Created

```
verl/envs/environments/overcooked/
├── __init__.py           # Action definitions, constants
├── jaxmarl_wrapper.py    # Gymnasium wrapper for JaxMARL V2
├── base.py               # LLMAgentsWrapper (text obs, action extraction)
├── overcooked_env.py     # Factory function for verl integration
└── interactive_play.py   # Manual testing script (WASD controls)
```

### Key Features

- **Text observations** for LLM consumption:
  - Coordinate-based descriptions (agent positions, pot contents, ingredient locations)
  - ASCII grid visualization
  - Configurable via `print_visualization` / `print_coordinates`

- **Rich state decoding**:
  - Ingredient names (onion, tomato, lettuce) instead of indices
  - Pot contents with ingredient counts (e.g., "2x onion, 1x tomato")
  - Cooking timer display
  - Inventory status (e.g., "SOUP (3x onion) on plate")

- **Action space**: 6 discrete actions
  - Movement: right, down, left, up
  - stay (wait in place)
  - interact (pick up / place / use object)

### Integration Points

- Registered in `verl/envs/environments/__init__.py`
- Added to `env_wrapper.py` for observation processing
- Factory function: `make_overcooked_env(env_name, task, config, render_mode)`

### Testing

Interactive play:
```bash
python -m verl.envs.environments.overcooked.interactive_play --layout cramped_room
```

Options:
- `--layout <name>` - Kitchen layout (use `--list-layouts` to see all)
- `--partner noop|random|none` - Partner agent behavior (none = solo mode)
- `--cook-time <n>` - Override cooking duration in ticks (default: 20)
- `--no-grid` / `--no-coords` - Toggle display modes
- `--max-steps <n>` - Episode length

### Config Options (sbatch)

```bash
envs.env_name=overcooked \
envs.overcooked_kwargs.layout_name=cramped_room \
envs.overcooked_kwargs.horizon=200 \
envs.overcooked_kwargs.partner_policy=none \
envs.overcooked_kwargs.pot_cook_time=10 \
envs.overcooked_kwargs.shaped_reward=True \
```

See `docs/configs/overcooked.md` for full documentation.

### JaxMARL Limitations

- **3 ingredients per recipe** - Hardcoded in jaxmarl, cannot be changed
- **Solo mode caveat** - Partner agent still exists physically (blocks movement at start position), just hidden from observations/render
- **Cook time override** - Works via monkey-patching `overcooked_module.POT_COOK_TIME` before env creation

### Remaining Work

- [ ] Create experiment config for training runs
- [ ] Test on cluster with full training pipeline
- [ ] Consider shaped reward tuning
- [ ] Multi-agent training mode (future)
- [ ] Solo mode improvement: move hidden partner to inaccessible location
