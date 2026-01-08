
Recently, we added adaptive entropy along with several new config settings for controlling it. These include strict upper and lower entropy_coeff bounds, as well as soft upper and lower entropy bounds that we aim to keep the token entropy inside by updating the entropy_coeff

What i would like to do, is have it so that the upper and lower entropy bounds decay over time. So that we have a period of allowed high entropy followed by a period of lower entropy in which we encourage more exploitation of learned behavior.

---

## Agreed Design

### Schedule
- **Type**: Cosine decay
- **Warmup**: None (decay starts from step 0)

### Config Approach
- Reuse existing `entropy_low`/`entropy_high` as **initial** values
- Add new `entropy_low_final`/`entropy_high_final` for **target** values
- Enable inferred from presence of `*_final` params (no explicit flag)

### New Config Params
```python
# In verl/workers/config/actor.py
entropy_low_final: Optional[float] = None
entropy_high_final: Optional[float] = None
```

### Behavior
- If `entropy_low_final` is None → static bounds (current behavior)
- If set → cosine decay from `entropy_low` → `entropy_low_final` over training

### Implementation Location
- **Actor** computes scheduled bounds (keeps all entropy logic together)
- **Trainer** passes `global_step` and `total_training_steps` via `meta_info`

### Cosine Decay Formula
```python
progress = global_step / total_training_steps
decay = 0.5 * (1 + cos(pi * progress))  # 1 → 0
current_low = entropy_low_final + decay * (entropy_low - entropy_low_final)
current_high = entropy_high_final + decay * (entropy_high - entropy_high_final)
```

---

## Files to Modify

1. `verl/workers/config/actor.py` - Add `entropy_low_final`, `entropy_high_final`
2. `verl/trainer/ppo/ray_multistep_trainer.py` - Pass step info to actor via meta_info
3. `verl/workers/actor/dp_actor.py` - Compute scheduled bounds, use in adaptive entropy logic