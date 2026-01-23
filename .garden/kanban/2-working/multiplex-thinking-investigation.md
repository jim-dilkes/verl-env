# Multiplex Thinking Investigation

**Branch:** `multiplex-reasoning-investigation`
**Created:** 2025-01-23
**Status:** In Progress

## Goal
Implement a test version of Multiplex Thinking (Tang et al. 2026) - token-wise branch-and-merge reasoning where multiple candidate tokens are sampled and combined via weighted embedding averaging.

## Acceptance Criteria
- [ ] Weighted embeddings are actually fed back on next decode step (not just fields exist)
- [ ] End-to-end generation produces different outputs with soft_thinking enabled vs disabled
- [ ] Gradient flows from loss through topk_probs to model parameters

## Scope
- [x] Clone sglang 0.5.5 for patching
- [x] Patch sglang sampler with soft_thinking support (Dirichlet noise, top-K extraction)
- [x] Patch sglang vocab_parallel_embedding with weighted_forward methods
- [x] Patch sglang ForwardBatch and LogitsProcessorOutput with topk fields
- [x] Create MultiplexQwen3ForCausalLM for sglang (inference)
- [x] Create MultiplexQwen3ForCausalLM wrapper for transformers (training)
- [x] Create test scripts verifying patches
- [x] **CRITICAL: Wire topk propagation in sglang decode loop**
- [x] **FIX: Discrete token selection bug (use argmax not index 0)**
- [ ] Document sampling semantics (top-p/k/min-p interaction)
- [ ] Set up external deps as submodules or documented clone script
- [ ] Integrate with dp_actor.py for 3D log-probs
- [ ] Add compute_multiplex_policy_loss() to core_algos.py
- [ ] Pass soft_thinking params through sglang_rollout.py
- [ ] End-to-end test on login node

## Working Notes

### 2025-01-23: Critical Bug Fixes

**Fixed discrete token selection:**
- Changed from `topk_indices[:, 0]` to `topk_indices.gather(1, topk_probs.argmax(dim=-1, keepdim=True))`
- Now correctly selects the highest-probability token from the sampled K tokens

**Fixed end-to-end propagation - complete wiring:**
1. `sampler.py` → sets `logits_output.topk_probs/topk_indices`
2. `scheduler_output_processor_mixin.py` → stores to `req.output_topk_p/output_topk_index`
3. `schedule_batch.py` → collects from reqs into `ModelWorkerBatch.topk_probs/topk_indices`
4. `forward_batch_info.py` → transfers to device into `ForwardBatch.topk_probs/topk_indices`
5. `qwen3_multiplex.py` → checks ForwardBatch and calls `weighted_forward()`

All 8 tests still pass.

### 2025-01-23: Code Review Findings (CRITICAL)

**Bugs identified:**
1. ~~**Missing end-to-end propagation**~~ FIXED
2. ~~**Discrete token selection bug**~~ FIXED
3. **Soft-thinking bypasses top-p/top-k/min-p** - Branched before those filters are applied. Need to document whether this is intentional or apply filters first.

**Gaps identified:**
- No acceptance criteria beyond "tests pass"
- External clones (`external/sglang-multiplex/`, `external/Multiplex-Thinking/`) are untracked - not reproducible
- PRD has more params (gumbel, max_topk, entropy masking) - only implemented minimal subset
- "Qwen3" naming but wrapper is model-agnostic and tests use Qwen2.5
- Tests are string-based (fragile), not pytest integrated, require network

**Decisions needed:**
- Sampling semantics: Should soft thinking respect top-p/top-k/min-p?
- External deps: Submodules vs clone script with pinned SHAs?

### 2025-01-23: Initial Implementation
- Cloned sglang v0.5.5 to `external/sglang-multiplex/`
- Cloned Multiplex-Thinking reference repo to `external/Multiplex-Thinking/`
- Applied patches to sglang:
  - `sampler.py`: Added `enable_soft_thinking`, `soft_thinking_topk`, `dirichlet_alpha` params
  - `vocab_parallel_embedding.py`: Added `weighted_forward()` and `weighted_forward_tp()`
  - `forward_batch_info.py`: Added `topk_probs`, `topk_indices` fields
  - `logits_processor.py`: Added `topk_probs`, `topk_indices`, `entropy` fields
  - Created `qwen3_multiplex.py` model for sglang
- Created `verl/models/transformers/qwen3_multiplex.py` wrapper for training
- All 8 patch verification tests pass
- Qwen wrapper integration test passes (gradient flow, one-hot equivalence confirmed)

### Key Files
- **sglang patches:** `external/sglang-multiplex/python/sglang/srt/`
- **transformers wrapper:** `verl/models/transformers/qwen3_multiplex.py`
- **tests:** `experiments/multiplex/test_multiplex_sglang.py`, `test_qwen3_wrapper.py`
- **PRD:** `/Users/jim/Documents/PhD/Research Projects/4) Exploration in SDM for LLMs/Implementation/PRD - Multiplex Thinking Integration.md`

### Architecture (UPDATED - now fully wired)
```
ROLLOUT (sglang):
  Sampler → logits_output.topk_probs/indices
    → scheduler stores to req.output_topk_p/index
    → ScheduleBatch collects into ModelWorkerBatch
    → ForwardBatch.init_new transfers to device
    → MultiplexQwen3Model.forward → weighted_forward()

TRAINING (transformers):
  dp_actor → topk_probs, topk_indices → MultiplexQwen3ForCausalLM → compute_weighted_embeddings()
```

## Original Notes
PRD-driven implementation of multiplex thinking for verl. Token-wise branch-and-merge approach where K tokens are sampled per step, combined into continuous "multiplex token" via weighted embedding average, then fed back for next prediction.
