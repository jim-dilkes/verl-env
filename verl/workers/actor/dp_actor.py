# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import math
import os

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty, kl_penalty_forward
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@dataclass
class ActorForwardOutput:
    log_probs: torch.Tensor
    entropy_full: Optional[torch.Tensor] = None
    entropy_top_p: Optional[torch.Tensor] = None


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Adaptive entropy requires all five params (validated in config __post_init__)
        adaptive_entropy_fully_configured = (
            self.config.entropy_coeff_low is not None
            and self.config.entropy_coeff_high is not None
            and self.config.entropy_low is not None
            and self.config.entropy_high is not None
            and self.config.entropy_coeff_lr > 0
        )
        if adaptive_entropy_fully_configured:
            self.use_adaptive_entropy = True
            # Store as plain attribute; checkpointed via extra_state_dict (not model buffer)
            self.adaptive_entropy_coeff = float(self.config.entropy_coeff)
        else:
            self.use_adaptive_entropy = False

        # Entropy bounds decay: enabled if both initial AND final bounds are set
        self.use_entropy_bounds_decay = (
            self.config.entropy_low is not None
            and self.config.entropy_high is not None
            and self.config.entropy_low_final is not None
            and self.config.entropy_high_final is not None
        )
        if self.use_entropy_bounds_decay:
            if self.config.entropy_low_final > self.config.entropy_low:
                logger.warning(
                    f"entropy_low_final ({self.config.entropy_low_final}) > entropy_low ({self.config.entropy_low}). "
                    "Bounds will increase over training instead of decay."
                )
            if self.config.entropy_high_final > self.config.entropy_high:
                logger.warning(
                    f"entropy_high_final ({self.config.entropy_high_final}) > entropy_high ({self.config.entropy_high}). "
                    "Bounds will increase over training instead of decay."
                )

    def _get_entropy_bounds(self, global_step: int, total_steps: int):
        """Compute entropy bounds, with optional cosine decay over training.

        Returns:
            (entropy_low, entropy_high): Current bounds for this training step.
        """
        if not self.use_entropy_bounds_decay or total_steps <= 0:
            return self.config.entropy_low, self.config.entropy_high

        progress = min(global_step / total_steps, 1.0)
        # Cosine decay: 1 -> 0 over training
        decay = 0.5 * (1 + math.cos(math.pi * progress))

        entropy_low = self.config.entropy_low_final + decay * (
            self.config.entropy_low - self.config.entropy_low_final
        )
        entropy_high = self.config.entropy_high_final + decay * (
            self.config.entropy_high - self.config.entropy_high_final
        )
        return entropy_low, entropy_high

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False, entropy_top_p=1
    ) -> ActorForwardOutput:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        entropy_top_p = 1 if entropy_top_p is None else entropy_top_p
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        effective_clamp = max(0.0, min(1.0, 1.0 - float(entropy_top_p)))
        need_top_p_entropy = calculate_entropy and effective_clamp > 0

        if self.use_fused_kernels and need_top_p_entropy:
            raise NotImplementedError(
                "entropy_top_p < 1 requires logits; disable fused kernels to use this feature."
            )

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy_full = None
            entropy_top = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )
                        if need_top_p_entropy:
                            entropy_rmpad_top = verl_F.clamped_entropy_from_logits(
                                logits_rmpad, clamp_p=effective_clamp
                            )
                        else:
                            entropy_rmpad_top = None

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if need_top_p_entropy and entropy_rmpad_top is not None:
                            entropy_rmpad_top = gather_outputs_and_unpad(
                                entropy_rmpad_top,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]
                        if need_top_p_entropy and entropy_rmpad_top is not None:
                            entropy_rmpad_top = entropy_rmpad_top[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if need_top_p_entropy:
                        if entropy_rmpad_top is None:
                            entropy_top = full_entropy
                        else:
                            entropy_top_padded = pad_input(
                                hidden_states=entropy_rmpad_top.unsqueeze(-1),
                                indices=indices,
                                batch=batch_size,
                                seqlen=seqlen,
                            )
                            entropy_top = entropy_top_padded
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy_full = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if entropy_top is not None:
                        entropy_top = entropy_top.squeeze(-1)[:, -response_length - 1 : -1]
                    else:
                        entropy_top = entropy_full
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy_full = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)
                    entropy_top = entropy_full

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_full = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy_full = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

                        if need_top_p_entropy:
                            entropy_top = verl_F.clamped_entropy_from_logits(logits, clamp_p=effective_clamp)
                        else:
                            entropy_top = entropy_full

            return ActorForwardOutput(
                log_probs=log_probs,
                entropy_full=entropy_full,
                entropy_top_p=entropy_top if calculate_entropy else None,
            )

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                forward_out = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(forward_out.log_probs)
            if calculate_entropy:
                entropy_lst.append(forward_out.entropy_top_p)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # Extract step info for entropy bounds decay
        global_step = data.meta_info.get("global_step", 0)
        total_training_steps = data.meta_info.get("total_training_steps", 1)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        # DIME parallel optimisation: include teacher data if present
        dime_keys = ["teacher_input_ids", "teacher_attention_mask",
                     "teacher_position_ids", "teacher_old_log_probs", "has_instruction"]
        for k in dime_keys:
            if k in data.batch.keys():
                select_keys.append(k)

        dime_enabled = "teacher_input_ids" in data.batch.keys()
        dime_alpha = data.meta_info.get('dime_alpha', 0.5) if dime_enabled else 0
        dime_kl_beta_teacher = data.meta_info.get('dime_kl_beta_teacher', 0.0) if dime_enabled else 0
        dime_kl_beta_student = data.meta_info.get('dime_kl_beta_student', 0.0) if dime_enabled else 0

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # DIME: sort so non-instructed samples come first → early micro-batches skip
        # redundant teacher forward pass. Safe: _balance_batch already reorders,
        # data.reorder() moves all tensors together, micro-batches are independent.
        if dime_enabled:
            sort_idx = torch.argsort(data.batch['has_instruction'].long())  # False(0) first
            data.reorder(sort_idx)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # Accumulators for mini-batch level adaptive entropy update
                mini_batch_entropy_sum = 0.0
                mini_batch_token_count = 0
                mini_batch_entropy_ok_count = 0

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.adaptive_entropy_coeff if self.use_adaptive_entropy else self.config.entropy_coeff
                    entropy_top_p = getattr(self.config, "entropy_top_p", 1.0)
                    loss_agg_mode = self.config.loss_agg_mode

                    # If coeff ever becomes non-finite, reset to config value
                    if not torch.isfinite(torch.tensor(entropy_coeff)):
                        micro_batch_metrics["actor/entropy_coeff_reset"] = 1.0
                        entropy_coeff = float(self.config.entropy_coeff)
                    else:
                        micro_batch_metrics["actor/entropy_coeff_reset"] = 0.0

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if not dime_enabled:
                        # === Standard PPO path (unchanged) ===
                        forward_out = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            entropy_top_p=entropy_top_p,
                        )
                        log_prob = forward_out.log_probs
                        entropy = forward_out.entropy_top_p
                        entropy_full = forward_out.entropy_full

                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs
                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)
                    else:
                        # === DIME parallel optimisation path ===
                        # Student forward (base prompts — current input_ids)
                        student_out = self._forward_micro_batch(
                            model_inputs,
                            temperature=temperature,
                            calculate_entropy=calculate_entropy,
                            entropy_top_p=entropy_top_p,
                        )
                        student_log_prob = student_out.log_probs
                        entropy = student_out.entropy_top_p
                        entropy_full = student_out.entropy_full

                        # Teacher forward (instructed prompts — skip if no instructed in micro-batch)
                        has_inst = model_inputs['has_instruction']
                        if has_inst.any():
                            # Shallow copy safe: _forward_micro_batch reads but doesn't mutate the dict
                            teacher_inputs = {**model_inputs}
                            teacher_inputs['input_ids'] = model_inputs['teacher_input_ids']
                            teacher_inputs['attention_mask'] = model_inputs['teacher_attention_mask']
                            teacher_inputs['position_ids'] = model_inputs['teacher_position_ids']
                            teacher_out = self._forward_micro_batch(
                                teacher_inputs,
                                temperature=temperature,
                                calculate_entropy=False,
                            )
                            teacher_log_prob = teacher_out.log_probs
                        else:
                            teacher_log_prob = student_log_prob

                        # Student PG loss (base prompt old_log_probs)
                        student_pg_loss, student_pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=student_log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )

                        # Teacher PG loss (instructed prompt old_log_probs)
                        # For non-instructed samples: teacher_old_log_probs == old_log_probs by construction
                        # (inject_focus_into_obs skips None entries, so prompts are identical)
                        teacher_old_log_prob = model_inputs['teacher_old_log_probs']
                        teacher_pg_loss, teacher_pg_metrics = policy_loss_fn(
                            old_log_prob=teacher_old_log_prob,
                            log_prob=teacher_log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )

                        # Combined PG:
                        # - No instructed samples in this micro-batch: standard PPO loss
                        # - Otherwise: α*teacher + (1-α)*student
                        if has_inst.any():
                            pg_loss = dime_alpha * teacher_pg_loss + (1 - dime_alpha) * student_pg_loss
                        else:
                            pg_loss = student_pg_loss

                        # KL terms (instructed-only mask for correct normalisation)
                        # Use masked response_mask so agg_loss normalises by instructed tokens only
                        has_inst_f = has_inst.float().unsqueeze(-1)  # (bsz, 1)
                        inst_response_mask = response_mask * has_inst_f

                        if dime_kl_beta_teacher > 0 and has_inst.any():
                            # kl_penalty_forward(logprob=A, ref_logprob=B, 'k3') ≈ D_KL(πA || πB)
                            # Here: D_KL(π^T || sg(π^S)) — gradient through teacher, pulls teacher→student
                            kl_t = kl_penalty_forward(teacher_log_prob, student_log_prob.detach(), 'k3')
                            kl_t_agg = agg_loss(kl_t, inst_response_mask, loss_agg_mode)
                            pg_loss = pg_loss + dime_kl_beta_teacher * kl_t_agg
                            micro_batch_metrics["dime/kl_teacher"] = kl_t_agg.detach().item()

                        if dime_kl_beta_student > 0 and has_inst.any():
                            # D_KL(π^S || sg(π^T)) — gradient through student, pulls student→teacher
                            kl_s = kl_penalty_forward(student_log_prob, teacher_log_prob.detach(), 'k3')
                            kl_s_agg = agg_loss(kl_s, inst_response_mask, loss_agg_mode)
                            pg_loss = pg_loss + dime_kl_beta_student * kl_s_agg
                            micro_batch_metrics["dime/kl_student"] = kl_s_agg.detach().item()

                        # Log DIME metrics
                        teacher_loss_val = teacher_pg_loss.detach().item()
                        student_loss_val = student_pg_loss.detach().item()
                        micro_batch_metrics["dime/teacher_loss"] = teacher_loss_val
                        micro_batch_metrics["dime/student_loss"] = student_loss_val
                        micro_batch_metrics["dime/alpha"] = dime_alpha
                        for k, v in teacher_pg_metrics.items():
                            micro_batch_metrics[f"dime/teacher_{k.replace('actor/', '')}"] = v
                        for k, v in student_pg_metrics.items():
                            micro_batch_metrics[f"dime/student_{k.replace('actor/', '')}"] = v

                        # Student log_prob for downstream entropy/ref-KL (ref_log_prob was computed
                        # on base prompts at rollout time, so student comparison is correct)
                        log_prob = student_log_prob
                        # Populate actor/* metrics from student (already in dime/student_* above)
                        micro_batch_metrics.update(student_pg_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_ok = False  # Assume failure, explicitly mark success

                        if response_mask.sum() == 0:
                            micro_batch_metrics["actor/entropy_mask_empty"] = 1.0
                        else:
                            micro_batch_metrics["actor/entropy_mask_empty"] = 0.0
                            entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                            micro_batch_metrics["actor/entropy"] = float(entropy_agg.detach().item())
                            micro_batch_metrics["actor/entropy_loss"] = micro_batch_metrics["actor/entropy"]
                            if entropy_full is not None:
                                entropy_full_agg = agg_loss(
                                    loss_mat=entropy_full, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                                )
                                micro_batch_metrics["actor/entropy_full"] = float(entropy_full_agg.detach().item())
                            else:
                                entropy_full_agg = None

                            if not torch.isfinite(entropy_agg).all():
                                micro_batch_metrics["actor/entropy_nan"] = 1.0
                            else:
                                micro_batch_metrics["actor/entropy_nan"] = 0.0
                                entropy_ok = True  # Only success path

                        if entropy_ok and entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                        # Accumulate entropy for mini-batch level adaptive update
                        if entropy_ok:
                            token_count = int(response_mask.sum().item())
                            mini_batch_entropy_sum += float(entropy_agg.detach().item()) * token_count
                            mini_batch_token_count += token_count
                            mini_batch_entropy_ok_count += 1

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                # Mini-batch level adaptive entropy update (once per optimizer step)
                mini_batch_metrics = {}
                if self.use_adaptive_entropy and mini_batch_entropy_ok_count > 0:
                    # Log the coefficient only when entropy was actually applied to loss
                    mini_batch_metrics["actor/entropy_coeff_used"] = float(self.adaptive_entropy_coeff)

                    # Compute weighted average entropy for the mini-batch
                    if mini_batch_token_count > 0:
                        mini_batch_entropy_avg = mini_batch_entropy_sum / mini_batch_token_count
                    else:
                        mini_batch_entropy_avg = mini_batch_entropy_sum / mini_batch_entropy_ok_count

                    # Get current entropy bounds (may be decayed over training)
                    current_entropy_low, current_entropy_high = self._get_entropy_bounds(
                        global_step, total_training_steps
                    )

                    # lower_violation ≤ 0 when entropy < low (need to increase coeff)
                    # upper_violation ≥ 0 when entropy > high (need to decrease coeff)
                    lower_violation = min(mini_batch_entropy_avg - current_entropy_low, 0.0)
                    upper_violation = max(mini_batch_entropy_avg - current_entropy_high, 0.0)
                    entropy_coeff_update = self.config.entropy_coeff_lr * (lower_violation + upper_violation)
                    new_coeff = self.adaptive_entropy_coeff - entropy_coeff_update
                    self.adaptive_entropy_coeff = float(
                        min(max(new_coeff, self.config.entropy_coeff_low), self.config.entropy_coeff_high)
                    )
                    mini_batch_metrics["actor/mini_batch_entropy_avg"] = float(mini_batch_entropy_avg)
                    # Log current bounds if decay is enabled
                    if self.use_entropy_bounds_decay:
                        mini_batch_metrics["actor/entropy_low_current"] = float(current_entropy_low)
                        mini_batch_metrics["actor/entropy_high_current"] = float(current_entropy_high)

                grad_norm = self._optimizer_step()
                mini_batch_metrics["actor/grad_norm"] = grad_norm.detach().item()
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()

        # Log final adaptive entropy coefficient for this training step
        if self.use_adaptive_entropy:
            metrics["actor/entropy_coeff_final"] = [self.adaptive_entropy_coeff]

        return metrics
