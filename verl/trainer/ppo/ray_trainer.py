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
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import re
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint
from typing import Any, Dict, List, Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup, ResourcePoolManager
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    compute_variance_proxy_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
import verl.utils.torch_functional as VF
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _ensure_numpy_arrays(non_tensor_batch: Dict[str, Any]) -> None:
    """
    Normalize non-tensor payload to numpy arrays (Ray DataProto chunking requires numpy arrays).
    """
    for k, v in list(non_tensor_batch.items()):
        if isinstance(v, list):
            non_tensor_batch[k] = np.array(v, dtype=object)
        elif isinstance(v, tuple):
            non_tensor_batch[k] = np.array(list(v), dtype=object)
        elif isinstance(v, np.ndarray):
            # leave as is
            continue
        else:
            # keep scalars / others untouched
            non_tensor_batch[k] = v


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]
        # Add sum_pi_squared for Optimal Token Baseline
        if adv_estimator in (AdvantageEstimator.OPTIMAL_TOKEN_BASELINE, AdvantageEstimator.TIR_OPTIMAL_TOKEN_BASELINE):
            # Check if sum_pi_squared is available
            assert "sum_pi_squared" in data.batch, (
                "Step-dependent optimal baseline requires sum_pi_squared from actor. "
                "Please set actor.calculate_sum_pi_squared=True in config."
            )
            adv_kwargs["sum_pi_squared"] = data.batch["sum_pi_squared"]
            # Get pre-computed rollout IS weights if available
            rollout_is_weights = data.batch.get("rollout_is_weights", None)
            adv_kwargs["rollout_is_weights"] = rollout_is_weights

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.config)

        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        self.ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    # ---------------------- VCR helper functions ---------------------- #
    def _safe_compute_log_probs(self, data: DataProto) -> DataProto:
        """
        Ray DP split requires non_tensor_batch to be numpy arrays; normalize then call worker.
        """
        _ensure_numpy_arrays(data.non_tensor_batch)
        if hasattr(self.actor_rollout_wg, "compute_log_prob"):
            return self.actor_rollout_wg.compute_log_prob(data)
        if hasattr(self.actor_rollout_wg, "compute_log_probs"):
            return self.actor_rollout_wg.compute_log_probs(data)
        raise AttributeError("RayWorkerGroup has neither compute_log_prob nor compute_log_probs")

    def _build_probability_batch(
        self,
        prompts: List[str],
        answers: List[str],
        assistant_contents: Optional[List[Optional[str]]] = None,
        multi_modal_data: Optional[List[Any]] = None,
        max_length: Optional[int] = None,
        truncation: str = "right",
        temperature: Optional[float] = None,
    ) -> tuple[Optional[DataProto], Optional[torch.Tensor]]:
        """
        Build DataProto for scoring answers conditioned on prompts (supports optional assistant prefill).
        Returns (DataProto, responses_tensor) or (None, None) if inputs invalid.
        """
        if len(prompts) == 0 or len(answers) == 0:
            return None, None

        if assistant_contents is None:
            assistant_contents = [None] * len(prompts)

        prob_temperature = 0.6 if temperature is None else float(temperature)
        answer_ids = [self.tokenizer.encode(a, add_special_tokens=False) for a in answers]
        resp_tensors = [torch.tensor(a, dtype=torch.long) for a in answer_ids]
        responses = pad_sequence(resp_tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        def _is_qwen2_vl_processor() -> bool:
            return (
                self.processor is not None
                and hasattr(self.processor, "image_processor")
                and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__
            )

        def _normalize_mm_items(items: Any) -> Optional[List[Any]]:
            if items is None:
                return None
            if isinstance(items, np.ndarray):
                items = items.tolist()
            if isinstance(items, tuple):
                items = list(items)
            if isinstance(items, list):
                return items if len(items) > 0 else None
            return [items]

        def _resize_last_dim(
            tensor: torch.Tensor,
            target_len: int,
            pad_value: int,
            left_pad: bool,
            trunc_mode: str,
        ) -> torch.Tensor:
            assert trunc_mode in ["left", "right", "middle", "error"]
            seq_len = tensor.shape[-1]
            if seq_len > target_len:
                if trunc_mode == "left":
                    tensor = tensor[..., -target_len:]
                elif trunc_mode == "right":
                    tensor = tensor[..., :target_len]
                elif trunc_mode == "middle":
                    left_half = target_len // 2
                    right_half = target_len - left_half
                    tensor = torch.cat([tensor[..., :left_half], tensor[..., -right_half:]], dim=-1)
                else:
                    raise NotImplementedError(f"{seq_len=} is larger than {target_len=}")
            elif seq_len < target_len:
                tensor = VF.pad_sequence_to_length(
                    tensor,
                    max_seq_len=target_len,
                    pad_token_id=pad_value,
                    left_pad=left_pad,
                )
            return tensor

        def _build_qwen2_vl_position_ids(
            input_ids_1d: torch.Tensor,
            attention_mask_1d: torch.Tensor,
            model_inputs: Dict[str, Any],
        ) -> torch.Tensor:
            from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids_1d,
                image_grid_thw=model_inputs.get("image_grid_thw", None),
                video_grid_thw=model_inputs.get("video_grid_thw", None),
                second_per_grid_ts=model_inputs.get("second_per_grid_ts", None),
                attention_mask=attention_mask_1d,
            )  # (3, seq_len)

            valid_mask = attention_mask_1d.bool()
            text_position_ids = torch.ones(
                (1, input_ids_1d.shape[0]),
                dtype=torch.long,
                device=input_ids_1d.device,
            )
            text_position_ids[0, valid_mask] = torch.arange(
                valid_mask.sum().item(),
                dtype=torch.long,
                device=input_ids_1d.device,
            )

            return torch.cat((text_position_ids, vision_position_ids.to(dtype=torch.long)), dim=0)  # (4, seq_len)

        # If has multimodal data but no processor, abort to avoid shape mismatch
        has_mm = (multi_modal_data is not None) and self.processor is not None

        def _chat_template_text(prompt: str, assistant_content: Optional[str]) -> str:
            messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
            if assistant_content is not None:
                messages.append({"role": "assistant", "content": assistant_content})

            templater = None
            if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
                templater = self.processor
            elif hasattr(self.tokenizer, "apply_chat_template"):
                templater = self.tokenizer

            if templater is not None:
                if assistant_content is not None:
                    # p2/p3 should continue the same assistant turn after prefilled
                    # <description>/<think> rather than opening a fresh assistant turn.
                    try:
                        return templater.apply_chat_template(
                            messages,
                            add_generation_prompt=False,
                            continue_final_message=True,
                            tokenize=False,
                        )
                    except TypeError:
                        text = templater.apply_chat_template(
                            messages,
                            add_generation_prompt=False,
                            tokenize=False,
                        )
                        return re.sub(r"<\|im_end\|>\s*$", "", text)

                return templater.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

            return prompt if assistant_content is None else f"{prompt}\n{assistant_content}"

        # multimodal branch
        if has_mm:
            max_length = max_length or getattr(
                getattr(self, "train_dataloader", None).dataset, "max_prompt_length", self.tokenizer.model_max_length
            )
            input_ids_list, attn_list, pos_list = [], [], []
            new_mm_inputs = []

            for idx, (prompt, ans_ids, assistant_content) in enumerate(zip(prompts, answer_ids, assistant_contents)):
                if multi_modal_data is None or idx >= len(multi_modal_data):
                    has_mm = False
                    break
                raw_mm = multi_modal_data[idx]

                images = videos = None
                if isinstance(raw_mm, np.ndarray):
                    raw_mm = raw_mm.tolist()
                if isinstance(raw_mm, dict):
                    images = raw_mm.get("image", raw_mm.get("images", None))
                    videos = raw_mm.get("video", raw_mm.get("videos", None))
                else:
                    images = raw_mm

                images = _normalize_mm_items(images)
                videos = _normalize_mm_items(videos)

                try:
                    prompt_text = _chat_template_text(prompt, assistant_content)
                    mm_inp = self.processor(
                        text=[prompt_text],
                        images=images,
                        videos=videos,
                        return_tensors="pt",
                    )
                except Exception as e:
                    print(f"[VCR] processor failed on sample {idx}: {e}")
                    has_mm = False
                    break

                model_inputs = dict(mm_inp)

                input_ids = model_inputs.pop("input_ids")[0]
                attention_mask = model_inputs.pop("attention_mask")[0]

                # position ids
                if _is_qwen2_vl_processor():
                    position_ids = _build_qwen2_vl_position_ids(input_ids, attention_mask, model_inputs)
                else:
                    if "position_ids" in model_inputs:
                        position_ids = model_inputs.pop("position_ids")[0]
                        if position_ids.dim() == 2 and position_ids.size(0) == 1:
                            position_ids = position_ids.squeeze(0)
                    else:
                        position_ids = torch.clip(attention_mask.cumsum(dim=0) - 1, min=0)

                ans_tensor = torch.tensor(ans_ids, dtype=torch.long, device=input_ids.device)
                input_ids = torch.cat([input_ids, ans_tensor], dim=0)
                attention_mask = torch.cat([attention_mask, torch.ones_like(ans_tensor)], dim=0)

                if position_ids.shape[-1] > 0:
                    last_pos = position_ids[..., -1:]
                else:
                    if position_ids.dim() == 1:
                        last_pos = torch.zeros((1,), dtype=position_ids.dtype, device=position_ids.device)
                    else:
                        last_pos = torch.zeros((position_ids.size(0), 1), dtype=position_ids.dtype, device=position_ids.device)

                tail_delta = torch.arange(1, len(ans_ids) + 1, device=position_ids.device, dtype=position_ids.dtype)
                if position_ids.dim() == 1:
                    position_ids = torch.cat([position_ids, last_pos + tail_delta], dim=-1)
                else:
                    tail_delta = tail_delta.view(1, -1)
                    position_ids = torch.cat([position_ids, last_pos + tail_delta], dim=-1)

                input_ids = _resize_last_dim(
                    input_ids,
                    target_len=max_length,
                    pad_value=self.tokenizer.pad_token_id,
                    left_pad=True,
                    trunc_mode=truncation,
                )
                attention_mask = _resize_last_dim(
                    attention_mask,
                    target_len=max_length,
                    pad_value=0,
                    left_pad=True,
                    trunc_mode=truncation,
                )
                position_ids = _resize_last_dim(
                    position_ids,
                    target_len=max_length,
                    pad_value=0,
                    left_pad=True,
                    trunc_mode=truncation,
                )

                input_ids_list.append(input_ids)
                attn_list.append(attention_mask)
                pos_list.append(position_ids)
                new_mm_inputs.append(dict(model_inputs))

            if has_mm:
                input_ids = torch.stack(input_ids_list, dim=0)
                attention_mask = torch.stack(attn_list, dim=0)
                position_ids = torch.stack(pos_list, dim=0)
                data = DataProto.from_dict(
                    tensors={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "position_ids": position_ids,
                        "responses": responses,
                    },
                    non_tensors={"multi_modal_inputs": np.array(new_mm_inputs, dtype=object)},
                    meta_info={"temperature": prob_temperature},
                )
                return data, responses

        # text-only branch
        prompt_texts = [_chat_template_text(p, a) for p, a in zip(prompts, assistant_contents)]
        prompt_ids = [self.tokenizer.encode(p, add_special_tokens=False) for p in prompt_texts]
        seq_tensors = [torch.tensor(p + a, dtype=torch.long) for p, a in zip(prompt_ids, answer_ids)]

        input_ids = pad_sequence(seq_tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        if _is_qwen2_vl_processor():
            pos_list = []
            for i in range(input_ids.size(0)):
                pos_list.append(
                    _build_qwen2_vl_position_ids(
                        input_ids[i],
                        attention_mask[i],
                        model_inputs={},
                    )
                )
            position_ids = torch.stack(pos_list, dim=0)  # (bs, 4, seq_len)
        else:
            position_ids = torch.clip(attention_mask.cumsum(dim=1) - 1, min=0)

        if max_length is not None:
            input_ids = _resize_last_dim(
                input_ids,
                target_len=max_length,
                pad_value=self.tokenizer.pad_token_id,
                left_pad=True,
                trunc_mode=truncation,
            )
            attention_mask = _resize_last_dim(
                attention_mask,
                target_len=max_length,
                pad_value=0,
                left_pad=True,
                trunc_mode=truncation,
            )
            position_ids = _resize_last_dim(
                position_ids,
                target_len=max_length,
                pad_value=0,
                left_pad=True,
                trunc_mode=truncation,
            )

        data = DataProto.from_dict(
            tensors={
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
            },
            meta_info={"temperature": prob_temperature},
        )
        return data, responses

    def _compute_answer_probability(
        self,
        batch: DataProto,
        first_round_texts: List[str],
        auxiliary_non_tensor_batch: Optional[Dict[str, Any]] = None,
    ):
        """
        Compute p1/p2/p3 probabilities and progression reward (description -> think).
        Text branch always works; multimodal works when processor is available.
        If question/multimodal fields were popped before generation, pass them via
        auxiliary_non_tensor_batch.
        """
        trainer_cfg = getattr(self.config, "trainer", None)
        score_batch_size = getattr(trainer_cfg, "score_batch_size", 16) or 16
        score_temperature = getattr(trainer_cfg, "score_temperature", 0.6)
        debug_budget = getattr(trainer_cfg, "debug_ans_tokens", 0) or 0
        if score_batch_size <= 0:
            return {}, None

        primary_non_tensor_batch = batch.non_tensor_batch if batch is not None else {}
        auxiliary_non_tensor_batch = auxiliary_non_tensor_batch or {}

        def _as_list(x):
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, list | tuple):
                return list(x)
            return None

        def _extract_text_from_content(content: Any) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            if isinstance(content, np.ndarray):
                return _extract_text_from_content(content.tolist())
            if isinstance(content, list | tuple):
                parts: List[str] = []
                for seg in content:
                    if isinstance(seg, dict):
                        seg_type = str(seg.get("type", "")).lower()
                        if seg_type == "text":
                            parts.append(str(seg.get("text", "")))
                        elif seg_type in {"image", "video"}:
                            parts.append(f"<{seg_type}>")
                        else:
                            parts.append(str(seg.get("content", "")))
                    else:
                        parts.append(str(seg))
                return "".join(parts)
            return str(content)

        def _normalize_question(prompt_obj: Any) -> str:
            if isinstance(prompt_obj, np.ndarray):
                prompt_obj = prompt_obj.tolist()
            if isinstance(prompt_obj, str):
                return prompt_obj
            if isinstance(prompt_obj, dict):
                return _extract_text_from_content(prompt_obj.get("content", "")).strip()
            if isinstance(prompt_obj, list | tuple):
                last_user_text = ""
                first_non_empty = ""
                for message in prompt_obj:
                    if not isinstance(message, dict):
                        continue
                    text = _extract_text_from_content(message.get("content", "")).strip()
                    if text and not first_non_empty:
                        first_non_empty = text
                    if str(message.get("role", "")).lower() == "user" and text:
                        last_user_text = text
                if last_user_text:
                    return last_user_text
                if first_non_empty:
                    return first_non_empty
            return str(prompt_obj).strip()

        def _normalize_ground_truth(gt: Any) -> str:
            if isinstance(gt, np.ndarray):
                gt = gt.tolist()
            if isinstance(gt, list | tuple | set):
                for candidate in gt:
                    if candidate is None:
                        continue
                    candidate_text = str(candidate).strip()
                    if candidate_text:
                        return candidate_text
                return ""
            if gt is None:
                return ""
            return str(gt)

        def _try_get_list_from_sources(key: str) -> Optional[List[Any]]:
            for source in (primary_non_tensor_batch, auxiliary_non_tensor_batch):
                if key not in source:
                    continue
                values = _as_list(source.get(key))
                if values is not None:
                    return values
            return None

        questions = None
        for key in ("question", "problem"):
            questions = _try_get_list_from_sources(key)
            if questions is not None:
                break

        if questions is None:
            prompt_key = getattr(getattr(self.config, "data", None), "prompt_key", None)
            if prompt_key:
                questions = _try_get_list_from_sources(prompt_key)

        if questions is None:
            extra_infos = _try_get_list_from_sources("extra_info")
            if extra_infos is not None:
                questions = []
                for info in extra_infos:
                    if isinstance(info, dict):
                        questions.append(str(info.get("question") or info.get("problem") or ""))
                    else:
                        questions.append("")

        gts = None
        gts = _try_get_list_from_sources("ground_truth")

        if gts is None:
            reward_models = _try_get_list_from_sources("reward_model")
            if reward_models is not None:
                gts = []
                for reward_model in reward_models:
                    if isinstance(reward_model, dict):
                        gts.append(reward_model.get("ground_truth"))
                    else:
                        gts.append(None)

        if questions is None or gts is None:
            return {}, None

        dataset_indices = primary_non_tensor_batch.get("dataset_index")
        if dataset_indices is None:
            dataset_indices = auxiliary_non_tensor_batch.get("dataset_index")
        if isinstance(dataset_indices, np.ndarray):
            dataset_indices = dataset_indices.tolist()

        if len(gts) != len(questions):
            if len(gts) == 0:
                return {}, None
            if len(questions) % len(gts) == 0:
                gts = (gts * (len(questions) // len(gts)))[: len(questions)]
            else:
                gts = gts[: len(questions)]

        def _align_optional_list(values: Optional[List[Any]], target_len: int) -> Optional[List[Any]]:
            if values is None:
                return None
            if len(values) == target_len:
                return values
            if len(values) == 0:
                return None
            if len(values) == 1 and target_len > 1:
                return values * target_len
            if target_len % len(values) == 0:
                return (values * (target_len // len(values)))[:target_len]
            return values[:target_len]

        multi_modal_data = primary_non_tensor_batch.get("multi_modal_data")
        if multi_modal_data is None:
            multi_modal_data = auxiliary_non_tensor_batch.get("multi_modal_data")
        if isinstance(multi_modal_data, np.ndarray):
            multi_modal_data = multi_modal_data.tolist()

        # Fallback: rebuild multi-modal payload from dataset image/video columns
        # (aligned with RL dataset keys used in rollout).
        if multi_modal_data is None:
            image_key = getattr(getattr(self.config, "data", None), "image_key", "images")
            video_key = getattr(getattr(self.config, "data", None), "video_key", "videos")
            image_values = _try_get_list_from_sources(image_key)
            video_values = _try_get_list_from_sources(video_key)
            image_values = _align_optional_list(image_values, len(questions))
            video_values = _align_optional_list(video_values, len(questions))
            if image_values is not None or video_values is not None:
                if image_values is None:
                    image_values = [None] * len(questions)
                if video_values is None:
                    video_values = [None] * len(questions)
                multi_modal_data = [
                    {"image": image_item, "video": video_item}
                    if (image_item is not None or video_item is not None)
                    else None
                    for image_item, video_item in zip(image_values, video_values, strict=True)
                ]

        multi_modal_data = _align_optional_list(multi_modal_data, len(questions))

        # Keep per-sample progression rewards aligned with rollout responses.
        # When rollout.n > 1, first_round_texts has prompt-level samples expanded by repeat-interleave.
        num_outputs = len(first_round_texts)
        num_prompts = len(questions)
        if num_outputs != num_prompts:
            if num_prompts > 0 and num_outputs > num_prompts and num_outputs % num_prompts == 0:
                repeat_factor = num_outputs // num_prompts
                questions = [q for q in questions for _ in range(repeat_factor)]
                gts = [gt for gt in gts for _ in range(repeat_factor)]
                if multi_modal_data is not None:
                    multi_modal_data = [mm for mm in multi_modal_data for _ in range(repeat_factor)]
            else:
                target = min(num_outputs, num_prompts)
                first_round_texts = first_round_texts[:target]
                questions = questions[:target]
                gts = gts[:target]
                if multi_modal_data is not None:
                    multi_modal_data = multi_modal_data[:target]

        captions, thinks = [], []
        for text in first_round_texts:
            text = text or ""
            captions.append(self._extract_description(text) or "")
            m = re.search(r"<think>([\s\S]*?)</think>", text)
            thinks.append(m.group(1).strip() if m else "")

        answers = []
        answer_contents = []
        answer_token_spans: List[tuple[int, int]] = []

        def _find_subsequence(source_ids: List[int], target_ids: List[int]) -> int:
            if len(target_ids) == 0 or len(source_ids) < len(target_ids):
                return -1
            for pos in range(0, len(source_ids) - len(target_ids) + 1):
                if source_ids[pos : pos + len(target_ids)] == target_ids:
                    return pos
            return -1

        def _boxed_content_char_span(text: str) -> Optional[tuple[int, int]]:
            boxed_idx = text.rfind("\\boxed")
            if boxed_idx < 0:
                return None

            tail_start = boxed_idx + len("\\boxed")
            tail = text[tail_start:].lstrip()
            if not tail:
                return None

            # support both "\\boxed{...}" and "\\boxed ..."
            if tail[0] != "{":
                content_start = len(text) - len(tail)
                content_end = len(text)
                return (content_start, content_end)

            open_brace_idx = len(text) - len(tail)
            content_start = open_brace_idx + 1
            depth = 0
            for idx in range(content_start, len(text)):
                ch = text[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    if depth == 0:
                        return (content_start, idx)
                    depth -= 1
            return (content_start, len(text))

        def _content_token_span_from_chars(boxed_text: str, content_text: str) -> tuple[int, int]:
            boxed_ids = self.tokenizer.encode(boxed_text, add_special_tokens=False)
            if len(boxed_ids) == 0:
                return (0, 0)

            char_span = _boxed_content_char_span(boxed_text)
            if char_span is not None:
                char_start, char_end = char_span
                try:
                    encoded = self.tokenizer(
                        boxed_text,
                        add_special_tokens=False,
                        return_offsets_mapping=True,
                    )
                    offsets = encoded.get("offset_mapping", None)
                except Exception:
                    offsets = None

                if offsets is not None:
                    token_start = None
                    token_end = None
                    for token_idx, offset in enumerate(offsets):
                        if offset is None or len(offset) != 2:
                            continue
                        start_char, end_char = int(offset[0]), int(offset[1])
                        if end_char <= start_char:
                            continue
                        if end_char <= char_start:
                            continue
                        if start_char >= char_end:
                            continue
                        if token_start is None:
                            token_start = token_idx
                        token_end = token_idx + 1
                    if token_start is not None and token_end is not None and token_end > token_start:
                        return (token_start, token_end)

            # fallback: token-level subsequence match
            content_ids = self.tokenizer.encode(content_text, add_special_tokens=False)
            content_start = _find_subsequence(boxed_ids, content_ids)
            if content_start >= 0:
                return (content_start, content_start + len(content_ids))
            return (0, len(boxed_ids))

        for gt in gts:
            gt_text = _normalize_ground_truth(gt).strip()
            answer_content = self._extract_answer_content(gt_text)
            if not answer_content:
                answer_content = gt_text
            boxed_answer = self._format_answer(answer_content)
            content_start, content_end = _content_token_span_from_chars(boxed_answer, answer_content)

            answers.append(boxed_answer)
            answer_contents.append(answer_content)
            answer_token_spans.append((content_start, content_end))

        prompts_all = {
            "p1": {"user": [], "assistant": []},
            "p2": {"user": [], "assistant": []},
            "p3": {"user": [], "assistant": []},
        }
        for q, c, t in zip(questions, captions, thinks):
            raw_prompt = _normalize_question(q).strip()
            pinfo = self._build_scoring_prompts(raw_prompt, c, t)
            for name in ("p1", "p2", "p3"):
                prompts_all[name]["user"].append(pinfo[name]["user_text"])
                prompts_all[name]["assistant"].append(pinfo[name].get("assistant"))

        max_model_len = (
            getattr(getattr(self.config, "trainer", None), "score_max_model_len", None)
            or getattr(getattr(self.config.worker, "rollout", None), "max_model_len", None)
            or getattr(self.tokenizer, "model_max_length", None)
            or 8192
        )
        probs_np: List[np.ndarray] = []
        debug_budget_map = {k: debug_budget for k in ("p1", "p2", "p3")}
        debug_records: List[Dict[str, Any]] = []

        def _compute_probs_for_key(key: str) -> np.ndarray:
            mean_lp_chunks: List[torch.Tensor] = []
            total = len(prompts_all[key]["user"])
            for start in range(0, total, score_batch_size):
                end = start + score_batch_size
                if key == "p1" and start < len(first_round_texts):
                    batch_id = start // score_batch_size
                    sample_idx = start
                    print(f"[VCR Score Debug] data_batch={batch_id} sample={sample_idx}")
                    print("[prompt_content]", prompts_all["p1"]["user"][sample_idx])
                    print("[question]", _normalize_question(questions[sample_idx]))
                    print("[model_answer]", first_round_texts[sample_idx])
                    print("[scoring_answer_source]", "ground_truth")
                    print("[scoring_answer_boxed]", answers[sample_idx])
                    print("[scoring_target_content]", answer_contents[sample_idx])
                    print("[scoring_target_span]", answer_token_spans[sample_idx])
                    print("[prompt1_user]", prompts_all["p1"]["user"][sample_idx])
                    print("[prompt1_assistant]", prompts_all["p1"]["assistant"][sample_idx])
                    print("[prompt2_user]", prompts_all["p2"]["user"][sample_idx])
                    print("[prompt2_assistant]", prompts_all["p2"]["assistant"][sample_idx])
                    print("[prompt3_user]", prompts_all["p3"]["user"][sample_idx])
                    print("[prompt3_assistant]", prompts_all["p3"]["assistant"][sample_idx])
                    print("[ground_truth]", _normalize_ground_truth(gts[sample_idx]))

                prob_batch, prob_responses = self._build_probability_batch(
                    prompts=prompts_all[key]["user"][start:end],
                    answers=answers[start:end],
                    assistant_contents=prompts_all[key]["assistant"][start:end],
                    multi_modal_data=multi_modal_data[start:end] if multi_modal_data is not None else None,
                    max_length=max_model_len,
                    truncation="right",
                    temperature=score_temperature,
                )
                if prob_batch is None or prob_responses is None:
                    chunk_size = len(prompts_all[key]["user"][start:end])
                    if chunk_size > 0:
                        mean_lp_chunks.append(torch.full((chunk_size,), float("nan"), dtype=torch.float32))
                    continue
                logprob_proto = self._safe_compute_log_probs(prob_batch)
                log_probs = logprob_proto.batch["old_log_probs"]
                valid_mask = (prob_responses != self.tokenizer.pad_token_id).float()
                target_mask = torch.zeros_like(valid_mask)
                for j in range(log_probs.size(0)):
                    span_start, span_end = answer_token_spans[start + j]
                    if span_end <= span_start:
                        continue
                    left = max(0, int(span_start))
                    right = min(target_mask.size(1), int(span_end))
                    if right > left:
                        target_mask[j, left:right] = 1.0

                has_target = target_mask.sum(dim=-1, keepdim=True) > 0
                mask = torch.where(has_target, target_mask, valid_mask)
                tok_counts = mask.sum(dim=-1)
                mean_lp = torch.full_like(tok_counts, float("nan"), dtype=torch.float32)
                valid = tok_counts > 0
                if valid.any():
                    mean_lp[valid] = (log_probs * mask).sum(dim=-1)[valid] / tok_counts[valid]
                mean_lp_chunks.append(mean_lp)

                if debug_budget_map.get(key, 0) > 0:
                    probs_tok = torch.exp(log_probs)
                    for j in range(log_probs.size(0)):
                        if debug_budget_map.get(key, 0) <= 0:
                            break
                        ans_mask = mask[j].bool()
                        tok_ids = prob_responses[j][ans_mask]
                        if tok_ids.numel() == 0:
                            continue
                        tok_probs = probs_tok[j][ans_mask]
                        tokens = self.tokenizer.convert_ids_to_tokens(tok_ids.tolist())
                        debug_records.append(
                            {
                                "key": key,
                                "idx": start + j,
                                "ground_truth": _normalize_ground_truth(gts[start + j]),
                                "target_content": answer_contents[start + j],
                                "target_span": answer_token_spans[start + j],
                                "tokens": list(zip(tokens, tok_probs.tolist())),
                            }
                        )
                        debug_budget_map[key] = max(0, debug_budget_map.get(key, 0) - 1)

            if not mean_lp_chunks:
                return np.array([], dtype=np.float32)
            mean_lp_all = torch.cat(mean_lp_chunks, dim=0)
            probs = torch.exp(mean_lp_all)
            return probs.cpu().numpy().astype(np.float32)

        for key in ("p1", "p2", "p3"):
            probs_np.append(_compute_probs_for_key(key))

        if len(probs_np) != 3:
            return {}, None
        p1, p2, p3 = probs_np
        if not (len(p1) == len(p2) == len(p3) == len(questions)):
            min_len = min(len(p1), len(p2), len(p3), len(questions))
            if min_len == 0:
                return {}, None
            p1, p2, p3 = p1[:min_len], p2[:min_len], p3[:min_len]
        prog_desc = (p2 > p1).astype(np.float32)
        prog_reason = (p3 > p2).astype(np.float32)
        progression_rewards = 0.5 * prog_desc + 0.5 * prog_reason
        metrics = {
            "p/prob_p1": float(np.nanmean(p1)) if len(p1) else float("nan"),
            "p/prob_p2": float(np.nanmean(p2)) if len(p2) else float("nan"),
            "p/prob_p3": float(np.nanmean(p3)) if len(p3) else float("nan"),
            "p/prog_description": float(np.mean(prog_desc)) if len(prog_desc) else float("nan"),
            "p/prog_reasoning": float(np.mean(prog_reason)) if len(prog_reason) else float("nan"),
            "p/prog_delta": float(np.nanmean((p2 - p1) + (p3 - p2))) if len(p1) else float("nan"),
        }
        if debug_records:
            for rec in debug_records:
                toks = ", ".join([f"{t}:{p:.4g}" for t, p in rec["tokens"]])
                print(
                    f"[VCR Debug] key={rec['key']} sample={rec['idx']} "
                    f"gt={rec['ground_truth']} target={rec['target_content']} span={rec['target_span']} "
                    f"tokens={toks}"
                )
        return metrics, progression_rewards

    def _get_vcr_rollout_instruction(self) -> str:
        trainer_cfg = getattr(self.config, "trainer", None)
        instruction = getattr(trainer_cfg, "vcr_rollout_instruction", None)
        if instruction is None:
            instruction = os.environ.get("VCR_ROLLOUT_INSTRUCTION", None)
        if instruction is None:
            return ""
        return str(instruction).strip()

    @staticmethod
    def _append_instruction_to_user_message(content: Any, instruction: str) -> Any:
        if content is None:
            return instruction

        instruction_marker = "<description>"

        if isinstance(content, str):
            if instruction_marker in content and "\\boxed" in content:
                return content
            content = content.rstrip()
            return f"{content}\n\n{instruction}" if content else instruction

        if isinstance(content, np.ndarray):
            content = content.tolist()

        if isinstance(content, list | tuple):
            normalized = deepcopy(list(content))

            for seg in normalized:
                if isinstance(seg, dict):
                    seg_text = str(seg.get("text", ""))
                    if instruction_marker in seg_text and "\\boxed" in seg_text:
                        return normalized

            text_seg_idx = None
            for idx in range(len(normalized) - 1, -1, -1):
                seg = normalized[idx]
                if isinstance(seg, dict) and str(seg.get("type", "")).lower() == "text":
                    text_seg_idx = idx
                    break

            if text_seg_idx is not None:
                old_text = str(normalized[text_seg_idx].get("text", "")).rstrip()
                normalized[text_seg_idx]["text"] = f"{old_text}\n\n{instruction}" if old_text else instruction
            else:
                normalized.append({"type": "text", "text": instruction})

            return normalized

        return content

    def _inject_vcr_rollout_instruction(self, non_tensor_batch: Dict[str, Any]) -> None:
        if non_tensor_batch is None or "raw_prompt" not in non_tensor_batch:
            return

        instruction = self._get_vcr_rollout_instruction()
        if not instruction:
            return

        raw_prompts = non_tensor_batch.get("raw_prompt")
        if isinstance(raw_prompts, np.ndarray):
            raw_prompts_list = raw_prompts.tolist()
        elif isinstance(raw_prompts, list | tuple):
            raw_prompts_list = list(raw_prompts)
        else:
            return

        updated = []
        for prompt in raw_prompts_list:
            prompt_messages = deepcopy(prompt.tolist() if isinstance(prompt, np.ndarray) else prompt)
            if not isinstance(prompt_messages, list | tuple):
                updated.append(prompt)
                continue

            prompt_messages = deepcopy(list(prompt_messages))
            user_idx = None
            for idx in range(len(prompt_messages) - 1, -1, -1):
                msg = prompt_messages[idx]
                if isinstance(msg, dict) and str(msg.get("role", "")).lower() == "user":
                    user_idx = idx
                    break

            if user_idx is None:
                updated.append(prompt)
                continue

            prompt_messages[user_idx]["content"] = self._append_instruction_to_user_message(
                prompt_messages[user_idx].get("content", ""), instruction
            )
            updated.append(prompt_messages)

        non_tensor_batch["raw_prompt"] = np.array(updated, dtype=object)

    @staticmethod
    def _build_scoring_prompts(raw_prompt: str, desc: str, think: str) -> Dict[str, Dict[str, Any]]:
        instr_p1 = (
            "Provide a single word or phrase answer to the question in \\boxed{}.\n"
            "The output format should be: \\boxed{FINAL ANSWER here}."
        )
        instr_p2 = (
            "Analyze the image/video and produce a self-contained description detailed enough to answer the question. "
            "Wrap the entire description in <description> </description> tags.\n"
            "Next, provide a single word or phrase answer to the question in \\boxed{}.\n"
            "The output format should be: <description> image/video description here </description> \\boxed{FINAL ANSWER here}."
        )
        instr_p3 = (
            "Analyze the image/video and produce a self-contained description detailed enough to answer the question. "
            "Wrap the entire description in <description> </description> tags.\n"
            "Then engage in an internal dialogue and include self-reflection or verification in your reasoning process. "
            "Provide your detailed, step-by-step reasoning and enclose this part within <think> </think> tags.\n"
            "Finally, provide a single word or phrase answer to the question in \\boxed{}.\n"
            "The output format should be: <description> image/video description here </description> <think> reasoning process here </think> \\boxed{FINAL ANSWER here}."
        )

        def user_text(prompt: str, instr: str) -> str:
            return f"{prompt}\n\n{instr}"

        prompts: Dict[str, Dict[str, Any]] = {}
        prompts["p1"] = {"user_text": user_text(raw_prompt, instr_p1), "assistant": None}
        prompts["p2"] = {
            "user_text": user_text(raw_prompt, instr_p2),
            "assistant": f"<description>{desc}</description>",
        }
        prompts["p3"] = {
            "user_text": user_text(raw_prompt, instr_p3),
            "assistant": f"<description>{desc}</description><think>{think}</think>",
        }
        return prompts

    @staticmethod
    def _extract_description(text: str) -> Optional[str]:
        match = re.search(r"<description>([\s\S]*?)</description>", text)
        if not match:
            return ""
        return match.group(1).strip()

    @staticmethod
    def _extract_answer_content(text: Any) -> str:
        if text is None:
            return ""
        text = str(text).strip()
        if not text:
            return ""

        boxed_idx = text.rfind("\\boxed")
        if boxed_idx < 0:
            return ""

        tail = text[boxed_idx + len("\\boxed") :].lstrip()
        if not tail:
            return ""

        if tail[0] != "{":
            return tail.splitlines()[0].strip()

        depth = 0
        content_chars: List[str] = []
        for ch in tail[1:]:
            if ch == "{":
                depth += 1
                content_chars.append(ch)
                continue
            if ch == "}":
                if depth == 0:
                    break
                depth -= 1
                content_chars.append(ch)
                continue
            content_chars.append(ch)
        return "".join(content_chars).strip()

    @staticmethod
    def _format_answer(gt: Any) -> str:
        """Ensure ground-truth is boxed for scoring consistency."""
        gt = "" if gt is None else str(gt).strip()
        if "\\boxed{" in gt or "\\boxed " in gt:
            return gt
        return f"\\boxed{{{gt}}}"

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        reward_for_val: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            reward_for_val: Whether this is for validation
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If reward_for_val=False and sum_reward=True: summed reward_tensor (1D tensor)
            Otherwise: tuple of (reward_tensor, reward_extra_infos_dict)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if not reward_for_val and sum_reward:
                return reward_tensor

            reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
            reward_extra_infos_dict = (
                {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
            )
            return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if reward_for_val:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_infos_dict = result.get("reward_extra_info", {})
            return reward_tensor, reward_extra_infos_dict
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        self._inject_vcr_rollout_instruction(batch.non_tensor_batch)

        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self, merged: bool = False):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            # evaluate using reward_function
            reward_tensor, reward_extra_info = self._compute_or_extract_reward(
                test_batch, reward_fn=self.val_reward_fn, reward_for_val=True
            )
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            print("_merge_validation_results validate result will be merged")
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }
        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _val_metrics_update(self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns):
        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def _merge_validation_results(self, result_a, result_b):
        if result_a is None and result_b is None:
            return {}
        if result_a is None:
            result_a = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}
        if result_b is None:
            result_b = {"data_sources": [], "sample_uids": [], "sample_turns": [], "reward_extra_infos_dict": {}}

        if not result_a.get("data_sources") and not result_b.get("data_sources"):
            return {}

        data_sources = np.concatenate(result_a["data_sources"] + result_b["data_sources"], axis=0)
        sample_uids = result_a["sample_uids"] + result_b["sample_uids"]
        sample_turns = result_a["sample_turns"] + result_b["sample_turns"]

        reward_extra_infos_dict = {}
        all_keys = set(result_a["reward_extra_infos_dict"].keys()) | set(result_b["reward_extra_infos_dict"].keys())
        for key in all_keys:
            list_a = result_a["reward_extra_infos_dict"].get(key, [])
            list_b = result_b["reward_extra_infos_dict"].get(key, [])
            reward_extra_infos_dict[key] = list_a + list_b

        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            actor_rollout_resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[actor_rollout_resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if self.use_rm and not self.use_reward_loop:
            raise RuntimeError("Reward model worker group is not supported, please set use_reward_loop=True")

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create reward loop manager
        if self.use_reward_loop:
            from verl.experimental.reward_loop import RewardLoopManager

            # initalize reward loop manager
            # reward model (colocate or standalone): get resource_pool
            # no reward model: resource_pool = None
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel) if self.use_rm else None
            self.reward_loop_manager = RewardLoopManager(
                config=self.config,
                rm_resource_pool=resource_pool,
            )

        # create async rollout manager and request scheduler
        # Note: mode is always "async" since sync mode is deprecated
        self.async_rollout_mode = True

        # Support custom AgentLoopManager via config
        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            from verl.experimental.agent_loop import AgentLoopManager

        # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
        # agent_reward_loop: streaming reward computation with actor rollout
        # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
        enable_agent_reward_loop = self.use_reward_loop and (
            not self.use_rm or self.config.reward_model.enable_resource_pool
        )
        # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
        # to stream reward computation with actor rollout

        reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None
        self.async_rollout_manager = AgentLoopManager(
            config=self.config,
            worker_group=self.actor_rollout_wg,
            rollout_resource_pool=actor_rollout_resource_pool,
            reward_loop_worker_handles=reward_loop_worker_handles,
        )

        self.checkpoint_manager = CheckpointEngineManager(
            backend=self.config.actor_rollout_ref.rollout.checkpoint_engine.backend,
            trainer=self.actor_rollout_wg,
            replicas=self.async_rollout_manager.rollout_replicas,
        )

        # sleep all replicas to load checkpoint
        self.checkpoint_manager.sleep_replicas()

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst.tolist(), partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            metadata = {"calculate_entropy": False, "compute_loss": False}
            if self.ref_in_actor:
                metadata["no_lora_adapter"] = True
            tu.assign_non_tensor(batch_td, **metadata)
            if self.ref_in_actor:
                output = self.actor_rollout_wg.compute_log_prob(batch_td)
            else:
                output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)

        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                progression_aux_non_tensor_batch = {}
                prompt_key = getattr(getattr(self.config, "data", None), "prompt_key", None)
                image_key = getattr(getattr(self.config, "data", None), "image_key", None)
                video_key = getattr(getattr(self.config, "data", None), "video_key", None)
                aux_keys = {
                    "question",
                    "problem",
                    "raw_prompt",
                    "extra_info",
                    "ground_truth",
                    "reward_model",
                    "multi_modal_data",
                    "dataset_index",
                }
                if prompt_key:
                    aux_keys.add(prompt_key)
                if image_key:
                    aux_keys.add(image_key)
                if video_key:
                    aux_keys.add(video_key)

                for key in aux_keys:
                    if key in batch.non_tensor_batch:
                        progression_aux_non_tensor_batch[key] = batch.non_tensor_batch[key]

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            if curr_step_profile:
                                self.async_rollout_manager.start_profile(global_step=self.global_steps)
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                            self.checkpoint_manager.sleep_replicas()
                            if curr_step_profile:
                                self.async_rollout_manager.stop_profile()

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    # optional VCR progression reward: compute on first-round outputs before repeat
                    reward_mode = getattr(self.config.trainer, "reward_mode", None)
                    first_round_texts = None
                    progression_rewards = None
                    if reward_mode == "vcr":
                        try:
                            first_round_texts = self.tokenizer.batch_decode(
                                gen_batch_output.batch["responses"], skip_special_tokens=True
                            )
                            prob_metrics, progression_rewards = self._compute_answer_probability(
                                batch,
                                first_round_texts,
                                auxiliary_non_tensor_batch=progression_aux_non_tensor_batch,
                            )
                            metrics.update(prob_metrics)
                        except Exception as e:
                            print(f"[VCR] progression reward skipped due to error: {e}")

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                if curr_step_profile:
                                    self.async_rollout_manager.start_profile()
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                                self.checkpoint_manager.sleep_replicas()
                                if curr_step_profile:
                                    self.async_rollout_manager.stop_profile()
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                if not self.use_reward_loop:
                                    rm_scores = self.rm_wg.compute_rm_score(batch)
                                else:
                                    assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                    rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                                batch = batch.union(rm_scores)

                            # Compute or extract reward for REMAX baseline
                            reward_baseline_tensor = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, sum_reward=True
                            )

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if reward_mode == "vcr" and progression_rewards is not None:
                        pr = np.asarray(progression_rewards, dtype=np.float32).reshape(-1)
                        target_len = len(batch)
                        if pr.size == target_len:
                            aligned_pr = pr
                        elif pr.size > 0 and target_len % pr.size == 0:
                            aligned_pr = np.repeat(pr, target_len // pr.size)
                        elif pr.size > target_len:
                            aligned_pr = pr[:target_len]
                        elif pr.size > 0:
                            repeats = int(np.ceil(target_len / pr.size))
                            aligned_pr = np.tile(pr, repeats)[:target_len]
                        else:
                            aligned_pr = np.zeros((target_len,), dtype=np.float32)
                        batch.non_tensor_batch["progression_reward"] = aligned_pr.astype(np.float32)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    # get images_seqlens
                    images_seqlens_all = []
                    for multi_modal_input in batch.non_tensor_batch["multi_modal_inputs"]:
                        if "image_grid_thw" not in multi_modal_input.keys():
                            continue
                        images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
                    batch.meta_info["images_seqlens"] = images_seqlens_all
                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # Compute or extract reward for training
                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                                batch, reward_fn=self.reward_fn, reward_for_val=False
                            )

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                        apply_bypass_mode(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            actor_config = self.config.actor_rollout_ref.actor
                            entropy_agg = agg_loss(
                                loss_mat=entropys,
                                loss_mask=response_masks,
                                loss_agg_mode=actor_config.loss_agg_mode,
                                loss_scale_factor=actor_config.loss_scale_factor,
                            )
                            old_log_prob_metrics = {
                                "actor/entropy": entropy_agg.detach().item(),
                                "perf/mfu/actor_infer": old_log_prob_mfu,
                            }
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            if "routed_experts" in batch.batch and "routed_experts" in old_log_prob.batch:
                                router_mode = getattr(
                                    self.config.actor_rollout_ref.actor.router_replay, "mode", "disabled"
                                )
                                if router_mode == "R2":
                                    batch.batch.pop("routed_experts")
                                else:
                                    old_log_prob.batch.pop("routed_experts")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            ref_log_prob = self._compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            for component_key, metric_name in (
                                ("accuracy", "acc"),
                                ("format", "format"),
                                ("progression_reward", "progression"),
                            ):
                                component_vals = reward_extra_infos_dict.get(component_key, None)
                                if component_vals is None:
                                    continue
                                try:
                                    component_arr = np.asarray(component_vals, dtype=np.float32)
                                except Exception:
                                    continue
                                if component_arr.size == 0:
                                    continue
                                metrics[f"training/reward_components/{metric_name}/mean"] = float(
                                    np.nanmean(component_arr)
                                )

                        metrics["training/reward_components/overall/mean"] = (
                            reward_tensor.sum(dim=-1).float().mean().item()
                        )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            actor_output = self._update_actor(batch)

                        # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        # Check if the conditions for saving a checkpoint are met.
                        # The conditions include a mandatory condition (1) and
                        # one of the following optional conditions (2/3/4):
                        # 1. The save frequency is set to a positive value.
                        # 2. It's the last training step.
                        # 3. The current step number is a multiple of the save frequency.
                        # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, color="green"):
                                self._save_checkpoint()

                        # update weights from trainer to rollout
                        with marked_timer("update_weights", timing_raw, color="red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # compute variance proxy metrics
                gradient_norm = metrics.get("actor/grad_norm", None)
                metrics.update(compute_variance_proxy_metrics(batch=batch, gradient_norm=gradient_norm))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
