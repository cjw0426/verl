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

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("vcr")
class VCRRewardManager(AbstractRewardManager):
    """
    Reward manager that fuses base task reward with progression_reward (p1/p2/p3) computed during rollout.
    - base reward: default_compute_score(data_source, response, ground_truth, extra_info)
    - progression_reward: sample-level scalar in non_tensor_batch["progression_reward"]
    """

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key: str = "data_source",
        progression_weight: float = 1.0,
        gate_by_format: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.progression_weight = progression_weight
        self.gate_by_format = gate_by_format

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print = {}

        for i in range(len(data)):
            item = data[i]
            prompt_ids = item.batch["prompts"]
            prompt_len = prompt_ids.shape[-1]
            valid_prompt_len = item.batch["attention_mask"][:prompt_len].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_len:]

            response_ids = item.batch["responses"]
            valid_response_len = item.batch["attention_mask"][prompt_len:].sum()
            valid_response_ids = response_ids[:valid_response_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            extra_info = item.non_tensor_batch.get("extra_info", {})

            # optional progression reward
            prog = item.non_tensor_batch.get("progression_reward", 0.0)
            try:
                prog = float(prog[0]) if hasattr(prog, "__len__") else float(prog)
            except Exception:
                prog = 0.0

            base_score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                progression_reward=prog,
                progression_weight=self.progression_weight,
            )
            logged_progression_reward = 0.0
            if isinstance(base_score, dict):
                reward = base_score.get("score", base_score.get("overall", 0.0))
                logged_progression_reward = float(base_score.get("progression_reward", 0.0))
                for k, v in base_score.items():
                    if k in {"progression_score", "progression_reward"}:
                        continue
                    reward_extra_info[k].append(v)
            else:
                reward = float(base_score)
            final_score = reward

            reward_tensor[i, valid_response_len - 1] = final_score
            reward_extra_info["base_score"].append(reward)
            reward_extra_info["progression_reward"].append(logged_progression_reward)

            if data_source not in already_print:
                already_print[data_source] = 0
            if already_print[data_source] < self.num_examine:
                already_print[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                print("[base_score]", reward)
                print("[progression_reward]", logged_progression_reward)
                print("[final_score]", final_score)

        if len(data) > 0:
            overall_arr = torch.sum(reward_tensor, dim=-1).detach().cpu().numpy().astype(float)
            mean_overall = float(overall_arr.mean()) if overall_arr.size else float("nan")

            def _mean_from_extra_info(key: str) -> float:
                vals = reward_extra_info.get(key)
                if not vals:
                    return float("nan")
                try:
                    tensor_vals = torch.as_tensor(vals, dtype=torch.float32)
                except Exception:
                    return float("nan")
                if tensor_vals.numel() == 0:
                    return float("nan")
                return float(tensor_vals.mean().item())

            print(
                "[VCR Reward BatchMean] "
                f"overall={mean_overall:.6f} "
                f"format={_mean_from_extra_info('format'):.6f} "
                f"acc={_mean_from_extra_info('accuracy'):.6f} "
                f"progression={_mean_from_extra_info('progression_reward'):.6f}"
            )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        return reward_tensor
