# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Custom reward for Vision-SR1 VCR:
- format: strict match <description>...</description><think>...</think>\boxed{...}
- accuracy: mathruler extract_boxed_content + grade_answer
- progression_reward: optional external scalar (0/0.5/1) from p1/p2/p3
"""

from __future__ import annotations

import os
import re
from itertools import zip_longest
from typing import Any, Dict, Optional

from mathruler.grader import extract_boxed_content, grade_answer

DEFAULT_REWARD_MODE = os.environ.get("REWARD_MODE", "vcr").lower()


def format_reward(predict: str) -> float:
    """
    Strict Vision-SR1 format: <description>...</description><think>...</think>\boxed{...}
    """
    pattern = re.compile(
        r"^\s*<description>.*?</description>\s*<think>.*?</think>\s*\\boxed\{.*?\}\s*$",
        re.DOTALL,
    )
    return 1.0 if pattern.fullmatch(predict) else 0.0


def accuracy_reward(predict: str, ground_truth: str) -> float:
    try:
        answer = extract_boxed_content(predict)
    except Exception:
        return 0.0
    if answer is None:
        return 0.0

    gts = ground_truth if isinstance(ground_truth, (list, tuple, set)) else [ground_truth]
    for gt in gts:
        if gt is None:
            continue
        try:
            if grade_answer(answer, str(gt)):
                return 1.0
        except Exception:
            continue
    return 0.0


def _to_scalar(x, default: float = 0.0) -> float:
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None
    if isinstance(x, (list, tuple)):
        x = x[0] if len(x) > 0 else default
    if np is not None and isinstance(x, np.ndarray):
        x = x.flatten()[0]
    try:
        return float(x)
    except Exception:
        return float(default)


def _score_single(
    predict: str,
    ground_truth: str,
    progression_reward: float = 0.0,
    format_weight: float = 0.1,
    progression_weight: float = 1.0,
    reward_mode: str = "vcr",
) -> Dict[str, float]:
    predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # normalize spacing around tags
    fmt = format_reward(predict)
    acc = accuracy_reward(predict, ground_truth)

    if reward_mode == "grpo":
        overall = acc + format_weight * fmt
        return {
            "overall": overall,
            "format": fmt,
            "accuracy": acc,
            "progression_reward": 0.0,
            "progression_score": 0.0,
        }

    prog_scalar = _to_scalar(progression_reward)
    gated_prog = prog_scalar if fmt > 0 else 0.0
    prog_term = progression_weight * gated_prog
    overall = acc + format_weight * fmt + prog_term
    return {
        "overall": overall,
        "format": fmt,
        "accuracy": acc,
        "progression_reward": gated_prog,
        "progression_score": prog_term,
    }


def compute_score(
    predicts: Any = None,
    ground_truths: Any = None,
    progression_rewards: Any = None,
    format_weight: float = 0.1,
    progression_weight: float = 1.0,
    reward_mode: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """
    Vision-SR1 style reward.

    Supports:
    - single sample: solution_str / ground_truth / progression_reward
    - batch: predicts=list, ground_truths=list, progression_rewards=list

    Returns dict or list[dict] with keys: score(overall), format, accuracy, progression_reward.
    """
    reward_mode = (reward_mode or DEFAULT_REWARD_MODE).lower()
    zero_score = {
        "score": 0.0,
        "overall": 0.0,
        "format": 0.0,
        "accuracy": 0.0,
        "progression_reward": 0.0,
        "progression_score": 0.0,
    }

    # single sample branch (reward_manager.naive / reward_loop)
    if not isinstance(predicts, (list, tuple)):
        predict = predicts or kwargs.get("solution_str") or kwargs.get("response") or kwargs.get("solution")
        gt = ground_truths or kwargs.get("ground_truth") or kwargs.get("gt")
        if predict is None or gt is None:
            return zero_score
        predict = str(predict)
        gt = gt if isinstance(gt, (list, tuple, set)) else str(gt)
        prog = progression_rewards if progression_rewards is not None else kwargs.get("progression_reward", 0.0)
        res = _score_single(
            predict=predict,
            ground_truth=gt,
            progression_reward=prog,
            format_weight=format_weight,
            progression_weight=progression_weight,
            reward_mode=reward_mode,
        )
        return {
            "score": res["overall"],
            "overall": res["overall"],
            "format": res["format"],
            "accuracy": res["accuracy"],
            "progression_reward": res["progression_reward"],
            "progression_score": res["progression_score"],
        }

    # batch branch
    if predicts is None:
        # unexpected, but avoid NoneType iteration; fall back to single-sample path
        predict = kwargs.get("solution_str") or kwargs.get("response") or kwargs.get("solution")
        gt = ground_truths or kwargs.get("ground_truth") or kwargs.get("gt")
        if predict is None or gt is None:
            return zero_score
        return compute_score(
            predicts=predict,
            ground_truths=gt,
            progression_rewards=progression_rewards,
            format_weight=format_weight,
            progression_weight=progression_weight,
            reward_mode=reward_mode,
        )

    preds = [None if p is None else str(p) for p in list(predicts)]

    if ground_truths is None:
        gt_fallback = kwargs.get("ground_truth") or kwargs.get("gt")
        gts = [gt_fallback for _ in preds]
    else:
        gts = list(ground_truths)
        if len(gts) == 1 and len(preds) > 1:
            gts = gts * len(preds)
    gts = [None if gt is None else (gt if isinstance(gt, (list, tuple, set)) else str(gt)) for gt in gts]

    if progression_rewards is None:
        progs = [0.0 for _ in preds]
    else:
        progs = list(progression_rewards)
        if len(progs) == 1 and len(preds) > 1:
            progs = progs * len(preds)
        if len(progs) != len(preds):
            progs = progs[: len(preds)]

    scores = []
    for p, gt, pr in zip_longest(preds, gts, progs, fillvalue=None):
        if p is None or gt is None:
            scores.append(zero_score.copy())
            continue
        res = _score_single(
            predict=p,
            ground_truth=gt,
            progression_reward=pr if pr is not None else 0.0,
            format_weight=format_weight,
            progression_weight=progression_weight,
            reward_mode=reward_mode,
        )
        scores.append(
            {
                "score": res["overall"],
                "overall": res["overall"],
                "format": res["format"],
                "accuracy": res["accuracy"],
                "progression_reward": res["progression_reward"],
                "progression_score": res["progression_score"],
            }
        )

    return scores
