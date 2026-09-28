"""Shared binary-reward MaxRL advantage used by ImageNet experiments."""

import torch


def calculate_binary_maxrl_advantages(
    rewards: torch.Tensor,
    rollout_counts: torch.Tensor,
    rollout_mask: torch.Tensor,
    *,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Match the official MaxRL mean-normalized endpoint convention.

    In particular, an all-wrong group has zero advantage because both the
    reward and empirical success rate are zero. Masked rollout slots remain
    zero, including groups with zero allocated rollouts.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must be a [prompt, rollout] tensor")
    if rollout_counts.ndim != 1 or rollout_counts.shape[0] != rewards.shape[0]:
        raise ValueError("rollout_counts must contain one value per prompt")
    if rollout_mask.shape != rewards.shape:
        raise ValueError("rollout_mask must have the same shape as rewards")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")

    rollout_mask = rollout_mask.to(device=rewards.device, dtype=torch.bool)
    rollout_counts = rollout_counts.to(device=rewards.device)
    success_counts = (rewards * rollout_mask.to(rewards.dtype)).sum(
        dim=1,
        keepdim=True,
    )
    safe_rollout_counts = rollout_counts.clamp_min(1)
    sample_mean = success_counts / safe_rollout_counts[:, None].to(rewards.dtype)
    advantages = (rewards - sample_mean) / (sample_mean + epsilon)
    return advantages * rollout_mask.to(rewards.dtype)
