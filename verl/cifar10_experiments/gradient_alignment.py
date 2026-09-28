"""Exact log-p gradient alignment diagnostics for allocated rollout losses."""

from __future__ import annotations

from collections.abc import Iterable
import math

import torch


def measure_exact_logp_gradient_alignment(
    parameters: Iterable[torch.nn.Parameter],
    exact_nll_loss: torch.Tensor,
) -> dict[str, float]:
    """Compare the accumulated training gradient with exact full-batch NLL.

    ``parameter.grad`` must already contain the realized allocated-rollout loss
    gradient. ``exact_nll_loss`` must be a scalar on the retained forward graph
    for the same minibatch. ``torch.autograd.grad`` is used deliberately so the
    exact reference never changes the gradients consumed by the optimizer.

    Both vectors use the descent-loss convention: the allocated loss gradient
    is compared with ``grad(-mean_q log p_q)``. Multiplying both by ``-1`` gives
    the equivalent log-p ascent vectors and leaves cosine and norm ratio
    unchanged.
    """
    if exact_nll_loss.ndim != 0:
        raise ValueError("exact_nll_loss must be a scalar tensor")
    if not exact_nll_loss.requires_grad:
        raise ValueError("exact_nll_loss must retain an autograd graph")

    trainable = [parameter for parameter in parameters if parameter.requires_grad]
    if not trainable:
        raise ValueError("gradient alignment requires trainable parameters")

    exact_gradients = torch.autograd.grad(
        exact_nll_loss,
        trainable,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    device = exact_nll_loss.device
    dot = torch.zeros((), dtype=torch.float64, device=device)
    allocated_squared_norm = torch.zeros((), dtype=torch.float64, device=device)
    exact_squared_norm = torch.zeros((), dtype=torch.float64, device=device)

    with torch.no_grad():
        for parameter, exact_gradient in zip(trainable, exact_gradients):
            allocated_gradient = parameter.grad
            if allocated_gradient is not None:
                allocated = allocated_gradient.detach()
                allocated_squared_norm.add_(
                    torch.sum(allocated * allocated, dtype=torch.float64)
                )
            else:
                allocated = None

            if exact_gradient is not None:
                exact = exact_gradient.detach()
                exact_squared_norm.add_(
                    torch.sum(exact * exact, dtype=torch.float64)
                )
            else:
                exact = None

            if allocated is not None and exact is not None:
                dot.add_(torch.sum(allocated * exact, dtype=torch.float64))

    allocated_norm = float(torch.sqrt(allocated_squared_norm).cpu())
    exact_norm = float(torch.sqrt(exact_squared_norm).cpu())
    dot_value = float(dot.cpu())
    finite = all(math.isfinite(value) for value in (allocated_norm, exact_norm, dot_value))
    valid = finite and allocated_norm > 0.0 and exact_norm > 0.0

    if valid:
        cosine = dot_value / (allocated_norm * exact_norm)
        cosine = min(1.0, max(-1.0, cosine))
    else:
        cosine = float("nan")
    return {"rollout_allocation/gradient_cosine": float(cosine)}
