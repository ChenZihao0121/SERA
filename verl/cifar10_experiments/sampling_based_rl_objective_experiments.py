#!/usr/bin/env python3
"""SERA ImageNet-256 main experiment, retaining the MaxRL training protocol.

The allocator receives exact detached correct-class probabilities. The update
still uses sampled labels and the centered binary MaxRL advantage. Training
loss is normalized by the full minibatch size times the baseline N0, so
inactive images contribute zero while remaining in the batch average.
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from verl.cifar10_experiments.gradient_alignment import measure_exact_logp_gradient_alignment
from verl.cifar10_experiments.maxrl_advantage import calculate_binary_maxrl_advantages
from verl.trainer.ppo.SERA import allocate_rollouts_from_probabilities


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")



class HFImageNet(torch.utils.data.Dataset):
    def __init__(self, split, transform, hf_ds):
        self.ds = hf_ds[split]
        self.transform = transform

        print("Number of classes: ", self.get_num_classes())

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx):
        img = self.ds[idx]["image"]   # PIL image
        label = self.ds[idx]["label"]
        img = self.transform(img)
        
        return img, label
    
    def get_num_classes(self) -> int:
        return self.ds.features["label"].num_classes



class IndexedDataset(torch.utils.data.Dataset):
    """Attach a stable dense index to training samples for epoch estimators."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx



def get_dataloaders(dataset_name, data_dir, batch_size=256, num_workers=32):
    """Keep MaxRL's ImageNet source, transforms, ordering, and batch sizes."""
    if dataset_name != "imagenet256":
        raise ValueError("The main experiment uses ImageNet-256")
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    # As in MaxRL, Hugging Face uses HF_HOME/HF_DATASETS_CACHE; data_dir is
    # retained for launcher compatibility and does not override that cache.
    hf_ds = load_dataset("benjamin-paine/imagenet-1k-256x256")
    train_set = HFImageNet("train", train_transform, hf_ds)
    test_set = HFImageNet("validation", test_transform, hf_ds)
    train_loader = DataLoader(
        IndexedDataset(train_set), batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_set, batch_size=1000, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, test_loader


def calculate_sera_oracle_loss(
    logits,
    targets,
    baseline_n,
    n_min,
    n_max,
    activation_useful_threshold=0.05,
    metrics=None,
    backward_state=None,
    epsilon=1e-6,
):
    """Oracle SERA loss with rollout-count correction and full-batch averaging."""
    if not bool(torch.isfinite(logits).all()):
        nonfinite_count = int((~torch.isfinite(logits)).sum().item())
        raise RuntimeError(
            f"oracle received {nonfinite_count} non-finite model logits"
        )

    log_probs_all = F.log_softmax(logits, dim=1)
    batch_size = targets.shape[0]

    # Compute both ideal p(correct) and the diagnostic rollout distribution in
    # float64, so probabilities below float32 range are not silently rounded
    # to zero before allocation or sampling.
    oracle_log_probs_all = F.log_softmax(
        logits.detach().to(torch.float64),
        dim=1,
    )
    oracle_log_prob = oracle_log_probs_all.gather(
        1,
        targets[:, None],
    ).squeeze(1)
    if not bool(torch.isfinite(oracle_log_prob).all()):
        nonfinite_count = int((~torch.isfinite(oracle_log_prob)).sum().item())
        raise RuntimeError(
            f"oracle log p(correct) contains {nonfinite_count} non-finite values"
        )

    oracle_sampling_probs = torch.exp(oracle_log_probs_all)
    raw_prob = oracle_sampling_probs.gather(
        1,
        targets[:, None],
    ).squeeze(1).cpu().numpy()
    lower_probability = np.finfo(np.float64).tiny
    upper_probability = np.nextafter(1.0, 0.0)
    allocation_prob = np.clip(
        raw_prob,
        lower_probability,
        upper_probability,
    )
    allocation = allocate_rollouts_from_probabilities(
        allocation_prob,
        raw_success_prob=raw_prob,
        baseline_n=baseline_n,
        n_min=n_min,
        n_max=n_max,
        allocation_mode="dynamic_active_equalized",
        normalize_by_rho=False,
        activation_useful_threshold=activation_useful_threshold,
    )

    rollout_counts = torch.as_tensor(
        allocation.rollouts,
        dtype=torch.long,
        device=logits.device,
    )
    max_rollouts = int(rollout_counts.max().item())
    actions = torch.zeros(
        (batch_size, max_rollouts),
        dtype=torch.long,
        device=logits.device,
    )
    for count in torch.unique(rollout_counts).tolist():
        count = int(count)
        if count == 0:
            continue
        prompt_positions = torch.nonzero(
            rollout_counts == count,
            as_tuple=False,
        ).squeeze(1)
        sampled_actions = torch.multinomial(
            oracle_sampling_probs.index_select(0, prompt_positions),
            num_samples=count,
            replacement=True,
        )
        actions[prompt_positions, :count] = sampled_actions

    rollout_mask = (
        torch.arange(max_rollouts, device=logits.device)[None, :]
        < rollout_counts[:, None]
    )
    rewards = ((actions == targets[:, None]) & rollout_mask).to(logits.dtype)
    training_rollout_mask = rollout_mask.bool()
    trainable_group_count = batch_size
    allocation_active = None
    if allocation.active_prompt_mask is not None:
        allocation_active = torch.as_tensor(
            allocation.active_prompt_mask,
            dtype=torch.bool,
            device=logits.device,
        )
        training_rollout_mask = (
            training_rollout_mask & allocation_active[:, None]
        )
        trainable_group_count = int(
            training_rollout_mask.any(dim=1).sum().item()
        )
    if backward_state is not None:
        backward_state["skip_optimizer_step"] = trainable_group_count == 0
    advantages = calculate_binary_maxrl_advantages(
        rewards,
        rollout_counts,
        rollout_mask,
        epsilon=epsilon,
    )

    prompt_weights = torch.as_tensor(
        allocation.prompt_weights,
        dtype=logits.dtype,
        device=logits.device,
    )
    chosen_log_probs = log_probs_all.gather(1, actions)
    weighted_advantages = advantages * prompt_weights[:, None]
    if trainable_group_count:
        loss = -(
            weighted_advantages.detach()
            * chosen_log_probs
            * training_rollout_mask
        ).sum()
        # N0/Nq prompt weights remove rollout-count multiplicity; dividing by
        # B*N0 averages per-image losses over the full input minibatch.
        loss = loss / float(batch_size * baseline_n)
    else:
        loss = chosen_log_probs.sum() * 0.0

    if metrics is not None:
        successes = rewards.sum(dim=1)
        sampled_groups = rollout_counts > 0
        useful = (successes > 0) & (successes < rollout_counts)
        # Group fractions are among sampled groups; count statistics include
        # all images in the minibatch, including inactive images with Nq=0.
        counts = allocation.rollouts.astype(np.float64)
        metrics.update({
            "rollout_allocation/useful_group_fraction": useful[sampled_groups].float().mean().item(),
            "rollout_allocation/all_zero_group_fraction": (successes[sampled_groups] == 0).float().mean().item(),
            "rollout_allocation/rollout_min": int(counts.min()),
            "rollout_allocation/rollout_max": int(counts.max()),
            "rollout_allocation/rollout_std": float(counts.std()),
        })
    return loss


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_validation_rollouts):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    # ks = [1, 2, 4, 8, ...]
    ks = []
    k = 1
    while k <= num_validation_rollouts:
        ks.append(k)
        k *= 2
    ks_tensor = torch.tensor(ks, device=device, dtype=torch.float32)  # (K,)

    passk_sum = torch.zeros(len(ks), device=device)  # (K,)

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)

        # -------- Accuracy --------
        _, predicted = outputs.max(1)
        correct += predicted.eq(targets).sum().item()

        # -------- Vectorized pass@k --------
        probs = F.softmax(outputs, dim=1)                      # (B, C)
        p_correct = probs.gather(1, targets[:, None]).squeeze(1)  # (B,)

        # Shape trick:
        # p_correct -> (B, 1)
        # ks_tensor -> (1, K)
        # result    -> (B, K)
        passk_batch = 1.0 - (1.0 - p_correct[:, None]) ** ks_tensor[None, :]
        passk_sum += passk_batch.sum(dim=0)

        total += targets.size(0)

    avg_loss = running_loss / total
    avg_acc = correct / total
    avg_passk = {f"pass@{k}": passk_sum[i].item() / total for i, k in enumerate(ks)}

    return avg_loss, avg_acc, avg_passk



def get_checkpoint_state(
    model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    best_acc,
):
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state": (
            model.module.state_dict() if isinstance(model, nn.DataParallel)
            else model.state_dict()
        ),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_acc": best_acc,
    }
    return state


def save_checkpoint(state, is_best, checkpoint_dir, steps):
    os.makedirs(checkpoint_dir, exist_ok=True)
    latest_path = os.path.join(checkpoint_dir, f"checkpoint_latest_{steps}.pth")
    torch.save(state, latest_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pth")
        torch.save(state, best_path)



def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/imagenet")
    parser.add_argument("--dataset-name", choices=("imagenet256",), default="imagenet256")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--model-type", choices=("resnet",), default="resnet")
    parser.add_argument("--model-depth", type=int, choices=(50,), default=50)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lr_scheduler", choices=("cosine_with_warmup",), default="cosine_with_warmup")
    parser.add_argument("--advantage-type", choices=("sera",), default="sera")
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--no-amp", action="store_true", help="Retained MaxRL flag; training uses FP32")
    parser.add_argument("--seed", type=int, default=69)
    parser.add_argument("--checkpoint-dir", default="./checkpoints/imagenet_sera")
    parser.add_argument("--save-steps", type=int, nargs="+", default=[],
                        help="Also save checkpoints at these steps; keep the full epoch/LR schedule")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sera_imagenet")
    parser.add_argument("--wandb_runname", default="sera_oracle_n128")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--num_train_rollouts_per_example", type=int, default=128)
    parser.add_argument("--num_test_rollouts_per_example", type=int, default=1024)
    parser.add_argument("--max-k", type=int, default=2048, help="Retained MaxRL configuration field; unused by SERA")
    parser.add_argument("--eval_every_k", type=int, default=1000)
    parser.add_argument("--max_grad_norm", type=float, default=1e5)
    parser.add_argument("--sera-n-min", type=int, default=2)
    parser.add_argument("--sera-n-max", type=int, default=None,
                        help="Maximum rollouts per image (default: 4 * N0)")
    parser.add_argument("--sera-activation-useful-threshold", type=float, default=0.05)
    parser.add_argument("--exact-logp-gradient-monitor-every-k", type=int, default=1000)
    args = parser.parse_args()
    if args.sera_n_max is None:
        args.sera_n_max = 4 * args.num_train_rollouts_per_example
    if not 2 <= args.sera_n_min <= args.num_train_rollouts_per_example <= args.sera_n_max:
        parser.error("Require 2 <= Nmin <= N0 <= Nmax")
    if not 0 <= args.sera_activation_useful_threshold <= 1:
        parser.error("The useful-group threshold must lie in [0, 1]")
    if args.exact_logp_gradient_monitor_every_k < 0:
        parser.error("The gradient-monitor interval must be nonnegative")
    if any(step <= 0 for step in args.save_steps):
        parser.error("Checkpoint save steps must be positive")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    if args.wandb:
        wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            name=args.wandb_runname, config=vars(args),
        )
    train_loader, test_loader = get_dataloaders(
        args.dataset_name, args.data_dir, args.batch_size, args.num_workers,
    )
    model = models.resnet50(num_classes=1000)
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9,
        weight_decay=1e-4, nesterov=False,
    )
    total_steps = args.epochs * len(train_loader)
    warmup_steps = 2 * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps + 1e-6)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    best_acc = 0.0
    global_step = 0
    val_loss, val_acc, val_passk = evaluate(
        model, test_loader, criterion, device,
        num_validation_rollouts=args.num_test_rollouts_per_example,
    )
    best_acc = max(best_acc, val_acc)
    print(f"Before training: val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
    if args.wandb:
        wandb.log({
            "val/loss": val_loss, "val/acc": val_acc,
            **{f"val/{k}": v for k, v in val_passk.items()},
            "step": global_step, "epoch": 0,
        })

    for epoch in range(1, args.epochs + 1):
        running_loss, correct, total = 0, 0, 0
        for inputs, targets, _sample_indices in train_loader:
            model.train()
            global_step += 1
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            rollout_metrics = {}
            backward_state = {}
            loss = calculate_sera_oracle_loss(
                logits, targets,
                baseline_n=args.num_train_rollouts_per_example,
                n_min=args.sera_n_min, n_max=args.sera_n_max,
                activation_useful_threshold=args.sera_activation_useful_threshold,
                metrics=rollout_metrics, backward_state=backward_state,
            )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite training loss: {rollout_metrics}")
            skip_optimizer_step = backward_state.get("skip_optimizer_step", False)
            measure_alignment = (
                args.exact_logp_gradient_monitor_every_k > 0
                and global_step % args.exact_logp_gradient_monitor_every_k == 0
            )
            if skip_optimizer_step:
                grad_norm = torch.tensor(0.0, device=logits.device)
            else:
                loss.backward(retain_graph=measure_alignment)
            if measure_alignment:
                exact_nll_loss = F.cross_entropy(logits, targets, reduction="mean")
                rollout_metrics.update(measure_exact_logp_gradient_alignment(
                    model.parameters(), exact_nll_loss,
                ))
            if not skip_optimizer_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()

            running_loss += loss.item() * targets.size(0)
            correct += logits.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
            if args.wandb:
                wandb.log({
                    "train/loss": running_loss / total, "train/acc": correct / total,
                    "lr": optimizer.param_groups[0]["lr"], "train/grad_norm": grad_norm.item(),
                    "step": global_step, "epoch": epoch, **rollout_metrics,
                })
            if global_step % args.eval_every_k == 0:
                val_loss, val_acc, val_passk = evaluate(
                    model, test_loader, criterion, device,
                    num_validation_rollouts=args.num_test_rollouts_per_example,
                )
                best_acc = max(best_acc, val_acc)
                print(f"[step {global_step}] train_loss={running_loss/total:.4f}, "
                      f"val_loss={val_loss:.4f}, val_acc={val_acc*100:.2f}%")
                if args.wandb:
                    wandb.log({
                        "val/loss": val_loss, "val/acc": val_acc,
                        **{f"val/{k}": v for k, v in val_passk.items()},
                        "step": global_step, "epoch": epoch,
                    })
            if global_step in args.save_steps:
                save_checkpoint(
                    get_checkpoint_state(model, optimizer, scheduler, epoch, global_step, best_acc),
                    is_best=False, checkpoint_dir=args.checkpoint_dir, steps=global_step,
                )
    save_checkpoint(
        get_checkpoint_state(model, optimizer, scheduler, args.epochs, global_step, best_acc),
        is_best=False, checkpoint_dir=args.checkpoint_dir, steps="final",
    )
    if args.wandb:
        wandb.summary["best_val_acc"] = best_acc
    print(f"Training complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
