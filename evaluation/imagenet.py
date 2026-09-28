"""Evaluate a trained ResNet-50 with the original ImageNet Pass@K routine."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-k", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if min(args.max_k, args.batch_size) < 1 or args.workers < 0:
        parser.error("K and batch size must be positive; workers must be nonnegative")

    import torch
    from datasets import load_dataset
    from torch.utils.data import DataLoader
    from torchvision import models, transforms
    from verl.cifar10_experiments.sampling_based_rl_objective_experiments import HFImageNet, evaluate

    dataset = load_dataset("benjamin-paine/imagenet-1k-256x256", split="validation")
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    loader = DataLoader(HFImageNet("validation", transform, {"validation": dataset}),
                        batch_size=args.batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)
    model = models.resnet50(weights=None, num_classes=1000)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    device = torch.device("cuda")
    loss, accuracy, passk = evaluate(model.to(device), loader, torch.nn.CrossEntropyLoss(), device, args.max_k)
    result = dict(checkpoint=args.checkpoint, images=len(dataset), cross_entropy=loss,
                  top1_accuracy=accuracy, **passk)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
