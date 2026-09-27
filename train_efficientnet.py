import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch import nn, optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


CLASS_NAMES = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]


class FundusDataset(Dataset):
    def __init__(self, records, image_dir, transform):
        self.records = records
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, label = self.records[index]
        image = Image.open(self.image_dir / f"{image_id}.png").convert("RGB")
        image = ImageOps.fit(image, (256, 256), method=Image.Resampling.BILINEAR)
        return self.transform(image), label


def load_records(label_file, image_dir):
    available = {path.stem for path in Path(image_dir).glob("*.png")}
    with open(label_file, newline="", encoding="utf-8-sig") as stream:
        records = [(row["id_code"], int(row["diagnosis"])) for row in csv.DictReader(stream) if row["id_code"] in available]
    if {label for _, label in records} != set(range(5)):
        raise ValueError("The dataset must contain all five classes")
    return records


def split_records(records, seed=42):
    rng = random.Random(seed)
    groups = {label: [] for label in range(5)}
    for record in records:
        groups[record[1]].append(record)
    train, validation = [], []
    for group in groups.values():
        rng.shuffle(group)
        split = max(1, round(len(group) * 0.2))
        validation.extend(group[:split])
        train.extend(group[split:])
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


def qwk(labels, predictions):
    matrix = torch.zeros((5, 5), dtype=torch.float64)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    expected = torch.outer(matrix.sum(1), matrix.sum(0)) / matrix.sum().clamp_min(1)
    weights = torch.tensor([[(actual - predicted) ** 2 / 16 for predicted in range(5)] for actual in range(5)])
    denominator = (weights * expected).sum()
    return float(1 - (weights * matrix).sum() / denominator) if denominator else 0.0


def calculate_metrics(labels, predictions):
    recalls = []
    for label in range(5):
        total = sum(actual == label for actual in labels)
        recalls.append(sum(actual == predicted == label for actual, predicted in zip(labels, predictions)) / total if total else 0)
    return {"accuracy": sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels), "balanced_accuracy": sum(recalls) / 5, "qwk": qwk(labels, predictions), "per_class_recall": recalls}


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad(), autocast("cuda"):
        for images, batch_labels in loader:
            probabilities.append(torch.softmax(model(images.to(device, non_blocking=True)), dim=1).float().cpu())
            labels.extend(batch_labels.tolist())
    return torch.cat(probabilities), labels


def main():
    parser = argparse.ArgumentParser(description="Fine-tune EfficientNet-B3 for diabetic retinopathy")
    parser.add_argument("--image-dir", default="train_images/train_images")
    parser.add_argument("--label-file", default="train_1.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output", default="efficientnet_b3_finetuned.pt")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; GPU training was not enabled")
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda")
    records = load_records(args.label_file, args.image_dir)
    train_records, validation_records = split_records(records)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Full dataset: {len(records)} | Training: {len(train_records)} | Validation: {len(validation_records)}", flush=True)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), transforms.RandomHorizontalFlip(), transforms.RandomRotation(10), transforms.ToTensor(), normalize])
    validation_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    train_dataset = FundusDataset(train_records, args.image_dir, train_transform)
    validation_dataset = FundusDataset(validation_records, args.image_dir, validation_transform)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_records]), minlength=5).float()
    sample_weights = torch.tensor([(class_counts.sum() / class_counts.clamp_min(1))[label] for _, label in train_records], dtype=torch.double)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True), num_workers=0, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler("cuda")
    best_accuracy = -1
    best_metrics = None
    for epoch in range(args.epochs):
        if epoch == 2:
            for block in list(model.features.children())[-3:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            optimizer = optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=2e-5, weight_decay=1e-4)
        elif epoch == 0:
            optimizer = optim.AdamW(model.classifier.parameters(), lr=5e-4, weight_decay=1e-4)
        model.train()
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_correct += (outputs.argmax(1) == labels).sum().item()
            train_total += labels.size(0)
        probabilities, labels = evaluate(model, validation_loader, device)
        current = calculate_metrics(labels, probabilities.argmax(1).tolist())
        current["train_accuracy"] = train_correct / train_total
        print(f"Epoch {epoch + 1}/{args.epochs} - train acc {current['train_accuracy']:.3f} - val acc {current['accuracy']:.3f} - qwk {current['qwk']:.3f}", flush=True)
        if current["accuracy"] > best_accuracy:
            best_accuracy = current["accuracy"]
            best_metrics = current
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "device": str(device), "metrics": current}, args.output)
    result = {"device": str(device), "gpu": torch.cuda.get_device_name(0), "records": len(records), "training_records": len(train_records), "validation_records": len(validation_records), "model": "EfficientNet-B3 fine-tuned", "metrics": best_metrics, "baseline_accuracy": 0.7768313458262351}
    Path("efficientnet_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved {args.output} and efficientnet_metrics.json", flush=True)


if __name__ == "__main__":
    main()