import argparse
import csv
import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


CLASS_NAMES = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]


class FundusDataset(Dataset):
    def __init__(self, records, image_dir, transform=None):
        self.records = records
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image_id, label = self.records[index]
        image = Image.open(self.image_dir / f"{image_id}.png").convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def stratified_split(records, validation_fraction=0.2, seed=42):
    rng = random.Random(seed)
    by_class = {label: [] for label in range(len(CLASS_NAMES))}
    for record in records:
        by_class[record[1]].append(record)
    training, validation = [], []
    for class_records in by_class.values():
        rng.shuffle(class_records)
        split = max(1, round(len(class_records) * validation_fraction))
        validation.extend(class_records[:split])
        training.extend(class_records[split:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def load_records(label_file):
    with open(label_file, newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        records = [(row["id_code"], int(row["diagnosis"])) for row in rows]
    if not records or {label for _, label in records} != set(range(5)):
        raise ValueError("CSV must contain non-empty labels for classes 0 through 4")
    return records


def main():
    parser = argparse.ArgumentParser(description="Train a ResNet-18 DR classifier")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-file", required=True)
    parser.add_argument("--output", default="drNet_resnet18.pt")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    records = load_records(args.label_file)
    training_records, validation_records = stratified_split(records)
    print(f"Device: {device}", flush=True)
    print(f"Training images: {len(training_records)} | Validation images: {len(validation_records)}", flush=True)

    normalization = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    training_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, scale=(0.9, 1.1)),
        transforms.ToTensor(),
        normalization,
    ])
    validation_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalization,
    ])
    training_dataset = FundusDataset(training_records, args.image_dir, training_transform)
    validation_dataset = FundusDataset(validation_records, args.image_dir, validation_transform)
    class_counts = torch.bincount(torch.tensor([label for _, label in training_records]), minlength=5).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1)
    sample_weights = torch.tensor([class_weights[label] for _, label in training_records], dtype=torch.double)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    best_accuracy = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in training_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        model.eval()
        validation_correct = 0
        validation_total = 0
        validation_loss = 0.0
        with torch.no_grad():
            for images, labels in validation_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                validation_loss += criterion(outputs, labels).item() * labels.size(0)
                validation_correct += (outputs.argmax(1) == labels).sum().item()
                validation_total += labels.size(0)
        train_loss = running_loss / total
        train_accuracy = correct / total
        val_loss = validation_loss / validation_total
        val_accuracy = validation_correct / validation_total
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_accuracy, "validation_loss": val_loss, "validation_accuracy": val_accuracy})
        print(f"Epoch {epoch + 1}/{args.epochs} - train loss {train_loss:.4f}, train acc {train_accuracy:.3f}, val loss {val_loss:.4f}, val acc {val_accuracy:.3f}", flush=True)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "history": history}, args.output)
            print(f"Saved best checkpoint: {args.output}", flush=True)

    Path("training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Training complete. Best validation accuracy: {best_accuracy:.3f}", flush=True)


if __name__ == "__main__":
    main()
