import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms


CLASS_NAMES = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]
MODEL_NAMES = ["efficientnet_b3", "densenet121", "convnext_tiny"]
MODEL_WEIGHTS = [0.45, 0.30, 0.25]


class FundusDataset(Dataset):
    def __init__(self, records, image_dir, transform):
        self.records = records
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.images = [self.transform(Image.open(self.image_dir / f"{image_id}.png").convert("RGB")) for image_id, _ in records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.images[index], self.records[index][1]


def load_records(label_file, image_dir):
    available = {path.stem for path in Path(image_dir).glob("*.png")}
    with open(label_file, newline="", encoding="utf-8-sig") as stream:
        records = [(row["id_code"], int(row["diagnosis"])) for row in csv.DictReader(stream) if row["id_code"] in available]
    if not records or {label for _, label in records} != set(range(5)):
        raise ValueError("The label file and image directory must contain all five classes")
    return records


def stratified_split(records, fraction=0.2, seed=42):
    rng = random.Random(seed)
    groups = {label: [] for label in range(5)}
    for record in records:
        groups[record[1]].append(record)
    training, validation = [], []
    for group in groups.values():
        rng.shuffle(group)
        split = max(1, round(len(group) * fraction))
        validation.extend(group[:split])
        training.extend(group[split:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def build_model(name):
    if name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    elif name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, 5)
    else:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 5)
    return model


def quadratic_weighted_kappa(labels, predictions):
    matrix = torch.zeros((5, 5), dtype=torch.float64)
    for actual, predicted in zip(labels, predictions):
        matrix[actual, predicted] += 1
    actual_marginals = matrix.sum(1)
    predicted_marginals = matrix.sum(0)
    expected = torch.outer(actual_marginals, predicted_marginals) / max(matrix.sum().item(), 1)
    weights = torch.zeros((5, 5), dtype=torch.float64)
    for actual in range(5):
        for predicted in range(5):
            weights[actual, predicted] = ((actual - predicted) ** 2) / 16
    observed = (weights * matrix).sum()
    expected_score = (weights * expected).sum()
    return float(1 - observed / expected_score) if expected_score else 0.0


def metrics(labels, predictions):
    labels = list(labels)
    predictions = list(predictions)
    accuracy = sum(actual == predicted for actual, predicted in zip(labels, predictions)) / len(labels)
    per_class = []
    for label in range(5):
        positives = sum(actual == label for actual in labels)
        per_class.append(sum(actual == predicted == label for actual, predicted in zip(labels, predictions)) / positives if positives else 0)
    return {"accuracy": accuracy, "balanced_accuracy": sum(per_class) / 5, "qwk": quadratic_weighted_kappa(labels, predictions), "per_class_recall": per_class}


def evaluate(model, loader, device):
    model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for images, batch_labels in loader:
            probabilities.append(torch.softmax(model(images.to(device)), dim=1).cpu())
            labels.extend(batch_labels.tolist())
    return torch.cat(probabilities), labels


def write_dashboard(metrics_data, output):
    baseline = metrics_data["baseline"]
    ensemble = metrics_data["ensemble"]
    rows = "".join(f"<tr><td>{name}</td><td>{value.get('train_accuracy', 0):.2%}</td><td>{value['accuracy']:.2%}</td><td>{value['balanced_accuracy']:.2%}</td><td>{value['qwk']:.3f}</td></tr>" for name, value in metrics_data["models"].items())
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>DR model comparison</title><style>body{{margin:0;background:#f4f1ea;color:#20241f;font:16px Georgia,serif}}main{{max-width:1050px;margin:0 auto;padding:48px 24px}}h1{{font-size:44px;font-weight:400;margin:0 0 8px}}.sub{{color:#667064;margin-bottom:32px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.card{{background:#fffdf8;border:1px solid #d8d3c8;padding:22px}}.value{{font-size:34px;margin-top:10px}}table{{width:100%;border-collapse:collapse;background:#fffdf8;margin-top:16px}}th,td{{padding:14px;text-align:left;border-bottom:1px solid #e4dfd5}}th{{color:#667064;font-weight:400}}.accent{{color:#a34b2d}}@media(max-width:700px){{h1{{font-size:34px}}.grid{{grid-template-columns:1fr}}main{{padding:28px 16px}}}}</style></head><body><main><h1>Retinal classifier review</h1><div class="sub">Full-dataset CUDA training with stratified holdout validation</div><section class="grid"><div class="card"><div>Previous ResNet-18</div><div class="value">{baseline['accuracy']:.2%}</div><div>validation accuracy</div></div><div class="card"><div>Hybrid ensemble</div><div class="value accent">{ensemble['accuracy']:.2%}</div><div>validation accuracy</div></div><div class="card"><div>Change</div><div class="value">{ensemble['accuracy'] - baseline['accuracy']:+.2%}</div><div>accuracy delta</div></div></section><h2>Metrics</h2><table><thead><tr><th>Model</th><th>Train accuracy</th><th>Validation accuracy</th><th>Balanced accuracy</th><th>QWK</th></tr></thead><tbody><tr><td>Previous ResNet-18</td><td>n/a</td><td>{baseline['accuracy']:.2%}</td><td>n/a</td><td>n/a</td></tr>{rows}<tr><td><strong>Weighted ensemble</strong></td><td>n/a</td><td><strong>{ensemble['accuracy']:.2%}</strong></td><td><strong>{ensemble['balanced_accuracy']:.2%}</strong></td><td><strong>{ensemble['qwk']:.3f}</strong></td></tr></tbody></table><p>Records: {metrics_data['records']} total, stratified 80/20 split. Ensemble weights: EfficientNet-B3 45%, DenseNet-121 30%, ConvNeXt-Tiny 25%.</p></main></body></html>'''
    Path(output).write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="train_images/train_images")
    parser.add_argument("--label-file", default="train_1.csv")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-dir", default="ensemble_results")
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this run; torch.cuda.is_available() is false")
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    records = load_records(args.label_file, args.image_dir)
    if args.max_records:
        per_class = max(1, args.max_records // 5)
        selected_records = []
        for label in range(5):
            selected_records.extend([record for record in records if record[1] == label][:per_class])
        records = selected_records[:args.max_records]
    print(f"Preparing {len(records)} labeled images from {args.image_dir}", flush=True)
    training_records, validation_records = stratified_split(records)
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), normalize])
    validation_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), normalize])
    train_dataset = FundusDataset(training_records, args.image_dir, train_transform)
    validation_dataset = FundusDataset(validation_records, args.image_dir, validation_transform)
    class_counts = torch.bincount(torch.tensor([label for _, label in training_records]), minlength=5).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1)
    sample_weights = torch.tensor([class_weights[label] for _, label in training_records], dtype=torch.double)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True), pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    print(f"Training images: {len(training_records)} | Validation images: {len(validation_records)}", flush=True)
    model_metrics, validation_probabilities = {}, []
    for name in MODEL_NAMES:
        model = build_model(name)
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        model = model.to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        best_accuracy = -1
        for epoch in range(args.epochs):
            model.train()
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
            probabilities, labels = evaluate(model, validation_loader, device)
            current_metrics = metrics(labels, probabilities.argmax(1).tolist())
            current_metrics["train_accuracy"] = train_correct / train_total
            print(f"{name} epoch {epoch + 1}/{args.epochs} - val acc {current_metrics['accuracy']:.3f} - qwk {current_metrics['qwk']:.3f}", flush=True)
            if current_metrics["accuracy"] > best_accuracy:
                best_accuracy = current_metrics["accuracy"]
                torch.save({"model_state_dict": model.state_dict(), "class_names": CLASS_NAMES, "model": name, "device": str(device)}, output_dir / f"{name}.pt")
                model_metrics[name] = current_metrics
        validation_probabilities.append(probabilities)
        del model
        torch.cuda.empty_cache()
    ensemble_probabilities = sum(weight * probability for weight, probability in zip(MODEL_WEIGHTS, validation_probabilities))
    ensemble_metrics = metrics(labels, ensemble_probabilities.argmax(1).tolist())
    result = {"device": str(device), "gpu": torch.cuda.get_device_name(0), "records": len(records), "models": model_metrics, "ensemble": ensemble_metrics, "ensemble_weights": dict(zip(MODEL_NAMES, MODEL_WEIGHTS)), "baseline": {"accuracy": 0.7768313458262351}}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_dashboard(result, "metrics_dashboard.html")
    print(f"Ensemble accuracy: {ensemble_metrics['accuracy']:.3f}", flush=True)
    print("Wrote ensemble_results/metrics.json and metrics_dashboard.html", flush=True)


if __name__ == "__main__":
    main()