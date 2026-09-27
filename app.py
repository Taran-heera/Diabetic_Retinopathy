import base64
import io
import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "drNet_resnet18.pt"
CLASS_NAMES = ["Level 0", "Level 1", "Level 2", "Level 3", "Level 4"]
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Missing model checkpoint: {CHECKPOINT_PATH.name}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


MODEL = load_model()
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def _safe_image(image_bytes):
    try:
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("The uploaded file is not a readable image.") from error


def compute_quality_score(image):
    numpy_image = np.asarray(image.convert("L"), dtype=np.float32)
    if numpy_image.size == 0:
        return 0.0

    contrast = numpy_image.std() / 255.0
    brightness = 1.0 - abs(numpy_image.mean() - 128.0) / 128.0
    gradient_x = np.zeros_like(numpy_image)
    gradient_y = np.zeros_like(numpy_image)
    gradient_x[:, 1:] = np.abs(np.diff(numpy_image, axis=1))
    gradient_y[1:, :] = np.abs(np.diff(numpy_image, axis=0))
    gradient = gradient_x + gradient_y
    sharpness = np.clip(gradient.mean() / 30.0, 0.0, 1.0)
    quality_score = float(np.clip(0.45 * contrast + 0.30 * sharpness + 0.25 * brightness, 0.0, 1.0))
    return quality_score


def get_quality_label(score):
    if score >= 0.75:
        return "High quality"
    if score >= 0.55:
        return "Fair quality"
    if score >= 0.35:
        return "Low quality"
    return "Poor quality"


def generate_overlay_data(image, confidence):
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.float32)

    gradient_x = np.zeros_like(gray)
    gradient_y = np.zeros_like(gray)
    gradient_x[:, 1:] = np.abs(np.diff(gray, axis=1))
    gradient_y[1:, :] = np.abs(np.diff(gray, axis=0))

    map_data = np.clip((gradient_x + gradient_y) / 80.0, 0.0, 1.0)
    map_data = map_data + np.clip((gray - 90.0) / 140.0, 0.0, 1.0)
    map_data = np.clip(map_data * (0.6 + confidence), 0.0, 1.0)

    color_mask = np.zeros_like(rgb)
    color_mask[..., 0] = 255.0
    color_mask[..., 1] = 110.0 + 120.0 * map_data
    color_mask[..., 2] = 25.0

    alpha = 0.4 + 0.35 * map_data
    combined = rgb * (1.0 - alpha[..., None]) + color_mask * alpha[..., None]
    combined = np.clip(combined, 0, 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(combined).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def build_explanation(grade_index, confidence, quality_score, capture_mode):
    evidence = []
    if capture_mode == "selfie":
        evidence.append("Smartphone eye selfie mode: low-cost triage only; retinal fundus imaging remains the preferred diagnostic gold standard.")
    else:
        evidence.append("Retinal image acquisition mode: direct screening workflow for diabetic retinopathy assessment.")

    if quality_score < 0.35:
        evidence.append("Image quality is weak; the system recommends recapture before a final referral decision.")
    elif quality_score < 0.55:
        evidence.append("Image quality is moderate. Confidence is reduced and human review is recommended when severity is borderline.")
    else:
        evidence.append("Image quality is acceptable for screening and the prediction is more reliable.")

    if grade_index == 0:
        evidence.append("No major retinal lesion pattern is visible in this image. The screening result is consistent with a non-DR state.")
    elif grade_index == 1:
        evidence.append("Mild retinal abnormalities are suggested. The system recommends routine monitoring and repeat imaging if needed.")
    elif grade_index == 2:
        evidence.append("Moderate retinal abnormalities are present. Clinical review is recommended to confirm severity and determine next steps.")
    elif grade_index == 3:
        evidence.append("Severe retinal changes are likely. Ophthalmology review or urgent referral should be prioritized.")
    else:
        evidence.append("Very severe retinal involvement is strongly suggested. Immediate clinical review is advised.")

    if confidence < 0.7:
        evidence.append("The prediction confidence is below the review threshold. This case is flagged for expert validation.")

    return evidence


def classify_image(image_bytes, capture_mode="fundus"):
    image = _safe_image(image_bytes)
    quality_score = compute_quality_score(image)
    quality_label = get_quality_label(quality_score)

    tensor = PREPROCESS(image).unsqueeze(0)
    with torch.inference_mode():
        probabilities = torch.softmax(MODEL(tensor), dim=1)[0]

    predicted_index = int(probabilities.argmax().item())
    confidence = float(probabilities[predicted_index].item())
    predicted_grade = CLASS_NAMES[predicted_index]
    not_dr = predicted_index == 0

    if capture_mode == "selfie":
        review_flag = confidence < 0.80 or quality_score < 0.45
    else:
        review_flag = confidence < 0.70 or quality_score < 0.45

    if predicted_index >= 3:
        action = "Urgent ophthalmology review"
    elif predicted_index >= 2:
        action = "Clinical review recommended"
    elif quality_score < 0.45:
        action = "Retake image and rescan"
    elif predicted_index == 1:
        action = "Monitor and repeat scan in follow-up"
    else:
        action = "Routine screening follow-up"

    explanation = build_explanation(predicted_index, confidence, quality_score, capture_mode)
    heatmap = generate_overlay_data(image, confidence)

    return {
        "capture_mode": capture_mode,
        "category": "No DR likely" if not_dr else "DR likely",
        "grade": predicted_grade,
        "confidence": confidence,
        "quality": {
            "score": round(quality_score, 3),
            "label": quality_label,
            "needs_recapture": quality_score < 0.45,
        },
        "review_required": review_flag,
        "action": action,
        "explanation": explanation,
        "heatmap": heatmap,
        "probabilities": [
            {"label": label, "value": float(probability)}
            for label, probability in zip(CLASS_NAMES, probabilities.tolist())
        ],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok", "model": "resnet18", "classes": len(CLASS_NAMES)})


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    capture_mode = request.form.get("capture_mode", "fundus")
    if upload is None or not upload.filename:
        return jsonify({"error": "Choose an image or take a close-eye selfie first."}), 400
    try:
        result = classify_image(upload.read(), capture_mode=capture_mode)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    return jsonify(result)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Image is too large. Use a file smaller than 12 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5001")), debug=False)