# DR Explainability Module

This folder contains architecture-independent MATLAB explainability utilities for the APTOS 2019 diabetic retinopathy task.

## Local screening application

The repository includes the trained `drNet_resnet18.pt` checkpoint, Flask backend, and browser interface. After cloning:

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5001` in a browser. The default port is `5001`; set the `PORT` environment variable to use another port. The application supports fundus image upload, close-eye camera capture, quality scoring, screening output, probability distribution, and explainability review panels.

The application is a research prototype and decision-support tool, not a clinical diagnosis device. The included ResNet checkpoint is required for inference and is versioned with the repository.

## Dataset

Download APTOS 2019 manually from Kaggle and place the files in this layout:

```text
data/aptos/train.csv
data/aptos/train_images/*.png
```

## MATLAB demo

1. Run `train_dr_classifier(fullfile(pwd, "data", "aptos", "train_images"), fullfile(pwd, "data", "aptos", "train.csv"), "resnet18")`.
2. Save the result as `drNet.mat` with variable name `net`.
3. Run `demo_explainability`.

`generateGradCAM` currently uses MATLAB's documented `gradCAM(net, img, targetClass)` API. The selected architecture may require the network's expected preprocessing or a feature-layer argument; that choice belongs in the confirmed training wrapper.
