# DR Explainability Module

This folder contains architecture-independent MATLAB explainability utilities for the APTOS 2019 diabetic retinopathy task.

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
