%DEMO_EXPLAINABILITY Run the DR explainability slice on one APTOS image.
% Configure APTOS_ROOT and provide a trained network in drNet.mat. The
% MAT-file must contain variable net; architecture and feature layers are
% intentionally left to the project owner.

clear; clc;
APTOS_ROOT = fullfile(pwd, 'data', 'aptos');
imageDir = fullfile(APTOS_ROOT, 'train_images');
labelFile = fullfile(APTOS_ROOT, 'train.csv');
networkFile = fullfile(pwd, 'drNet.mat');
if ~isfile(labelFile) || ~isfolder(imageDir)
    error('demo_explainability:MissingDataset', ...
        'Place Kaggle APTOS files under %s first.', APTOS_ROOT);
end
if ~isfile(networkFile)
    error('demo_explainability:MissingNetwork', ...
        'Provide a trained network in drNet.mat after selecting its architecture.');
end
loaded = load(networkFile, 'net');
net = loaded.net;
labels = readtable(labelFile, 'TextType', 'string');
imagePath = fullfile(imageDir, labels.id_code(1) + ".png");
img = imread(imagePath);
inputSize = net.Layers(1).InputSize;
modelImage = imresize(img, inputSize(1:2));
rawProbs = predict(net, single(modelImage));
rawProbs = toNumeric(rawProbs);
rawProbs = double(rawProbs(:).');
validationCount = min(height(labels), 32);
validationProbs = zeros(validationCount, 5);
for index = 1:validationCount
    validationImage = imread(fullfile(imageDir, labels.id_code(index) + ".png"));
    validationImage = imresize(validationImage, inputSize(1:2));
    validationProbs(index, :) = toNumeric(predict(net, single(validationImage)));
end
valLabels = categorical(labels.diagnosis(1:validationCount), 0:4, ...
    {'Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4'});
confidenceCalibrator = calibrateConfidence(rawProbs, valLabels, validationProbs);
predictedClass = find(rawProbs == max(rawProbs), 1);
[heatmap, overlay] = generateGradCAM(net, modelImage, predictedClass);
confidence = confidenceCalibrator(rawProbs);
generateScreeningReport(imagePath, overlay, predictedClass - 1, confidence, ...
    'Review highlighted regions alongside clinical findings.', fullfile(pwd, 'screening_report.png'));

fprintf('Grade: %d | Confidence: %.1f%% | Review: %s\n', predictedClass - 1, ...
    100 * confidence, string(flagForReview(confidence)));

function values = toNumeric(values)
if isa(values, 'dlarray')
    values = extractdata(values);
end
values = double(gather(values));
values = values(:).';
end
