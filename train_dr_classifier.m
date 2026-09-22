function net = train_dr_classifier(imageDir, labelFile, architecture)
%TRAIN_DR_CLASSIFIER Train a five-class DR classifier with transfer learning.
%   NET = TRAIN_DR_CLASSIFIER(IMAGEDIR, LABELFILE, ARCHITECTURE) trains a
%   ResNet-18 transfer-learning classifier on APTOS 2019. Images are
%   augmented and resized to the selected network's input size, and the
%   training split is stratified by diagnosis.
%
%   Inputs:
%       imageDir    - Folder containing APTOS training images.
%       labelFile   - APTOS train.csv path.
%       architecture- Currently supported architecture: 'resnet18'.
%
%   Outputs:
%       net - Trained classifier network.

arguments
    imageDir (1, :) char
    labelFile (1, :) char
    architecture (1, :) char
end
if ~isfolder(imageDir) || ~isfile(labelFile)
    error('train_dr_classifier:MissingData', 'Check imageDir and labelFile.');
end
if ~strcmpi(architecture, 'resnet18')
    error('train_dr_classifier:UnsupportedArchitecture', ...
        'This entry point currently supports only ResNet-18.');
end

labelsTable = readtable(labelFile, 'TextType', 'string');
imageFiles = fullfile(imageDir, labelsTable.id_code + ".png");
exists = isfile(imageFiles);
if ~all(exists)
    imageFiles = imageFiles(exists);
    labelsTable = labelsTable(exists, :);
end
imageLabels = categorical(labelsTable.diagnosis, 0:4, {'Level 0', 'Level 1', ...
    'Level 2', 'Level 3', 'Level 4'});
if numel(categories(imageLabels)) ~= 5
    error('train_dr_classifier:InvalidLabels', 'APTOS labels must contain classes 0 through 4.');
end

imds = imageDatastore(imageFiles, 'Labels', imageLabels, ...
    'ReadFcn', @readFundusImage);
[trainingImds, validationImds] = splitEachLabel(imds, 0.8, 'randomized');
baseNet = resnet18;
inputSize = baseNet.Layers(1).InputSize;
layers = layerGraph(baseNet);
layers = replaceLayer(layers, 'fc1000', fullyConnectedLayer(5, ...
    'Name', 'dr_fc', 'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10));
classNames = categories(trainingImds.Labels);
classCounts = countcats(trainingImds.Labels);
classWeights = sum(classCounts) ./ max(classCounts, 1);
layers = replaceLayer(layers, 'ClassificationLayer_predictions', ...
    classificationLayer('Name', 'dr_output', 'Classes', classNames, ...
    'ClassWeights', classWeights));

augmenter = imageDataAugmenter('RandXReflection', true, ...
    'RandRotation', [-10 10], 'RandScale', [0.9 1.1]);
augmentedTraining = augmentedImageDatastore(inputSize(1:2), trainingImds, ...
    'DataAugmentation', augmenter, 'ColorPreprocessing', 'gray2rgb');
augmentedValidation = augmentedImageDatastore(inputSize(1:2), validationImds, ...
    'ColorPreprocessing', 'gray2rgb');
options = trainingOptions('adam', 'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 8, 'MiniBatchSize', 32, 'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValidation, 'ValidationFrequency', 50, ...
    'Verbose', true, 'Plots', 'training-progress');
net = trainNetwork(augmentedTraining, layers, options);
end

function image = readFundusImage(filename)
image = imread(filename);
if ndims(image) == 2
    image = repmat(image, 1, 1, 3);
end
if size(image, 3) > 3
    image = image(:, :, 1:3);
end
end
