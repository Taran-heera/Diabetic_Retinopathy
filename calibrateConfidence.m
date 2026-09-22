function calibratedConfidence = calibrateConfidence(rawProbs, valLabels, valProbs)
%CALIBRATECONFIDENCE Fit temperature scaling on validation probabilities.
%   CALIBRATEDCONFIDENCE = CALIBRATECONFIDENCE(RAWPROBS, VALLABELS,
%   VALPROBS) returns a function handle. Calling it with an N-by-K matrix
%   of softmax probabilities returns the calibrated maximum class
%   confidence for each row. Temperature is fitted by minimizing
%   validation negative log-likelihood.
%
%   Inputs:
%       rawProbs  - N-by-K raw softmax probabilities, used as a fallback
%                   when valProbs is empty and checked for shape otherwise.
%       valLabels - N ground-truth labels, numeric 1..K, categorical, or
%                   string/cellstr labels matching the probability columns.
%       valProbs  - N-by-K validation softmax probabilities used for fitting.
%
%   Outputs:
%       calibratedConfidence - Function handle: probs -> N-by-1 confidence.

arguments
    rawProbs double {mustBeNonempty, mustBeFinite}
    valLabels
    valProbs double = []
end

if isempty(valProbs)
    valProbs = rawProbs;
end
if ~ismatrix(valProbs) || size(valProbs, 2) < 2 || any(valProbs(:) < 0)
    error('calibrateConfidence:InvalidProbabilities', ...
        'Probability inputs must be nonnegative N-by-K matrices with K >= 2.');
end
if size(rawProbs, 2) ~= size(valProbs, 2) || size(rawProbs, 1) ~= size(valProbs, 1)
    error('calibrateConfidence:SizeMismatch', ...
        'rawProbs and valProbs must have the same size.');
end
if any(abs(sum(valProbs, 2) - 1) > 1e-4)
    error('calibrateConfidence:NotSoftmax', 'valProbs rows must sum to one.');
end
classIndices = labelsToIndices(valLabels, size(valProbs, 2));
if numel(classIndices) ~= size(valProbs, 1)
    error('calibrateConfidence:LabelCount', 'valLabels must contain one label per row.');
end

probabilities = max(valProbs, eps);
logProbabilities = log(probabilities);
objective = @(logTemperature) negativeLogLikelihood(exp(logTemperature), ...
    logProbabilities, classIndices);
logTemperature = fminsearch(objective, 0, optimset('Display', 'off'));
temperature = max(exp(logTemperature), eps);
calibratedConfidence = @(probs) applyTemperature(probs, temperature);
end

function loss = negativeLogLikelihood(temperature, logProbabilities, labels)
logits = logProbabilities ./ temperature;
logNormalizer = logsumexp(logits, 2);
loss = -mean(logits(sub2ind(size(logits), (1:size(logits, 1))', labels)) - logNormalizer);
end

function confidence = applyTemperature(probs, temperature)
validateattributes(probs, {'double', 'single'}, {'2d', 'nonnegative', 'finite'});
if any(abs(sum(probs, 2) - 1) > 1e-3)
    error('calibrateConfidence:NotSoftmax', 'Input probability rows must sum to one.');
end
logits = log(max(double(probs), eps)) ./ temperature;
logits = logits - max(logits, [], 2);
scaled = exp(logits);
scaled = scaled ./ sum(scaled, 2);
confidence = max(scaled, [], 2);
end

function indices = labelsToIndices(labels, classCount)
if isnumeric(labels)
    indices = double(labels(:));
elseif iscategorical(labels)
    [~, ~, indices] = unique(labels(:));
elseif isstring(labels) || iscellstr(labels)
    [~, ~, indices] = unique(string(labels(:)));
else
    error('calibrateConfidence:InvalidLabels', 'Unsupported valLabels type.');
end
if any(indices < 1 | indices > classCount | indices ~= floor(indices))
    error('calibrateConfidence:InvalidLabels', 'Labels must map to class indices 1..K.');
end
indices = indices(:);
end

function value = logsumexp(values, dimension)
maximum = max(values, [], dimension);
value = maximum + log(sum(exp(values - maximum), dimension));
end
