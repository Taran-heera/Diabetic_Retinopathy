function reviewRequired = flagForReview(confidence, threshold)
%FLAGFORREVIEW Flag predictions below the human-review confidence threshold.
%   REVIEWREQUIRED = FLAGFORREVIEW(CONFIDENCE, THRESHOLD) returns a logical
%   array that is true where confidence is below THRESHOLD. THRESHOLD
%   defaults to 0.7.
%
%   Inputs:
%       confidence - Scalar or array of calibrated confidence values [0, 1].
%       threshold  - Optional scalar review threshold in [0, 1].
%
%   Outputs:
%       reviewRequired - Logical array with the same size as confidence.

if nargin < 2 || isempty(threshold)
    threshold = 0.7;
end
validateattributes(confidence, {'numeric'}, {'real', 'finite', 'nonnegative', '<=', 1});
validateattributes(threshold, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<=', 1});
reviewRequired = confidence < threshold;
end
