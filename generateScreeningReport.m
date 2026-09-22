function generateScreeningReport(imgPath, gradcamOverlay, drGrade, confidence, lesionNotes, outputPath)
%GENERATESCREENINGREPORT Create a compact screening report as PDF or PNG.
%   GENERATESCREENINGREPORT(IMGPATH, GRADCAMOVERLAY, DRGRADE, CONFIDENCE,
%   LESIONNOTES, OUTPUTPATH) places the original fundus image and Grad-CAM
%   overlay side by side with the prediction, calibrated confidence,
%   evidence notes, and human-review status. OUTPUTPATH must end in .png or
%   .pdf.
%
%   Inputs:
%       imgPath       - Path to the original fundus image.
%       gradcamOverlay- RGB Grad-CAM overlay image.
%       drGrade       - Predicted grade (0..4 or a display label).
%       confidence   - Calibrated confidence in [0, 1].
%       lesionNotes  - Short text describing lesion evidence.
%       outputPath    - Destination .png or .pdf path.

arguments
    imgPath (1, :) char
    gradcamOverlay
    drGrade
    confidence (1, 1) double {mustBeFinite, mustBeGreaterThanOrEqual(confidence, 0), mustBeLessThanOrEqual(confidence, 1)}
    lesionNotes (1, :) char
    outputPath (1, :) char
end
if ~isfile(imgPath)
    error('generateScreeningReport:MissingImage', 'Image not found: %s', imgPath);
end
[~, ~, extension] = fileparts(outputPath);
if ~ismember(lower(extension), {'.png', '.pdf'})
    error('generateScreeningReport:InvalidOutput', 'outputPath must end in .png or .pdf.');
end
original = imread(imgPath);
if ndims(original) == 2
    original = repmat(original, 1, 1, 3);
end
if ~isequal(size(original, 1), size(gradcamOverlay, 1)) || ...
        ~isequal(size(original, 2), size(gradcamOverlay, 2))
    gradcamOverlay = imresize(gradcamOverlay, [size(original, 1), size(original, 2)]);
end
reviewRequired = flagForReview(confidence);
figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 760]);
tiledlayout(3, 2, 'TileSpacing', 'compact', 'Padding', 'compact');
nexttile; imshow(original); title('Original fundus image', 'FontSize', 14);
nexttile; imshow(gradcamOverlay); title('Grad-CAM evidence', 'FontSize', 14);
nexttile([1 2]); axis off;
text(0, 0.75, sprintf('DR grade: %s', string(drGrade)), 'FontSize', 22, 'FontWeight', 'bold');
text(0, 0.42, sprintf('Calibrated confidence: %.1f%%', 100 * confidence), 'FontSize', 20);
status = 'NO - routine workflow';
if reviewRequired
    status = 'YES - human review required';
end
text(0, 0.12, ['Review required: ' status], 'FontSize', 19, ...
    'Color', [0.75 0.1 0.05], 'FontWeight', 'bold');
nexttile([1 2]); axis off;
text(0, 0.8, 'Lesion evidence', 'FontSize', 16, 'FontWeight', 'bold');
text(0, 0.55, lesionNotes, 'FontSize', 14, 'Interpreter', 'none', 'VerticalAlignment', 'top');
exportgraphics(gcf, outputPath, 'Resolution', 150);
close(gcf);
end
