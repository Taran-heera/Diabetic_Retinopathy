function [heatmap, overlay] = generateGradCAM(net, img, targetClass)
%GENERATEGRADCAM Generate and display a Grad-CAM explanation.
%   [HEATMAP, OVERLAY] = GENERATEGRADCAM(NET, IMG, TARGETCLASS) uses the
%   built-in GRADCAM function to create a class-activation heatmap for
%   TARGETCLASS, resizes it to IMG, and blends it with the RGB image.
%
%   Inputs:
%       net         - Trained DAGNetwork, SeriesNetwork, or dlnetwork.
%       img         - H-by-W-by-3 RGB image accepted by NET.
%       targetClass - Class name or one-based class index to explain.
%
%   Outputs:
%       heatmap - Single-channel heatmap normalized to [0, 1].
%       overlay - H-by-W-by-3 uint8 RGB visualization of the explanation.

arguments
    net
    img
    targetClass
end

validateattributes(img, {'uint8', 'uint16', 'single', 'double'}, ...
    {'nonempty', 'nonsparse'}, mfilename, 'img');
if ndims(img) ~= 3 || size(img, 3) ~= 3
    error('generateGradCAM:InvalidImage', 'img must be an H-by-W-by-3 RGB image.');
end
if ~(ischar(targetClass) || isstring(targetClass) || ...
        (isnumeric(targetClass) && isscalar(targetClass) && ...
        isfinite(targetClass) && targetClass >= 1 && targetClass == floor(targetClass)))
    error('generateGradCAM:InvalidClass', ...
        'targetClass must be a class name or positive integer class index.');
end

% MATLAB returns a score map whose spatial size depends on the selected feature layer.
rawMap = gradCAM(net, img, targetClass);
if ndims(rawMap) == 3
    rawMap = rawMap(:, :, 1);
end
if isa(rawMap, 'dlarray')
    rawMap = extractdata(rawMap);
end
rawMap = imresize(gather(rawMap), [size(img, 1), size(img, 2)]);
rawMap = double(rawMap);
rawMap = rawMap - min(rawMap(:));
maximum = max(rawMap(:));
if maximum > 0
    heatmap = rawMap ./ maximum;
else
    heatmap = zeros(size(rawMap));
end

baseImage = im2uint8(img);
if ~isa(img, 'uint8')
    baseImage = im2uint8(im2double(img));
end
colorMap = uint8(255 * ind2rgb(max(1, round(heatmap * 255) + 1), jet(256)));
alpha = 0.45 * heatmap;
overlay = uint8((1 - alpha) .* double(baseImage) + alpha .* double(colorMap));

figure('Name', 'Grad-CAM', 'Color', 'w');
tiledlayout(1, 2, 'Padding', 'compact', 'TileSpacing', 'compact');
nexttile; imshow(baseImage); title('Original');
nexttile; imshow(overlay); title('Grad-CAM');
colormap(jet(256)); colorbar;
end
