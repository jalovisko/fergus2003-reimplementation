function features = kadir_brady(img, N_features, s_min, s_max)
% KADIR_BRADY  Entropy-based salient region detector (Kadir & Brady, 2001).
%
%   features = kadir_brady(img, N_features, s_min, s_max)
%
%   Returns the top-N_features salient regions.
%   Each row of features: [x, y, scale, saliency]
%
%   img       - grayscale image (double, 0-1 or 0-255)
%   N_features - number of features to return
%   s_min     - minimum scale (radius) in pixels  (default 4)
%   s_max     - maximum scale (radius) in pixels  (default 40)

if nargin < 3, s_min = 4;  end
if nargin < 4, s_max = 40; end

img = double(img);
if max(img(:)) > 1
    img = img / 255;
end
img = round(img * 255);   % integer bins 0-255

[rows, cols] = size(img);
scales = s_min:2:s_max;
n_bins = 16;                % coarse histogram for speed

% Pre-allocate saliency map (x, y, best_scale, best_saliency)
best_H    = zeros(rows, cols);
best_dPds = zeros(rows, cols);
best_s    = zeros(rows, cols);

prev_H = zeros(rows, cols);

for si = 1:length(scales)
    s = scales(si);

    % Build circular mask of radius s
    [mx, my] = meshgrid(-s:s, -s:s);
    mask = (mx.^2 + my.^2) <= s^2;
    n_pix = sum(mask(:));

    % Compute entropy H(s, i) for every pixel using integral images trick:
    % Approximate with box filter, fast enough for course-work quality.
    H_map = compute_entropy_map(img, s, n_bins, rows, cols);

    % Shannon entropy difference |dH/ds| ≈ |H(s) - H(s_prev)|
    if si == 1
        dHds = zeros(rows, cols);
    else
        dHds = abs(H_map - prev_H);
    end

    % Saliency = H * |dP/ds| ≈ H * dHds  (following Kadir & Brady eq.)
    saliency = H_map .* dHds;

    % Update best-scale map
    update = saliency > best_saliency_map_or_zero(best_dPds);
    best_H(update)    = H_map(update);
    best_dPds(update) = saliency(update);
    best_s(update)    = s;

    prev_H = H_map;
end

% Suppress border (largest scale radius) and collect local maxima
border = s_max + 1;
best_dPds(1:border, :) = 0;
best_dPds(end-border+1:end, :) = 0;
best_dPds(:, 1:border) = 0;
best_dPds(:, end-border+1:end) = 0;

% Non-maximum suppression in 5x5 neighbourhood
sal_nms = imregionalmax(best_dPds, 8) .* best_dPds;

% Pick top-N
[vals, idx] = sort(sal_nms(:), 'descend');
valid = vals > 0;
idx = idx(valid);
vals = vals(valid);

n_pick = min(N_features, length(idx));
[ys, xs] = ind2sub([rows, cols], idx(1:n_pick));
scales_out = best_s(idx(1:n_pick));
sals_out   = vals(1:n_pick);

features = [xs(:), ys(:), scales_out(:), sals_out(:)];
end

% -------------------------------------------------------------------------
function H_map = compute_entropy_map(img, s, n_bins, rows, cols)
% Compute per-pixel entropy of circular neighbourhood of radius s.
% Uses a sliding rectangular approximation for speed.
H_map = zeros(rows, cols);

bin_edges = linspace(0, 255, n_bins + 1);

% Pad image
pad = s;
img_pad = padarray(img, [pad, pad], 'replicate');

for r = 1:rows
    for c = 1:cols
        patch = img_pad(r:r+2*pad, c:c+2*pad);
        counts = histcounts(patch(:), bin_edges);
        p = counts / sum(counts);
        p = p(p > 0);
        H_map(r, c) = -sum(p .* log2(p));
    end
end
end

% -------------------------------------------------------------------------
function out = best_saliency_map_or_zero(m)
out = m;
end
