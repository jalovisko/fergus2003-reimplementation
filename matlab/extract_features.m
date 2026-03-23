function [X, S, A, pca_model] = extract_features(img_list, N_feat, patch_sz, k_pca, s_min, s_max, pca_model)
% EXTRACT_FEATURES  Detect salient regions, crop patches, apply PCA.
%
%   [X, S, A, pca_model] = extract_features(img_list, N_feat, patch_sz,
%                                            k_pca, s_min, s_max, pca_model)
%
%   img_list  - cell array of grayscale images (double [0,1])
%   N_feat    - max features per image
%   patch_sz  - patch size in pixels (scalar, square; default 11)
%   k_pca     - PCA dimensions to keep (default 15)
%   s_min/max - scale range for detector
%   pca_model - (optional) struct with .coeff, .mu from training;
%               if absent, PCA is fit on this call (training mode).
%
%   Returns cell arrays X{i}, S{i}, A{i} (one cell per image).
%   A{i} is (N_i x k_pca).

if nargin < 3 || isempty(patch_sz), patch_sz = 11; end
if nargin < 4 || isempty(k_pca),    k_pca    = 15; end
if nargin < 5 || isempty(s_min),    s_min    = 4;  end
if nargin < 6 || isempty(s_max),    s_max    = 40; end

n_imgs = numel(img_list);
X = cell(n_imgs, 1);
S = cell(n_imgs, 1);
raw_patches = cell(n_imgs, 1);

half = floor(patch_sz / 2);

for i = 1:n_imgs
    img = img_list{i};
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = im2double(img);

    feats = kadir_brady(img, N_feat, s_min, s_max);  % [x, y, s, sal]
    if isempty(feats)
        X{i} = zeros(0, 2);
        S{i} = zeros(0, 1);
        raw_patches{i} = zeros(0, patch_sz^2);
        continue;
    end

    [h, w] = size(img);
    valid = feats(:,1) > half & feats(:,1) <= w - half & ...
            feats(:,2) > half & feats(:,2) <= h - half;
    feats = feats(valid, :);

    n = size(feats, 1);
    patches = zeros(n, patch_sz^2);
    for j = 1:n
        cx = round(feats(j, 1));
        cy = round(feats(j, 2));
        patch = img(cy - half : cy + half, cx - half : cx + half);
        patch = imresize(patch, [patch_sz, patch_sz]);
        patches(j, :) = patch(:)';
    end

    X{i} = feats(:, 1:2);          % [x, y]
    S{i} = log(feats(:, 3));        % log-scale
    raw_patches{i} = patches;
end

% ---- PCA ----------------------------------------------------------------
if nargin < 7 || isempty(pca_model)
    % Training mode: fit PCA on all patches
    all_patches = cell2mat(raw_patches);
    mu = mean(all_patches, 1);
    centered = bsxfun(@minus, all_patches, mu);
    [coeff, ~, ~] = svds(centered, k_pca);
    pca_model.coeff = coeff;
    pca_model.mu    = mu;
end

% Project
for i = 1:n_imgs
    if isempty(raw_patches{i})
        A{i} = zeros(0, k_pca);
    else
        centered = bsxfun(@minus, raw_patches{i}, pca_model.mu);
        A{i} = centered * pca_model.coeff;
    end
end
end
