%% main_fergus2003.m
% Driver script for the Fergus et al. (2003) constellation model.
%
% Directory layout expected:
%   data/
%     motorbikes/   (*.jpg or *.png)
%     airplanes/
%     faces/
%     cars_rear/
%     background/
%
% Usage: run this script from the repo root.
%
% Nikita Letov, McGill ECSE 626 reimplementation

clear; clc; close all;
rng(42);  % reproducibility

%% ---- Settings ----------------------------------------------------------
DATA_ROOT  = fullfile('data');
CATEGORIES = {'motorbikes', 'airplanes', 'faces', 'cars_rear'};
BG_DIR     = fullfile(DATA_ROOT, 'background');

P          = 6;        % number of parts
N_FEAT     = 30;       % max features per image
K_PCA      = 15;       % PCA dimensions
PATCH_SZ   = 11;       % patch size in pixels
S_MIN      = 4;        % min detector scale
S_MAX      = 40;       % max detector scale
N_ITER     = 60;       % EM iterations

%% ---- Load background images -------------------------------------------
fprintf('=== Loading background images ===\n');
bg_files = [dir(fullfile(BG_DIR,'*.jpg')); dir(fullfile(BG_DIR,'*.png'))];
bg_imgs  = load_images({bg_files.name}, BG_DIR);

%% ---- Compute background PCA (will be reused across categories) ---------
fprintf('=== Background feature extraction ===\n');
[~, ~, A_bg, pca_bg] = extract_features(bg_imgs, N_FEAT, PATCH_SZ, K_PCA, S_MIN, S_MAX);

%% ---- Per-category train / test -----------------------------------------
results = struct();

for ci = 1:length(CATEGORIES)
    cat = CATEGORIES{ci};
    fprintf('\n========== Category: %s ==========\n', cat);

    % Load images
    cat_dir  = fullfile(DATA_ROOT, cat);
    files    = [dir(fullfile(cat_dir,'*.jpg')); dir(fullfile(cat_dir,'*.png'))];
    n_total  = length(files);
    idx_all  = randperm(n_total);
    n_train  = floor(n_total / 2);

    train_files = {files(idx_all(1:n_train)).name};
    test_files  = {files(idx_all(n_train+1:end)).name};

    fprintf('Train: %d  |  Test: %d\n', length(train_files), length(test_files));

    % ---- Feature extraction (training) ----------------------------------
    fprintf('  Extracting training features...\n');
    train_imgs = load_images(train_files, cat_dir);
    [X_tr, S_tr, A_tr, pca_model] = extract_features( ...
        train_imgs, N_FEAT, PATCH_SZ, K_PCA, S_MIN, S_MAX);

    % ---- EM learning ----------------------------------------------------
    fprintf('  Running EM learning...\n');
    model = learn_constellation(X_tr, S_tr, A_tr, bg_imgs, A_bg, P, N_ITER, true);
    model.pca = pca_model;

    % ---- Recognition on test set ----------------------------------------
    fprintf('  Recognizing test images...\n');
    test_imgs = load_images(test_files, cat_dir);
    [X_te, S_te, A_te] = extract_features( ...
        test_imgs, N_FEAT, PATCH_SZ, K_PCA, S_MIN, S_MAX, pca_model);

    % Background test images
    n_bg_test = length(test_files);
    bg_test_idx = randperm(length(bg_files), min(n_bg_test, length(bg_files)));
    bg_test_imgs = load_images({bg_files(bg_test_idx).name}, BG_DIR);
    [X_bg_te, S_bg_te, A_bg_te] = extract_features( ...
        bg_test_imgs, N_FEAT, PATCH_SZ, K_PCA, S_MIN, S_MAX, pca_model);

    % Compute likelihood ratios
    R_fg = zeros(length(test_imgs), 1);
    for i = 1:length(test_imgs)
        R_fg(i) = recognize(model, X_te{i}, S_te{i}, A_te{i});
    end

    R_bg = zeros(length(bg_test_imgs), 1);
    for i = 1:length(bg_test_imgs)
        R_bg(i) = recognize(model, X_bg_te{i}, S_bg_te{i}, A_bg_te{i});
    end

    % ROC equal error rate
    labels = [ones(length(R_fg),1); zeros(length(R_bg),1)];
    scores = [R_fg; R_bg];
    eer    = compute_eer(scores, labels);

    fprintf('  ROC Equal Error Rate: %.1f%%\n', eer * 100);

    results.(cat).model  = model;
    results.(cat).R_fg   = R_fg;
    results.(cat).R_bg   = R_bg;
    results.(cat).eer    = eer;

    % ---- Visualise shape model ------------------------------------------
    plot_shape_model(model, cat);
end

%% ---- Summary table -----------------------------------------------------
fprintf('\n\n===== RESULTS SUMMARY =====\n');
fprintf('%-20s  %s\n', 'Dataset', 'Accuracy (%)');
fprintf('%s\n', repmat('-',1,35));
for ci = 1:length(CATEGORIES)
    cat = CATEGORIES{ci};
    acc = (1 - results.(cat).eer) * 100;
    fprintf('%-20s  %.1f\n', cat, acc);
end


%% =========================================================================
%  Helper functions
%% =========================================================================

function imgs = load_images(fnames, dir_path)
imgs = cell(numel(fnames), 1);
for k = 1:numel(fnames)
    img = imread(fullfile(dir_path, fnames{k}));
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    imgs{k} = im2double(img);
end
end

function eer = compute_eer(scores, labels)
% Sweep threshold and find equal error rate
thresholds = unique(scores);
best = inf;
eer  = 0.5;
for t = thresholds'
    tp = sum(scores(labels==1) >= t);
    fn = sum(scores(labels==1) < t);
    fp = sum(scores(labels==0) >= t);
    tn = sum(scores(labels==0) < t);
    fpr = fp / max(fp + tn, 1);
    fnr = fn / max(fn + tp, 1);
    if abs(fpr - fnr) < best
        best = abs(fpr - fnr);
        eer  = (fpr + fnr) / 2;
    end
end
end

function plot_shape_model(model, title_str)
P  = model.P;
mu = model.mu;
Sigma = model.Sigma;

figure('Name', ['Shape model: ' title_str]);
hold on;
colors = lines(P);
for p = 1:P
    mx = mu(2*p - 1);
    my = mu(2*p);
    sx = sqrt(Sigma(2*p-1, 2*p-1));
    sy = sqrt(Sigma(2*p,   2*p));
    draw_ellipse(mx, my, sx, sy, colors(p,:));
    plot(mx, my, 'o', 'Color', colors(p,:), 'MarkerFaceColor', colors(p,:));
    text(mx + 0.01, my + 0.01, sprintf('P%d', p), 'Color', colors(p,:));
end
axis equal; grid on;
title([title_str ' shape model (P=' num2str(P) ')']);
xlabel('x (normalised)'); ylabel('y (normalised)');
hold off;
end

function draw_ellipse(cx, cy, rx, ry, color)
theta = linspace(0, 2*pi, 100);
x = cx + rx * cos(theta);
y = cy + ry * sin(theta);
plot(x, y, '-', 'Color', color, 'LineWidth', 1.5);
end
