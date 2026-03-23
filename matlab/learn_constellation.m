function model = learn_constellation(X, S, A, X_bg, A_bg, P, n_iter, verbose)
% LEARN_CONSTELLATION  EM learning for the Fergus et al. constellation model.
%
%   model = learn_constellation(X, S, A, X_bg, A_bg, P, n_iter, verbose)
%
%   X, S, A   - cell arrays (one per training image): locations [Nx2],
%               log-scales [Nx1], PCA appearances [NxK]
%   X_bg, A_bg - cell arrays for background images (used to init bg app.)
%   P          - number of parts (default 6)
%   n_iter     - EM iterations (default 60)
%   verbose    - print progress (default true)
%
%   model struct fields:
%     mu, Sigma    - shape: mean [2Px1], cov [2Px2P]
%     c, V         - appearance per part: {Px1} cell of [Kx1], [KxK] diag
%     c_bg, V_bg   - background appearance
%     t, U         - scale per part: [Px1], [Px1] (log-scale mean, var)
%     M            - Poisson mean for n_background features
%     occ          - occlusion probability table [2^P x 1]
%     P, K         - model dimensions

if nargin < 6 || isempty(P),       P       = 6;    end
if nargin < 7 || isempty(n_iter),  n_iter  = 60;   end
if nargin < 8 || isempty(verbose), verbose = true; end

n_train = numel(X);
K       = size(A{1}, 2);   % PCA dimensions

% ---- Initialise parameters ----------------------------------------------
% Background appearance from background images
all_bg_A = cell2mat(A_bg);
if isempty(all_bg_A)
    c_bg = zeros(K, 1);
    V_bg = eye(K);
else
    c_bg = mean(all_bg_A, 1)';
    V_bg = diag(max(var(all_bg_A, 0, 1), 1e-6));
end

% Foreground appearance: random init from training patches
all_A = cell2mat(A);
rand_idx = randperm(size(all_A, 1), min(P, size(all_A, 1)));
c = cell(P, 1);
V = cell(P, 1);
for p = 1:P
    c{p} = all_A(rand_idx(p), :)';
    V{p} = diag(max(var(all_A, 0, 1) * 2, 1e-4));
end

% Shape: random mean in normalised coords, large initial covariance
mu    = randn(2 * P, 1) * 0.1;
Sigma = eye(2 * P) * 0.3;

% Scale
all_S = cell2mat(S);
t = mean(all_S) * ones(P, 1);
U = var(all_S)  * ones(P, 1);
U = max(U, 1e-4);

% Poisson mean M for background features
avg_N = mean(cellfun(@(x) size(x,1), X));
M = max(avg_N - P, 1);

% Occlusion table: uniform initially
occ_states = dec2bin(0:2^P-1) - '0';   % [2^P x P]
occ_prob   = ones(2^P, 1) / 2^P;

% ---- EM iterations -------------------------------------------------------
for iter = 1:n_iter
    if verbose && mod(iter, 10) == 0
        fprintf('EM iter %d / %d\n', iter, n_iter);
    end

    % ---- E-step: compute responsibilities for each image ----------------
    % Accumulate sufficient statistics
    ss_A_sum  = zeros(K, P);       % weighted sum of appearances per part
    ss_A_sq   = zeros(K, P);       % weighted sum of sq appearances per part
    ss_XS     = zeros(2*P, 1);     % weighted location sum
    ss_XX     = zeros(2*P, 2*P);   % weighted outer product of locations
    ss_t      = zeros(P, 1);       % weighted scale sum
    ss_tt     = zeros(P, 1);       % weighted scale sq sum
    ss_d      = zeros(2^P, 1);     % occlusion pattern counts
    ss_n_bg   = 0;                 % total background features
    ss_weight = zeros(P, 1);       % total responsibility per part

    log_lik_total = 0;

    for i = 1:n_train
        Xi = X{i};
        Si = S{i};
        Ai = A{i};
        N  = size(Xi, 1);
        if N < P, continue; end

        % Enumerate hypotheses via greedy top-K search
        [hyps, log_w] = enumerate_hypotheses(Xi, Si, Ai, ...
            mu, Sigma, c, V, c_bg, V_bg, t, U, M, occ_prob, occ_states, P);

        if isempty(hyps), continue; end

        % Normalise weights
        log_w = log_w - max(log_w);
        w     = exp(log_w);
        w     = w / (sum(w) + 1e-300);

        log_lik_total = log_lik_total + log(sum(exp(log_w)) + 1e-300);

        % Accumulate sufficient statistics
        for hi = 1:size(hyps, 1)
            h   = hyps(hi, :);   % 1xP, 0 = occluded
            d   = (h > 0);
            whi = w(hi);

            % Occlusion table
            occ_idx = bin2dec(num2str(d)) + 1;
            ss_d(occ_idx) = ss_d(occ_idx) + whi;

            % Scale-normalised locations for shape
            % Find reference feature (first non-occluded part)
            ref_p = find(d, 1);
            if isempty(ref_p), continue; end
            s_ref = Si(h(ref_p));

            loc_norm = zeros(2*P, 1);
            for p = 1:P
                if d(p)
                    dx = Xi(h(p), 1) - Xi(h(ref_p), 1);
                    dy = Xi(h(p), 2) - Xi(h(ref_p), 2);
                    loc_norm(2*p-1:2*p) = [dx; dy] * exp(-s_ref);
                else
                    loc_norm(2*p-1:2*p) = mu(2*p-1:2*p);  % impute
                end
            end
            ss_XS = ss_XS + whi * loc_norm;
            ss_XX = ss_XX + whi * (loc_norm * loc_norm');

            % Appearance
            for p = 1:P
                if d(p)
                    a_feat = Ai(h(p), :)';
                    ss_A_sum(:, p)  = ss_A_sum(:, p)  + whi * a_feat;
                    ss_A_sq(:, p)   = ss_A_sq(:, p)   + whi * (a_feat.^2);
                    ss_weight(p)    = ss_weight(p) + whi;
                end
            end

            % Scale
            for p = 1:P
                if d(p)
                    ss_t(p)  = ss_t(p)  + whi * Si(h(p));
                    ss_tt(p) = ss_tt(p) + whi * Si(h(p))^2;
                end
            end

            % Background features count
            bg_count = N - sum(d);
            ss_n_bg = ss_n_bg + whi * bg_count;
        end
    end

    % ---- M-step: update parameters ---------------------------------------
    % Shape
    w_total = sum(ss_weight) + 1e-10;
    mu      = ss_XS / w_total;
    Sigma   = ss_XX / w_total - mu * mu' + eye(2*P) * 1e-4;
    Sigma   = (Sigma + Sigma') / 2;   % symmetrize

    % Appearance
    for p = 1:P
        wp = ss_weight(p) + 1e-10;
        c{p} = ss_A_sum(:, p) / wp;
        v_diag = ss_A_sq(:, p) / wp - c{p}.^2;
        V{p} = diag(max(v_diag, 1e-4));
    end

    % Scale
    for p = 1:P
        wp = ss_weight(p) + 1e-10;
        t(p) = ss_t(p) / wp;
        U(p) = max(ss_tt(p) / wp - t(p)^2, 1e-4);
    end

    % Poisson mean
    M = max(ss_n_bg / n_train, 1);

    % Occlusion table
    occ_prob = ss_d / (sum(ss_d) + 1e-10);
    occ_prob = max(occ_prob, 1e-6);
    occ_prob = occ_prob / sum(occ_prob);

    if verbose && mod(iter, 10) == 0
        fprintf('  log-lik = %.2f\n', log_lik_total);
    end
end

% ---- Package model -------------------------------------------------------
model.mu       = mu;
model.Sigma    = Sigma;
model.c        = c;
model.V        = V;
model.c_bg     = c_bg;
model.V_bg     = V_bg;
model.t        = t;
model.U        = U;
model.M        = M;
model.occ_prob = occ_prob;
model.occ_states = occ_states;
model.P        = P;
model.K        = K;
end


% =========================================================================
function [hyps, log_w] = enumerate_hypotheses(X, S, A, ...
    mu, Sigma, c, V, c_bg, V_bg, t, U, M, occ_prob, occ_states, P)
% Greedy top-hypothesis search (simplified A* substitute).
% Returns top hypotheses and their unnormalised log-weights.

N = size(X, 1);
MAX_HYPS = min(500, nchoosek(N, min(P, N)));

% Score each feature for each part
log_app = zeros(N, P);
log_app_bg = zeros(N, 1);
for j = 1:N
    a = A(j, :)';
    log_app_bg(j) = log_mvn_diag(a, c_bg, diag(V_bg));
    for p = 1:P
        log_app(j, p) = log_mvn_diag(a, c{p}, diag(V{p}));
    end
end

log_scale = zeros(N, P);
for j = 1:N
    for p = 1:P
        log_scale(j, p) = log_norm1d(S(j), t(p), U(p));
    end
end

% Greedy: for each part pick best-scoring non-overlapping feature
% Then score the full hypothesis
best_hyps = zeros(MAX_HYPS, P, 'int32');
log_ws    = -inf(MAX_HYPS, 1);

% Generate random hypotheses (Monte Carlo enumerate)
n_mc = MAX_HYPS;
used_hyps = containers.Map('KeyType', 'char', 'ValueType', 'logical');

count = 0;
attempts = 0;
while count < n_mc && attempts < n_mc * 10
    attempts = attempts + 1;
    h = zeros(1, P, 'int32');
    available = 1:N;
    for p = 1:P
        if rand() < 0.15   % allow occlusion
            h(p) = 0;
        else
            if isempty(available)
                h(p) = 0;
            else
                % Weighted random selection by appearance + scale score
                scores = log_app(available, p) + log_scale(available, p);
                scores = scores - max(scores);
                probs = exp(scores);
                probs = probs / sum(probs);
                sel = randsample(length(available), 1, true, probs);
                h(p) = available(sel);
                available(available == h(p)) = [];
            end
        end
    end

    key = num2str(h);
    if isKey(used_hyps, key), continue; end
    used_hyps(key) = true;

    lw = score_hypothesis(h, X, S, A, mu, Sigma, c, V, ...
                          c_bg, V_bg, t, U, M, occ_prob, occ_states, P, N);
    count = count + 1;
    best_hyps(count, :) = h;
    log_ws(count) = lw;
end

best_hyps = best_hyps(1:count, :);
log_ws    = log_ws(1:count);

% Keep top-50
[log_ws, ord] = sort(log_ws, 'descend');
keep = min(50, count);
hyps = double(best_hyps(ord(1:keep), :));
log_w = log_ws(1:keep);
end


% =========================================================================
function lw = score_hypothesis(h, X, S, A, mu, Sigma, c, V, ...
                                c_bg, V_bg, t, U, M, occ_prob, occ_states, P, N)
% Compute log-weight of hypothesis h.
d   = (h > 0);
f   = sum(d);
n   = N - f;

% Occlusion term
occ_row = find(all(bsxfun(@eq, occ_states, d), 2));
if isempty(occ_row)
    lw = -inf; return;
end
log_occ = log(occ_prob(occ_row) + 1e-300);

% Poisson term: log p_Poisson(n|M) - log p_Poisson(N|M)
log_poisson = n * log(M + 1e-10) - M - gammaln(n + 1) ...
            - (N * log(M + 1e-10) - M - gammaln(N + 1));

% Binomial book-keeping
log_binom = -log(nchoosek(N, f) + 1e-300);

% Appearance ratio
log_app = 0;
for p = 1:P
    if d(p)
        a = A(h(p), :)';
        log_app = log_app ...
            + log_mvn_diag(a, c{p}, diag(V{p})) ...
            - log_mvn_diag(a, c_bg, diag(V_bg));
    end
end

% Shape term (scale-normalised)
ref_p = find(d, 1);
if isempty(ref_p)
    log_shape = 0;
else
    s_ref = S(h(ref_p));
    loc = zeros(2*P, 1);
    for p = 1:P
        if d(p)
            dx = X(h(p), 1) - X(h(ref_p), 1);
            dy = X(h(p), 2) - X(h(ref_p), 2);
            loc(2*p-1:2*p) = [dx; dy] * exp(-s_ref);
        else
            loc(2*p-1:2*p) = mu(2*p-1:2*p);
        end
    end

    % Only use non-occluded parts in shape
    idx = [];
    for p = 1:P
        if d(p), idx = [idx, 2*p-1, 2*p]; end %#ok
    end
    if length(idx) >= 2
        mu_sub = mu(idx);
        S_sub  = Sigma(idx, idx) + eye(length(idx)) * 1e-6;
        log_shape = log_mvn_full(loc(idx), mu_sub, S_sub) ...
                  + f * log(1000);  % image area alpha (normalised)
    else
        log_shape = 0;
    end
end

% Scale ratio
log_scale = 0;
for p = 1:P
    if d(p)
        log_scale = log_scale ...
            + log_norm1d(S(h(p)), t(p), U(p)) ...
            - log(-log(rand() + 1e-10) + 1e-10); % uniform background approx
    end
end
log_scale = log_scale + f * log(10);  % range r ~ 10 for log-scale

lw = log_occ + log_poisson + log_binom + log_app + log_shape + log_scale;
end


% =========================================================================
function lp = log_mvn_diag(x, mu, sigma_vec)
% Log of N(x; mu, diag(sigma_vec))
K  = length(x);
d  = x - mu;
lp = -0.5 * (K * log(2*pi) + sum(log(sigma_vec + 1e-300)) ...
             + sum(d.^2 ./ (sigma_vec + 1e-300)));
end

function lp = log_mvn_full(x, mu, Sigma)
K  = length(x);
d  = x - mu;
[L, flag] = chol(Sigma, 'lower');
if flag ~= 0
    lp = -inf; return;
end
lp = -0.5 * (K * log(2*pi) + 2*sum(log(diag(L))) + ...
             sum((L \ d).^2));
end

function lp = log_norm1d(x, mu, sigma2)
lp = -0.5 * (log(2*pi * sigma2 + 1e-300) + (x - mu)^2 / (sigma2 + 1e-300));
end
