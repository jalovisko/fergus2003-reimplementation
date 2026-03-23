function [R, best_hyp] = recognize(model, X, S, A)
% RECOGNIZE  Compute Bayesian likelihood ratio R for a single image.
%
%   [R, best_hyp] = recognize(model, X, S, A)
%
%   model    - struct from learn_constellation
%   X        - [Nx2] feature locations
%   S        - [Nx1] log-scales
%   A        - [NxK] PCA appearances
%
%   R        - scalar log-likelihood ratio (positive = object present)
%   best_hyp - [1xP] assignment vector for best hypothesis

P = model.P;
N = size(X, 1);

if N < 1
    R = -inf;
    best_hyp = zeros(1, P);
    return;
end

[hyps, log_w] = enumerate_hypotheses_rec(X, S, A, model, N);

if isempty(hyps)
    R = -inf;
    best_hyp = zeros(1, P);
    return;
end

% Log-sum-exp over hypotheses gives log p(X,S,A | theta)
log_w_max = max(log_w);
log_fg = log_w_max + log(sum(exp(log_w - log_w_max)));

% Background model: all features are background
log_bg = compute_background_loglik(X, S, A, model, N);

% Prior ratio = 1  =>  log R = log_fg - log_bg
R = log_fg - log_bg;

[~, best_idx] = max(log_w);
best_hyp = hyps(best_idx, :);
end


% =========================================================================
function [hyps, log_w] = enumerate_hypotheses_rec(X, S, A, model, N)
P          = model.P;
mu         = model.mu;
Sigma      = model.Sigma;
c          = model.c;
V          = model.V;
c_bg       = model.c_bg;
V_bg       = model.V_bg;
t          = model.t;
U          = model.U;
M          = model.M;
occ_prob   = model.occ_prob;
occ_states = model.occ_states;

MAX_HYPS = 200;

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

used_hyps = containers.Map('KeyType','char','ValueType','logical');
count = 0;
hyps  = zeros(MAX_HYPS, P, 'int32');
log_ws = -inf(MAX_HYPS, 1);

n_mc = MAX_HYPS;
attempts = 0;
while count < n_mc && attempts < n_mc * 15
    attempts = attempts + 1;
    h = zeros(1, P, 'int32');
    available = 1:N;
    for p = 1:P
        if rand() < 0.15
            h(p) = 0;
        else
            if isempty(available)
                h(p) = 0;
            else
                scores = log_app(available, p) + log_scale(available, p);
                scores = scores - max(scores);
                probs  = exp(scores);
                probs  = probs / sum(probs);
                sel    = randsample(length(available), 1, true, probs);
                h(p)   = available(sel);
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
    hyps(count, :) = h;
    log_ws(count)  = lw;
end

hyps   = double(hyps(1:count, :));
log_ws = log_ws(1:count);
[log_ws, ord] = sort(log_ws, 'descend');
keep  = min(50, count);
hyps  = hyps(ord(1:keep), :);
log_w = log_ws(1:keep);
end


% =========================================================================
function log_bg = compute_background_loglik(X, S, A, model, N)
% All N features come from background model
log_bg = 0;
for j = 1:N
    a = A(j, :)';
    log_bg = log_bg + log_mvn_diag(a, model.c_bg, diag(model.V_bg));
end
% Add Poisson term for N features from background
log_bg = log_bg + N * log(model.M + 1e-10) - model.M - gammaln(N + 1);
end


% =========================================================================
%  --- Shared utility functions (duplicated so file is self-contained) ----
function lp = log_mvn_diag(x, mu, sigma_vec)
K  = length(x);
d  = x - mu;
lp = -0.5 * (K * log(2*pi) + sum(log(sigma_vec + 1e-300)) ...
             + sum(d.^2 ./ (sigma_vec + 1e-300)));
end

function lp = log_mvn_full(x, mu, Sigma)
K  = length(x);
d  = x - mu;
[L, flag] = chol(Sigma, 'lower');
if flag ~= 0, lp = -inf; return; end
lp = -0.5 * (K * log(2*pi) + 2*sum(log(diag(L))) + sum((L\d).^2));
end

function lp = log_norm1d(x, mu, sigma2)
lp = -0.5 * (log(2*pi*sigma2 + 1e-300) + (x-mu)^2 / (sigma2 + 1e-300));
end

function lw = score_hypothesis(h, X, S, A, mu, Sigma, c, V, ...
                                c_bg, V_bg, t, U, M, occ_prob, occ_states, P, N)
d = (h > 0);
f = sum(d);
n = N - f;

occ_row = find(all(bsxfun(@eq, occ_states, d), 2));
if isempty(occ_row), lw = -inf; return; end
log_occ = log(occ_prob(occ_row) + 1e-300);

log_poisson = n*log(M+1e-10) - M - gammaln(n+1) ...
            -(N*log(M+1e-10) - M - gammaln(N+1));
log_binom = -log(nchoosek(max(N,1), max(f,0)) + 1e-300);

log_app = 0;
for p = 1:P
    if d(p)
        a = A(h(p), :)';
        log_app = log_app + log_mvn_diag(a, c{p}, diag(V{p})) ...
                          - log_mvn_diag(a, c_bg, diag(V_bg));
    end
end

ref_p = find(d, 1);
if isempty(ref_p)
    log_shape = 0;
else
    s_ref = S(h(ref_p));
    loc = zeros(2*P, 1);
    for p = 1:P
        if d(p)
            dx = X(h(p),1) - X(h(ref_p),1);
            dy = X(h(p),2) - X(h(ref_p),2);
            loc(2*p-1:2*p) = [dx;dy]*exp(-s_ref);
        else
            loc(2*p-1:2*p) = mu(2*p-1:2*p);
        end
    end
    idx = [];
    for p = 1:P
        if d(p), idx = [idx, 2*p-1, 2*p]; end
    end
    if length(idx) >= 2
        mu_sub = mu(idx);
        S_sub  = Sigma(idx,idx) + eye(length(idx))*1e-6;
        log_shape = log_mvn_full(loc(idx), mu_sub, S_sub) + f*log(1000);
    else
        log_shape = 0;
    end
end

log_scale = 0;
for p = 1:P
    if d(p)
        log_scale = log_scale + log_norm1d(S(h(p)), t(p), U(p));
    end
end
log_scale = log_scale + f * log(10);

lw = log_occ + log_poisson + log_binom + log_app + log_shape + log_scale;
end
