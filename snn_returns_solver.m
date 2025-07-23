function [weights, selected_idx, convergence_data] = snn_returns_solver(mean_ret, cov_mat, params)

n_stocks = length(mean_ret);

% Initialize neuron population for each asset
population = init_population(n_stocks, params.pop_size, params.init_method);

best_sharpe = -inf;
best_weights = zeros(n_stocks, 1);

% Track convergence for analysis/plots
convergence_data = struct( ...
    'sharpe_history', zeros(params.n_epochs, 1), ...
    'weights_history', zeros(params.n_epochs, n_stocks), ...
    'n_selected_history', zeros(params.n_epochs, 1), ...
    'active_neurons_history', zeros(params.n_epochs, 1), ...
    'activation_rate_history', zeros(params.n_epochs, 1), ...
    'neuron_activation_matrix', zeros(params.n_epochs, n_stocks), ...
    'cumulative_activations', zeros(1, n_stocks) ...
);

current_threshold = params.threshold;
current_tau = params.tau;

for epoch = 1:params.n_epochs
    potentials = zeros(params.pop_size, n_stocks);
    total_activated = 0;
    epoch_activations = zeros(1, n_stocks);

    for i = 1:params.pop_size
        if isfield(params, 'adaptive_risk_aversion') && params.adaptive_risk_aversion
            risk_factor = max(0.1, min(0.9, 1 - epoch / params.n_epochs));
        else
            risk_factor = params.risk_aversion;
        end

        input_current = risk_factor * mean_ret' - (1 - risk_factor) * diag(cov_mat)';

        % Update neuron potentials with decay, input, and noise
        potentials(i, :) = current_tau * abs(population(i, :)) + input_current + params.noise_factor * randn(1, n_stocks);

        % Neuron "fires" if potential exceeds threshold
        spikes = potentials(i, :) > current_threshold;
        total_activated = total_activated + sum(spikes);
        epoch_activations = epoch_activations + spikes;

        % Update population based on spikes
        if any(spikes)
            population(i, spikes) = abs(population(i, spikes) + params.learning_rate * input_current(spikes));
            % Lateral inhibition: suppress non-spiking neurons
            if isfield(params, 'lateral_inhibition') && params.lateral_inhibition
                non_spikes = ~spikes;
                population(i, non_spikes) = population(i, non_spikes) * 0.95;
            end
        end
    end

    % Store neuron activation data for this epoch
    convergence_data.active_neurons_history(epoch) = total_activated;
    convergence_data.activation_rate_history(epoch) = total_activated / (params.pop_size * n_stocks);
    convergence_data.neuron_activation_matrix(epoch, :) = epoch_activations;
    convergence_data.cumulative_activations = convergence_data.cumulative_activations + epoch_activations;

    % Decode neuron population to portfolio weights
    epoch_weights = decode_weights(population, cov_mat, params.decoding_method);

    % Apply transaction cost if previous weights are available
    if isfield(params, 'prev_weights') && ~isempty(params.prev_weights) && isfield(params, 'transaction_cost')
        [epoch_weights, trans_cost] = apply_transaction_cost(epoch_weights, params.prev_weights, params.transaction_cost);
    else
        trans_cost = 0;
    end

    % Evaluate portfolio: compute Sharpe ratio and number selected
    [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat, trans_cost);

    % Enforce minimum cardinality
    if n_selected < params.cardinality(1)
        epoch_weights = enforce_min_selection(epoch_weights, params.cardinality(1));
        [sharpe, n_selected] = evaluate_portfolio(epoch_weights, mean_ret, cov_mat, trans_cost);
    end

    % Track best solution
    if sharpe > best_sharpe
        best_sharpe = sharpe;
        best_weights = epoch_weights;
    end

    % Store convergence data
    convergence_data.sharpe_history(epoch) = sharpe;
    convergence_data.weights_history(epoch, :) = epoch_weights';
    convergence_data.n_selected_history(epoch) = n_selected;

    % Decay neuron firing threshold over epochs
    current_threshold = max(params.min_threshold, current_threshold * params.threshold_decay);

    % Optionally adapt tau (decay) over epochs
    if isfield(params, 'adaptive_tau') && params.adaptive_tau
        current_tau = params.tau * (1 - 0.5 * epoch / params.n_epochs);
    end
end

% Enforce cardinality on best weights found
[weights, selected_idx] = apply_cardinality(best_weights, params.cardinality);

% Normalize weights to sum to 1
weights = weights / sum(weights);

end

%% Helper Functions

function pop = init_population(n_stocks, pop_size, method)
% Initialize neuron population matrix
switch lower(method)
    case 'uniform'
        pop = rand(pop_size, n_stocks) * 0.1;
    case 'normal'
        pop = abs(randn(pop_size, n_stocks) * 0.1);
    case 'xavier'
        scale = sqrt(2 / (n_stocks + pop_size));
        pop = abs(randn(pop_size, n_stocks) * scale);
    otherwise
        pop = abs(randn(pop_size, n_stocks) * 0.1);
end
end

function weights = decode_weights(population, cov_mat, method)
% Convert neuron population to portfolio weights
raw_weights = mean(population, 1)';
switch lower(method)
    case 'simple'
        weights = abs(raw_weights);
        weights = weights / sum(weights + eps);
    case 'volatility'
        vol_scaling = 1 ./ sqrt(diag(cov_mat));
        weights = abs(raw_weights) .* vol_scaling;
        weights = weights / sum(weights + eps);
    case 'softmax'
        temp = 2.0;
        exp_weights = exp(abs(raw_weights) / temp);
        weights = exp_weights / sum(exp_weights + eps);
    otherwise
        vol_scaling = 1 ./ sqrt(diag(cov_mat));
        weights = abs(raw_weights) .* vol_scaling;
        weights = weights / sum(weights + eps);
end
end

function [weights, selected_idx] = apply_cardinality(weights, card_range)
% Enforce minimum and maximum number of assets
[sorted_weights, rank] = sort(weights, 'descend');
min_card = card_range(1);

if length(card_range) > 1
    max_card = card_range(2);
    best_k = min_card;
    best_sharpe = -inf;
    % Try all cardinalities in range and pick the best by diversification
    for k = min_card:max_card
        test_idx = rank(1:k);
        test_weights = zeros(size(weights));
        test_weights(test_idx) = sorted_weights(1:k);
        test_weights = test_weights / sum(test_weights + eps);
        herfindahl = sum(test_weights.^2);
        effective_k = 1 / herfindahl;
        if effective_k > best_sharpe
            best_sharpe = effective_k;
            best_k = k;
        end
    end
    target = best_k;
else
    target = min_card;
end

selected_idx = rank(1:target);
final_weights = zeros(size(weights));
final_weights(selected_idx) = weights(selected_idx);
weights = final_weights / (sum(final_weights) + eps);
end

function [sharpe, n_selected] = evaluate_portfolio(weights, mean_ret, cov_mat, trans_cost)
% Compute Sharpe ratio and number of selected assets
valid = weights > 1e-4;
n_selected = sum(valid);
if n_selected < 1
    sharpe = -inf;
    return
end
ret_vector = mean_ret(valid);
weight_vector = weights(valid);
portfolio_return = ret_vector' * weight_vector - trans_cost;
portfolio_risk = sqrt(weight_vector' * cov_mat(valid, valid) * weight_vector);
sharpe = portfolio_return / (portfolio_risk + 1e-6);
end

function weights = enforce_min_selection(weights, min_stocks)
% Ensure at least min_stocks assets are present in the portfolio
[~, idx] = sort(weights, 'descend');
min_weight = 1e-3;
weights(idx(1:min_stocks)) = max(weights(idx(1:min_stocks)), min_weight);
weights = weights / sum(weights + eps);
end

function [adjusted_weights, cost] = apply_transaction_cost(new_weights, prev_weights, cost_rate)
% Penalize excessive turnover by applying transaction costs
turnover = sum(abs(new_weights - prev_weights));
cost = cost_rate * turnover;
max_turnover = 0.5; % Limit max turnover per period
if turnover > max_turnover
    adjustment_factor = max_turnover / turnover;
    adjusted_weights = prev_weights + adjustment_factor * (new_weights - prev_weights);
    cost = cost_rate * max_turnover;
else
    adjusted_weights = new_weights;
end
end