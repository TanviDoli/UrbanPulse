import numpy as np

# ============================================================
# URBANPULSE: HMM + MRF ENVIRONMENTAL RISK MODEL
# ============================================================
# HMM: Temporal reasoning
# MRF: Spatial reasoning
# Fusion: Zone-level risk estimation
# ============================================================


# ============================================================
# 1. HIDDEN MARKOV MODEL (HMM)
# ============================================================
# The HMM models how environmental risk changes over time.
#
# P(S_t | O_1:t) ∝ P(O_t | S_t)
#     × Σ P(S_t | S_{t-1}) P(S_{t-1} | O_1:t-1)
#
# S_t  : Hidden state at time t
# O_t  : Observation at time t
# P    : Probability
# ============================================================

def hmm_predict_state(previous_posterior, transition_matrix):
    """
    Predict the next hidden-state probabilities
    using the transition matrix.
    """
    predicted = previous_posterior @ transition_matrix
    return predicted


def hmm_update_observation(predicted, observation_likelihood):
    """
    Update state probabilities using the likelihood
    of the current observation.
    """
    posterior = predicted * observation_likelihood
    return posterior


def normalize_probabilities(probabilities):
    """
    Normalize probabilities so their sum equals 1.
    """
    total = probabilities.sum()

    if total > 0:
        probabilities = probabilities / total

    return probabilities


def hmm_temporal_posterior(
    previous_posterior,
    transition_matrix,
    observation_likelihood
):
    """
    Calculate the posterior probability of hidden states.
    """
    predicted = hmm_predict_state(
        previous_posterior,
        transition_matrix
    )

    posterior = hmm_update_observation(
        predicted,
        observation_likelihood
    )

    posterior = normalize_probabilities(posterior)

    return posterior


def hmm_forecast(previous_posterior, transition_matrix, steps=1):
    """
    Forecast hidden-state probabilities several steps ahead.
    """
    forecast = previous_posterior.copy()

    for step in range(steps):
        forecast = forecast @ transition_matrix

    return normalize_probabilities(forecast)


# ============================================================
# 2. MARKOV RANDOM FIELD (MRF)
# ============================================================
# The MRF models spatial relationships between neighboring zones.
#
# P(S | O) ∝ exp(
#     - Σ_i U_i(S_i, O_i)
#     - Σ_ij V_ij(S_i, S_j)
# )
#
# U_i : Unary cost based on local observations
# V_ij: Pairwise cost between neighboring zones
# S_i : Risk state of zone i
# O_i : Observation from zone i
# ============================================================

def mrf_unary_energy(unary_cost):
    """
    Calculate the total local observation energy.
    """
    return np.sum(unary_cost)


def mrf_pairwise_energy(pairwise_cost):
    """
    Calculate the total neighboring-zone energy.
    """
    return np.sum(pairwise_cost)


def mrf_total_energy(unary_cost, pairwise_cost):
    """
    Combine local and spatial energy terms.
    """
    local_energy = mrf_unary_energy(unary_cost)
    spatial_energy = mrf_pairwise_energy(pairwise_cost)

    total_energy = local_energy + spatial_energy

    return total_energy


def mrf_spatial_probability(unary_cost, pairwise_cost):
    """
    Calculate an unnormalized spatial probability weight.
    """
    energy = mrf_total_energy(
        unary_cost,
        pairwise_cost
    )

    probability = np.exp(-energy)

    return probability


def mrf_spatial_compatibility(neighbor_costs):
    """
    Calculate spatial compatibility from neighboring zones.
    """
    energy = np.sum(neighbor_costs)
    compatibility = np.exp(-energy)

    return compatibility


# ============================================================
# 3. PROBABILISTIC FUSION
# ============================================================
# HMM temporal posterior + MRF spatial compatibility
#                  ↓
#          Zone-level risk posterior
# ============================================================

def fuse_hmm_mrf(hmm_posterior, spatial_compatibility):
    """
    Combine temporal probabilities with spatial weights.
    """
    combined = hmm_posterior * spatial_compatibility

    total = combined.sum()

    if total > 0:
        combined = combined / total

    return combined


def estimate_zone_risk(risk_posterior):
    """
    Identify the most probable risk state.
    """
    risk_state = np.argmax(risk_posterior)
    confidence = np.max(risk_posterior)

    return risk_state, confidence


def calculate_risk_score(risk_posterior):
    """
    Calculate a weighted score from risk-state probabilities.
    """
    states = np.arange(len(risk_posterior))

    score = np.sum(states * risk_posterior)

    return score


# ============================================================
# 4. EXAMPLE INPUTS
# ============================================================

previous_posterior = np.array([
    0.2, 0.5, 0.3
])

transition_matrix = np.array([
    [0.7, 0.2, 0.1],
    [0.2, 0.6, 0.2],
    [0.1, 0.3, 0.6]
])

observation_likelihood = np.array([
    0.8, 0.7, 0.4
])


# ============================================================
# 5. HMM TEMPORAL PREDICTION
# ============================================================

predicted_states = hmm_predict_state(
    previous_posterior,
    transition_matrix
)

updated_states = hmm_update_observation(
    predicted_states,
    observation_likelihood
)

hmm_posterior = normalize_probabilities(
    updated_states
)


# ============================================================
# 6. MRF SPATIAL REASONING
# ============================================================

unary_cost = np.array([
    0.2, 0.4, 0.6
])

pairwise_cost = np.array([
    0.1, 0.3, 0.2
])

local_energy = mrf_unary_energy(unary_cost)

spatial_energy = mrf_pairwise_energy(
    pairwise_cost
)

total_energy = mrf_total_energy(
    unary_cost,
    pairwise_cost
)

spatial_probability = mrf_spatial_probability(
    unary_cost,
    pairwise_cost
)


# ============================================================
# 7. FUSION OF HMM AND MRF
# ============================================================

spatial_compatibility = np.array([
    0.9, 0.8, 0.4
])

risk_posterior = fuse_hmm_mrf(
    hmm_posterior,
    spatial_compatibility
)


# ============================================================
# 8. RISK INTERPRETATION
# ============================================================

risk_state, confidence = estimate_zone_risk(
    risk_posterior
)

risk_score = calculate_risk_score(
    risk_posterior
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("HMM Predicted States:", predicted_states)

print("HMM Temporal Posterior:", hmm_posterior)

print("MRF Local Energy:", local_energy)

print("MRF Spatial Energy:", spatial_energy)

print("MRF Total Energy:", total_energy)

print("MRF Spatial Probability:", spatial_probability)

print("Final Zone Risk Posterior:", risk_posterior)

print("Most Probable Risk State:", risk_state)

print("State Confidence:", confidence)

print("Weighted Risk Score:", risk_score)