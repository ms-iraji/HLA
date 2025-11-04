import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def UFSM_filter(X, t=10):
    """Unsupervised Filter-based Similarity Measure (Algorithm 1)."""
    n_samples, n_features = X.shape
    features = list(range(n_features))
    S = []
    def cosine_sim(a, b):
        return cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))[0, 0]
    beta = np.zeros(n_features)
    for j in features:
        for i in features:
            if i != j:
                beta[j] += cosine_sim(X[:, i], X[:, j])
    avg_beta = beta / (n_features - 1)
    first_feature = np.argmin(avg_beta)
    S.append(first_feature)
    features.remove(first_feature)
    while len(S) < t and features:
        p_values = []
        for j in features:
            beta_j = sum(cosine_sim(X[:, i], X[:, j]) for i in features if i != j)
            alpha_j = sum(cosine_sim(X[:, r], X[:, j]) for r in S)
            denom = (len(features) - len(S) - 1) if (len(features) - len(S) - 1) > 0 else 1
            p_j = (1 / denom) * beta_j - ((1 / denom) + (1 / len(S))) * alpha_j
            p_values.append(p_j)
        min_idx = np.argmin(p_values)
        selected_feature = features[min_idx]
        S.append(selected_feature)
        features.remove(selected_feature)
    return S