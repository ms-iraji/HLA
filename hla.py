import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from ufsm import UFSM_filter

def HLA(X, y, t=30, M=15, K=40, P=0.9, seed=42):
    """Hybrid-based Learning Automata (Algorithm 3)."""
    np.random.seed(seed)
    selected_from_ufsm = UFSM_filter(X, t=t)
    X = X[:, selected_from_ufsm]
    n_features = X.shape[1]
    prob_active = np.ones(n_features) * 0.5
    T_k, k, best_acc = 0.0, 1, 0.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    while k <= K:
        G_cur = [i for i in range(n_features) if np.random.rand() < prob_active[i]]
        if len(G_cur) == 0: G_cur = list(range(min(M, n_features)))
        clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=seed)
        clf.fit(X_train[:, G_cur], y_train)
        acc = accuracy_score(y_val, clf.predict(X_val[:, G_cur]))
        if acc >= T_k:
            for i in G_cur: prob_active[i] = min(prob_active[i] + 0.05, 1.0)
            T_k, best_acc = acc, acc
        else:
            for i in G_cur: prob_active[i] = max(prob_active[i] - 0.05, 0.0)
        if np.mean(prob_active) >= P: break
        k += 1
    rank = sorted(list(zip(selected_from_ufsm, prob_active)), key=lambda x: x[1], reverse=True)
    return rank