import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def WLA(X, y, n_clusters=10, K=40, P=0.9, seed=42):
    """Wrapper-based Learning Automata (Algorithm 2)."""
    np.random.seed(seed)
    n_samples, n_features = X.shape
    kmeans = KMeans(n_clusters=min(n_clusters, n_features), random_state=seed)
    feature_vectors = X.T
    labels = kmeans.fit_predict(feature_vectors)
    cluster_features = [np.random.choice(np.where(labels == c)[0]) for c in np.unique(labels)]
    n_automata = n_features
    prob_active = np.ones(n_automata) * 0.5
    T_k = 0.0
    k = 1
    best_acc = 0.0
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    while k <= K:
        G_cur = [i for i in range(n_features) if np.random.rand() < prob_active[i]]
        if not G_cur:
            G_cur = cluster_features.copy()
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
    rank = sorted(list(enumerate(prob_active)), key=lambda x: x[1], reverse=True)
    return rank