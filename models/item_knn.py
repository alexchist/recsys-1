import numpy as np
from scipy import sparse
from tqdm.auto import tqdm


def cosine_similarity(X, dense_output=False):
    X = sparse.csr_matrix(X, dtype=np.float32)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    inv_norms = np.divide(1.0, norms, out=np.zeros_like(norms), where=norms > 0)
    normalized = sparse.diags(inv_norms) @ X
    similarities = (normalized @ normalized.T).tocsr()
    return similarities.toarray() if dense_output else similarities


class ItemKNN:
    def __init__(self, k_neighbors=50):
        self.k = k_neighbors
        self.sim_matrix = None
        self.R = None

    def fit(self, R):

        self.R = sparse.csr_matrix(R, dtype=np.float32, copy=True)
        self.R.sum_duplicates()
        self.R.eliminate_zeros()
        R = self.R
        item_sim = cosine_similarity(R.T, dense_output=False)
        item_sim.setdiag(0)
        item_sim.eliminate_zeros()
        item_sim = item_sim.toarray()

        for i in range(item_sim.shape[0]):
            row = item_sim[i]
            if self.k < len(row):
                threshold = np.partition(row, -self.k)[-self.k]
                row[row < threshold] = 0.0
                if threshold > 0:
                    ties = np.flatnonzero(row == threshold)
                    remaining = self.k - np.count_nonzero(row > threshold)
                    row[ties[remaining:]] = 0.0
        self.sim_matrix = sparse.csr_matrix(item_sim)
        return self

    def recommend(self, user_ids, train_user_items, idx2item, user2idx,
                  item2idx, n=10, batch_size=256, show_progress=False):
        user_ids = list(user_ids)
        recommendations = {}

        for start in tqdm(range(0, len(user_ids), batch_size),
                          desc="i2i kNN recommend", disable=not show_progress):
            batch = user_ids[start:start + batch_size]
            scores = self.R[[user2idx[uid] for uid in batch]] @ self.sim_matrix
            for offset, uid in enumerate(batch):
                user_scores = np.asarray(scores[offset].todense()).flatten()

                seen_idx = [item2idx[iid] for iid in train_user_items.get(uid, set()) if iid in item2idx]
                user_scores[seen_idx] = -np.inf

                top_idx = np.argsort(-user_scores, kind="stable")
                top_idx = top_idx[np.isfinite(user_scores[top_idx])][:n]
                recommendations[uid] = [idx2item[i] for i in top_idx]

        return recommendations
