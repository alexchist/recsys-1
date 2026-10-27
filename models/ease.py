import numpy as np
from scipy import sparse, linalg


class EASE:
    def __init__(self, regularization=100.0):
        self.regularization = regularization

    def fit(self, R):
        self.R = sparse.csr_matrix(R, dtype=np.float32)
        G = (self.R.T @ self.R).toarray()
        diagonal = np.diag_indices_from(G)
        G[diagonal] += self.regularization
        factor = linalg.cho_factor(G, lower=True, overwrite_a=True, check_finite=False)
        P = linalg.cho_solve(factor, np.eye(G.shape[0], dtype=np.float32),
                            overwrite_b=True, check_finite=False)
        P /= -np.diag(P).copy()[None, :]
        P[diagonal] = 0
        self.B = P
        return self

    def score_batch(self, user_indices):
        return self.R[user_indices] @ self.B
