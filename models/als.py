from functools import partial
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm as _tqdm
tqdm = partial(_tqdm, disable=True)


class ALSExplicit:
    def __init__(self, n_factors=40, n_iters=10, reg=0.1, random_state=42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.random_state = random_state

        self.user_factors = None
        self.item_factors = None

    def _init_factors(self, n_users, n_items):
        rng = np.random.default_rng(self.random_state)
        self.user_factors = 0.01 * rng.standard_normal(size=(n_users, self.n_factors))
        self.item_factors = 0.01 * rng.standard_normal(size=(n_items, self.n_factors))

    def fit(self, R_csr):
        n_users, n_items = R_csr.shape
        self._init_factors(n_users, n_items)
        R_csc = R_csr.tocsc()
        I_k = np.eye(self.n_factors, dtype=np.float32)
        for it in range(self.n_iters):
            for u in tqdm(range(n_users), desc=f"ALS iter {it + 1}/{self.n_iters} [users]", leave=False):
                start, end = R_csr.indptr[u], R_csr.indptr[u + 1]
                item_idx = R_csr.indices[start:end]
                if item_idx.size == 0:
                    continue
                ratings_u = R_csr.data[start:end]

                V_u = self.item_factors[item_idx]
                A = V_u.T @ V_u + self.reg * I_k
                b = V_u.T @ ratings_u

                self.user_factors[u] = np.linalg.solve(A, b)


            for i in tqdm(range(n_items), desc=f"ALS iter {it + 1}/{self.n_iters} [items]", leave=False):
                start, end = R_csc.indptr[i], R_csc.indptr[i + 1]
                user_idx = R_csc.indices[start:end]
                if user_idx.size == 0:
                    continue
                ratings_i = R_csc.data[start:end]

                U_i = self.user_factors[user_idx]
                A = U_i.T @ U_i + self.reg * I_k
                b = U_i.T @ ratings_i

                self.item_factors[i] = np.linalg.solve(A, b)

        return self

    def predict_for_user(self, user_idx):
        return self.item_factors @ self.user_factors[user_idx]

    def predict_single(self, user_idx, item_idx):
        return float(self.user_factors[user_idx] @ self.item_factors[item_idx])
