from functools import partial
import numpy as np
from scipy import sparse
from tqdm.auto import tqdm as _tqdm
tqdm = partial(_tqdm, disable=True)


class FactorizationMachine:
    def __init__(
        self,
        n_factors: int = 16,
        n_epochs: int = 20,
        lr: float = 0.01,
        reg_w: float = 0.01,
        reg_v: float = 0.01,
        batch_size: int = 512,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr = lr
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.batch_size = batch_size
        self.random_state = random_state

    def _init_params(self, n_features: int) -> None:
        rng = np.random.default_rng(self.random_state)
        self.w0_: float = 0.0
        self.w_: np.ndarray = np.zeros(n_features, dtype=np.float32)
        self.V_: np.ndarray = (
            0.01 * rng.standard_normal((self.n_factors, n_features))
        ).astype(np.float32)


    def _forward(self, X: sparse.csr_matrix):
        X_sq = X.copy()
        X_sq.data **= 2
        linear = np.asarray(X @ self.w_).ravel()
        Vx = np.asarray(X @ self.V_.T, dtype=np.float32)
        V2x2 = np.asarray(X_sq @ (self.V_ ** 2).T, dtype=np.float32)
        interaction = 0.5 * np.sum(Vx ** 2 - V2x2, axis=1)
        return self.w0_ + linear + interaction, Vx, X_sq


    def _apply_gradients(self, X, errors, Vx, X_sq):
        B = X.shape[0]
        e = errors.astype(np.float32)
        self.w0_ -= self.lr * e.sum() / B
        grad_w = np.asarray(X.T @ e).ravel() / B
        self.w_ -= self.lr * (grad_w + 2 * self.reg_w * self.w_)
        term1 = X.T @ (e[:, None] * Vx)
        term2 = np.asarray(X_sq.T @ e).ravel()
        grad_VT = (np.asarray(term1, dtype=np.float32) - term2[:, None] * self.V_.T) / B
        self.V_ -= self.lr * (grad_VT.T + 2 * self.reg_v * self.V_)

    def _step_mse(self, X, y):
        y_hat, Vx, X_sq = self._forward(X)
        e = (y_hat - y).astype(np.float32)
        self._apply_gradients(X, 2 * e, Vx, X_sq)
        return float(np.mean(e ** 2))

    def _step_bpr(self, X_pos, X_neg):
        y_pos, Vx_pos, Xsq_pos = self._forward(X_pos)
        y_neg, Vx_neg, Xsq_neg = self._forward(X_neg)

        sigma = 1.0 / (1.0 + np.exp(-np.clip(y_pos - y_neg, -30, 30)))
        e_pos = (sigma - 1.0).astype(np.float32)
        e_neg = (1.0 - sigma).astype(np.float32)
        B = X_pos.shape[0]

        self.w0_ -= self.lr * (e_pos.sum() + e_neg.sum()) / B

        grad_w = (np.asarray(X_pos.T @ e_pos).ravel() +
                  np.asarray(X_neg.T @ e_neg).ravel()) / B
        self.w_ -= self.lr * (grad_w + 2 * self.reg_w * self.w_)

        term1 = (X_pos.T @ (e_pos[:, None] * Vx_pos) +
                 X_neg.T @ (e_neg[:, None] * Vx_neg))
        term2 = (np.asarray(Xsq_pos.T @ e_pos).ravel() +
                 np.asarray(Xsq_neg.T @ e_neg).ravel())
        grad_VT = (np.asarray(term1, dtype=np.float32) - term2[:, None] * self.V_.T) / B
        self.V_ -= self.lr * (grad_VT.T + 2 * self.reg_v * self.V_)

        return -float(np.mean(np.log(sigma + 1e-9)))


    def _build_X(self, user_indices, item_indices):
        B = len(user_indices)
        rows = np.arange(B)
        u_block = sparse.csr_matrix(
            (np.ones(B, dtype=np.float32), (rows, user_indices)),
            shape=(B, self._n_users), dtype=np.float32,
        )
        i_block = sparse.csr_matrix(
            (np.ones(B, dtype=np.float32), (rows, item_indices)),
            shape=(B, self._n_items), dtype=np.float32,
        )
        return sparse.hstack(
            [u_block, self._user_side[user_indices],
             i_block, self._item_side[item_indices]],
            format="csr", dtype=np.float32,
        )

    def fit(self, X: sparse.csr_matrix, y: np.ndarray):
        X = sparse.csr_matrix(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self._init_params(X.shape[1])
        rng = np.random.default_rng(self.random_state)
        self.train_rmse_: list[float] = []

        epoch_bar = tqdm(range(self.n_epochs), desc="Training (MSE)", unit="epoch")
        for _ in epoch_bar:
            order  = rng.permutation(X.shape[0])
            losses = []
            for start in range(0, X.shape[0], self.batch_size):
                idx = order[start: start + self.batch_size]
                losses.append(self._step_mse(X[idx], y[idx]))
            rmse = float(np.sqrt(np.mean(losses)))
            self.train_rmse_.append(rmse)
            epoch_bar.set_postfix({"RMSE": f"{rmse:.4f}"})

        return self


    def fit_bpr(self, user_indices, item_indices,
                n_users, n_items, user_side_feat, item_side_feat):
        self._n_users = n_users
        self._n_items = n_items
        self._user_side = sparse.csr_matrix(user_side_feat, dtype=np.float32)
        self._item_side = sparse.csr_matrix(item_side_feat, dtype=np.float32)

        n_features = n_users + self._user_side.shape[1] + n_items + self._item_side.shape[1]
        self._init_params(n_features)

        user_indices = np.asarray(user_indices, dtype=np.int32)
        item_indices = np.asarray(item_indices, dtype=np.int32)
        n_samples = len(user_indices)

        rng = np.random.default_rng(self.random_state)
        self.train_bpr_loss_: list[float] = []

        epoch_bar = tqdm(range(self.n_epochs), desc="Training (BPR)", unit="epoch")
        for _ in epoch_bar:
            order = rng.permutation(n_samples)
            losses = []
            for start in range(0, n_samples, self.batch_size):
                idx = order[start: start + self.batch_size]
                u_batch = user_indices[idx]
                i_batch = item_indices[idx]
                j_batch = rng.integers(0, n_items, size=len(idx), dtype=np.int32)
                X_pos = self._build_X(u_batch, i_batch)
                X_neg = self._build_X(u_batch, j_batch)
                losses.append(self._step_bpr(X_pos, X_neg))

            avg = float(np.mean(losses))
            self.train_bpr_loss_.append(avg)
            epoch_bar.set_postfix({"BPR loss": f"{avg:.4f}"})

        return self

    def predict(self, X: sparse.csr_matrix) -> np.ndarray:
        y_hat, _, _ = self._forward(sparse.csr_matrix(X, dtype=np.float32))
        return y_hat

    def predict_user(self, user_idx: int) -> np.ndarray:
        u_arr = np.full(self._n_items, user_idx, dtype=np.int32)
        i_arr = np.arange(self._n_items, dtype=np.int32)
        return self.predict(self._build_X(u_arr, i_arr))


class FMRecommender(FactorizationMachine):

    def _sample_negative_items(self, users, rng):
        negatives = rng.integers(0, self._n_items, size=len(users), dtype=np.int32)
        while True:
            keys = np.asarray(users, dtype=np.int64) * self._n_items + negatives
            bad = np.isin(keys, self._observed_keys)
            if not bad.any():
                return negatives
            negatives[bad] = rng.integers(0, self._n_items, size=int(bad.sum()), dtype=np.int32)

    def fit_bpr(self, user_indices, item_indices,
                n_users, n_items, user_side_feat, item_side_feat, observed_pairs=None):
        observed_users, observed_items = observed_pairs if observed_pairs is not None else (user_indices, item_indices)
        self._observed_keys = np.unique(np.asarray(observed_users, dtype=np.int64) * n_items + observed_items)
        counts = np.bincount(self._observed_keys // n_items, minlength=n_users)
        self._n_users = n_users
        self._n_items = n_items
        self._user_side = sparse.csr_matrix(user_side_feat, dtype=np.float32)
        self._item_side = sparse.csr_matrix(item_side_feat, dtype=np.float32)

        n_features = n_users + self._user_side.shape[1] + n_items + self._item_side.shape[1]
        self._init_params(n_features)

        user_indices = np.asarray(user_indices, dtype=np.int32)
        item_indices = np.asarray(item_indices, dtype=np.int32)
        n_samples = len(user_indices)

        rng = np.random.default_rng(self.random_state)
        self.train_bpr_loss_: list[float] = []

        epoch_bar = tqdm(range(self.n_epochs), desc="Training (BPR)", unit="epoch")
        for _ in epoch_bar:
            order = rng.permutation(n_samples)
            losses = []
            for start in range(0, n_samples, self.batch_size):
                idx = order[start: start + self.batch_size]
                u_batch = user_indices[idx]
                i_batch = item_indices[idx]
                j_batch = self._sample_negative_items(u_batch, rng)
                X_pos = self._build_X(u_batch, i_batch)
                X_neg = self._build_X(u_batch, j_batch)
                losses.append(self._step_bpr(X_pos, X_neg))

            avg = float(np.mean(losses))
            self.train_bpr_loss_.append(avg)
            epoch_bar.set_postfix({"BPR loss": f"{avg:.4f}"})

        self._cache_scores()
        return self

    def _cache_scores(self):
        nu, ni = self._n_users, self._n_items
        du = self._user_side.shape[1]
        user_block = sparse.hstack([sparse.eye(nu, dtype=np.float32, format="csr"), self._user_side], format="csr")
        item_block = sparse.hstack([sparse.eye(ni, dtype=np.float32, format="csr"), self._item_side], format="csr")

        def expand(X, offset):
            V = self.V_[:, offset:offset + X.shape[1]]
            w = self.w_[offset:offset + X.shape[1]]
            embeddings = np.asarray(X @ V.T)
            squared = X.copy()
            squared.data **= 2
            bias = np.asarray(X @ w).ravel() + .5 * np.sum(embeddings ** 2 - squared @ (V ** 2).T, axis=1)
            return embeddings, bias

        self._user_embedding, self._user_bias = expand(user_block, 0)
        self._item_embedding, self._item_bias = expand(item_block, nu + du)

    def score_batch(self, user_indices):
        return (self.w0_ + self._user_bias[user_indices, None] + self._item_bias[None, :]
                + self._user_embedding[user_indices] @ self._item_embedding.T)
