import numpy as np
from scipy import sparse
from .als import ALSExplicit


def custom_targets(train, reference_timestamp, gamma=1.0, tau_days=180.0):
    ratings = train["rating"].to_numpy(dtype=np.float32)
    means = train.groupby("user_id")["rating"].transform("mean").to_numpy(dtype=np.float32)
    ages = (reference_timestamp - train["timestamp"].to_numpy()) / 86400.0
    return (ratings + gamma * (ratings - means) * np.exp(-ages / tau_days)).astype(np.float32)


class CustomTargetALS(ALSExplicit):
    def fit_frame(self, train, user2idx, item2idx, reference_timestamp,
                  gamma=1.0, tau_days=180.0):
        self.gamma = gamma
        self.tau_days = tau_days
        targets = custom_targets(train, reference_timestamp, gamma, tau_days)
        R = sparse.csr_matrix((targets, (train.user_id.map(user2idx), train.item_id.map(item2idx))),
                              shape=(len(user2idx), len(item2idx)))
        return self.fit(R)
