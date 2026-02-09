from typing import List
import numpy as np

class MarkovianPolicy:
    """
    It models a Markovian policy pi: Sx[H]->Delta(A).
    """

    pi: List[np.ndarray]

    def __init__(self, pi):
        self.pi = pi
    
    def get_action_batch(self, h: int, s: int, omega: np.ndarray):
        """
        Vectorized action selection for a batch of (s, omega).
        
        Args:
            h: time step
            s: shape (N,) — states
            omega: shape (N,2*h) — 
        
        Returns:
            actions: shape (N,) — sampled actions
        """
        N = s.shape[0]

        probs = self.pi[h][s,:]

        # Sample actions
        actions = (probs.cumsum(axis=1) > np.random.rand(N, 1)).argmax(axis=1)  # shape: (N,)

        return actions
    
    
class OurNonMarkovianPolicy:
    """
    It models a non-Markovian policy of the kind proposed in our paper:
    pi: SxYx[H]->Delta(A).
    """

    pi: np.ndarray
    theta: float
    r: List[np.ndarray]

    def __init__(self, pi, theta, r):
        self.pi = pi
        self.theta = theta
        self.r = r
    
    def get_action_batch(self, h: int, s: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """
        Vectorized action selection for a batch of (s, omega).
        
        Args:
            h: time step
            s: shape (N,) — states
            omega: shape (N,2*h) — 
        
        Returns:
            actions: shape (N,) — sampled actions
        """
        s = np.asarray(s)
        N = s.shape[0]

        # Compute return indices
        indices = np.zeros(N).astype(int)
        for hh in range(omega.shape[1]//2):
            ss = omega[:, 2*hh].astype(int)
            aa = omega[:, 2*hh + 1].astype(int)
            indices += np.rint(self.r[hh][ss, aa] / self.theta).astype(int)

        # Lookup action distributions
        probs_batch = self.pi[h][s,indices,:]  # shape: (N, A)

        # Sample actions from categorical distributions
        # Vectorized sampling using CDF + uniform trick
        rand = np.random.rand(N, 1)
        cdf = probs_batch.cumsum(axis=1)
        actions = (cdf > rand).argmax(axis=1)  # shape: (N,)

        return actions
    

class NonMarkovianPolicy:
    """
    It models a general non-Markovian policy using a feature_dim layer for
    computing features from the past history to decide which action to play.
    """

    def __init__(self, S, A, H, deterministic=False, feature_dim=16):
        """
        Args:
            S (int): number of states
            A (int): number of actions
            H (int): horizon (max stages)
            deterministic (bool): flag for deterministic vs stochastic policy
            feature_dim (int): dimension of compressed trajectory features
            seed (int): random seed
        """
        self.S = S
        self.A = A
        self.H = H
        self.deterministic = deterministic
        self.feature_dim = feature_dim

        # Random projection matrix for compressing trajectories into features
        self.proj_matrix = np.random.normal(size=(2*H, feature_dim))

        # Policy parameters: (S, feature_dim, A)
        self.weights = np.random.normal(size=(S, feature_dim, A))

    def _featurize(self, omega):
        """
        Compress past trajectory omega into features using random projection.
        Args:
            omega (np.ndarray): shape (N, 2(h-1)), past trajectory (state, action)
        Returns:
            features (np.ndarray): shape (N, feature_dim)
        """
        if omega.shape[1] == 0:  # no history
            return np.zeros((omega.shape[0], self.feature_dim))
        # Pad to full length 2H for projection
        padded = np.zeros((omega.shape[0], 2*self.H))
        padded[:, :omega.shape[1]] = omega
        return padded @ self.proj_matrix  # (N, feature_dim)

    def get_action_batch(self, h, s, omega):
        """
        Get a batch of actions from the policy.

        Args:
            h (int): current stage
            s (np.ndarray): shape (N,), current states
            omega (np.ndarray): shape (N, 2(h-1)), past trajectory

        Returns:
            actions (np.ndarray): shape (N,), actions in {0,...,A-1}
        """
        N = s.shape[0]
        features = self._featurize(omega)  # (N, feature_dim)

        logits = np.einsum('nd,nda->na', features, self.weights[s])

        if self.deterministic:
            return np.argmax(logits, axis=1)
        else:
            # softmax sampling
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            r = np.random.random(N)
            cumprobs = np.cumsum(probs, axis=1)
            return (r[:, None] < cumprobs).argmax(axis=1)
