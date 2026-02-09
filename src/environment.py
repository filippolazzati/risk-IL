from typing import List
import numpy as np

class MDP:
    S: int
    A: int
    H: int
    p: List[np.ndarray]  # h,s,a,s'
    s0: int
    r: List[np.ndarray]

    def __init__(self,S,A,H,rho=5e-2,sparser=True,prob=0.7):
        """
        Create an MDP instance with the specified size. The reward values are
        sampled uniformly from a grid with distances rho. The transition model
        is sampled uniformly at random. If sparser is True, then the transition
        model is made more deterministic based on prob.
        """
        self.S = S
        self.A = A
        self.H = H
        self.s0 = np.random.randint(S)
        self.r = [np.random.choice(np.arange(0,1,rho), size=(S,A)) for _ in range(H)]
        self.p = [np.random.dirichlet(alpha=(1,)*S, size=(S,A)) for _ in range(H)]

        if sparser:
                # make a portion of p deterministic
                for h in range(H):

                        k = np.random.rand(S,A)
                        det_s_prime = np.random.choice(S, size=(S,A))

                        for s in range(S):
                                for a in range(A):
                                        if k[s,a] < prob:
                                                self.p[h][s,a,:] = 0
                                                self.p[h][s,a,det_s_prime[s,a]] = 1

        # precompute the cdf of p for efficiency
        self.cdf = [np.cumsum(self.p[h], axis=2) for h in range(self.H)]

    def play(self, pi, N: int):
        """
        Vectorized trajectory collection over N environments.

        Args:
            pi: policy with get_action(h, s_batch, omega_batch) -> a_batch
            N: number of parallel rollouts

        Returns:
            DE: List of N trajectories, each a list [s0, a0, s1, a1, ..., sH]
        """
        # Preallocate trajectory buffer: each trajectory is length 2*H + 1
        DE = np.zeros((N, 2 * self.H + 1), dtype=np.int8)

        # Initialize state
        s = np.full(N, self.s0, dtype=int)   # (N,)

        DE[:,0] = s

        # precompute samples for next state for efficiency
        samples = np.random.rand(N, self.H)

        for h in range(self.H):
            # Get actions from batched policy
            a = pi.get_action_batch(h, s, DE[:,:2*h])  # shape: (N,)

            # Sample next states
            s_prime = (self.cdf[h][s,a,:] < samples[:,h,None]).sum(axis=1)

            # Update s
            s = s_prime

            # Append to each trajectory
            DE[:, 2*h+1] = a
            DE[:, 2*h+2] = s

        return DE

    def est_ret(self, DE: List[np.ndarray], eps: float = 1e-4):
        """
        Estimate return distribution from dataset of trajectories. Do not
        discretize the per-step reward, but just the final return values.

        Args:
            DE (np.ndarray): shape (N, 2H+1), trajectories (s0,a0,...,sH)
            eps (float): discretization step size

        Returns:
            eta (np.ndarray): shape (1+H//eps,), normalized histogram of returns
        """
        N = DE.shape[0]

        # Compute per-trajectory return vectorized
        g = self.compute_returns(DE)

        # Bin the returns into multiples of eps
        indices = np.rint(g / eps).astype(int)

        # Count occurrences
        eta = np.bincount(indices, minlength=int(self.H / eps)+1).astype(float)

        # Normalize
        eta /= N

        return eta
    
    def compute_returns(self, D: np.ndarray, r: List[np.ndarray] = None):
        """
        Given N trajectories, return an array long N with the return of each
        trajectory.
        """
        N = D.shape[0]
        rew = self.r if r is None else r

        # Compute per-trajectory return vectorized
        g = np.zeros(N)
        for h in range(self.H):
            s = D[:, 2*h]
            a = D[:, 2*h + 1]
            g += rew[h][s, a]
        
        return g


    def est_ret_v2(self, DE: List[np.ndarray], eps: float = 1e-4):
        """
        Estimate return distribution from dataset of trajectories. Discretize
        each per-step reward, and then compute the return distribution.

        Args:
            DE (np.ndarray): shape (N, 2H+1), trajectories (s0,a0,...,sH)
            eps (float): discretization step size

        Returns:
            eta (np.ndarray): shape (1+H/eps,), normalized histogram of returns
        """
        N = DE.shape[0]

        # Compute per-trajectory return vectorized
        g = np.zeros(N).astype(int)
        for h in range(self.H):
            s = DE[:, 2*h]
            a = DE[:, 2*h + 1]
            g += np.rint(self.r[h][s, a] / eps).astype(int)

        # Bin the returns into multiples of eps
        indices = g

        # Count occurrences
        eta = np.bincount(indices, minlength=int(self.H / eps)+1).astype(float)

        # Normalize
        eta /= N

        return eta