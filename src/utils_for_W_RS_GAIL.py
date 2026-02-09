import numpy as np

from src.policies import MarkovianPolicy

def compute_grad(D,g,var,M,S,A,H,alpha,lam):
    """
    Computes and returns the gradient w.r.t. the cost function parameters.
    See Theorem 4 and Corollary 6.
    """
    # compute coefficients of sum
    ind = g >= var
    coeff = alpha + lam*ind

    # compute gradients
    states  = D[:,:-1][:, 0::2]     # shape (M, H)
    actions = D[:,:-1][:, 1::2]     # shape (M, H)
    grad = np.zeros((M, S, A), dtype=float)

    # Vectorized accumulation: add 1 at (i, state, action)
    np.add.at(grad, (np.repeat(np.arange(M), H),
                    states.reshape(-1),
                    actions.reshape(-1)),  1)

    # compute sum
    grad_pi = np.tensordot(coeff, grad, axes=(0,0))

    # division
    grad_pi = grad_pi / (alpha*M)

    return grad_pi

def get_ret_var(D,w,H,alpha,M,mdp):
    """
    Compute the return of each trajectory in D under cost/reward w, as well as
    the VaR at level alpha. Return both.
    """
    g = mdp.compute_returns(D, [w for _ in range(H)])
    gsorted = np.sort(g)

    k = int(np.ceil((1 - alpha) * M)) - 1  # subtract 1 for 0-based index
    k = max(k, 0)

    var = gsorted[k]

    return g, var

def get_nablalogpi(S,A,softmax):
    """
    return a np array with shape (S,A,S,A), where
    nablalogpi[s,a,s',a']=dlogpi(s,a)/ds',a', i.e., nablalogpi[s,a,s',a']
    represents the derivative of logpi[s,a] w.r.t. theta[s',a'].
    """
    # initialize nablalogpi
    nablalogpi = np.zeros((S,A,S,A))

    for s in range(S):
        for a in range(A):
            nablalogpi[s,a,s,a] = 1
            nablalogpi[s,a,s,:] -= softmax[s,:]

    return nablalogpi

def get_grad_rho(D,w,H,alpha,M,mdp,S,A,nablalogpi):
    """
    Compute the gradient of the rho term (Theorem 5 of "Risk-Sensitive
    Generative Adversarial Imitation Learning"). Estimate it with the empirical
    mean, as done in REINFORCE.
    """
    # compute coefficients
    g, var = get_ret_var(D, w, H, alpha, M, mdp)
    coeff = np.maximum(0.0, g - var) / alpha     # we include division by α here

    states  = D[:,:-1][:, 0::2]     # shape (M, H)
    actions = D[:,:-1][:, 1::2]     # shape (M, H)

    # nablalogpi_flat[s,a] = gradient vector flattened (SA,)
    nlp_flat = nablalogpi.reshape(S, A, -1)     # (S,A,SA)

    # Gather all ∇logπ(a|s) at every visited (s,a)
    # grad_per_timestep[i,h] = gradient vector (SA,)
    grad_per_timestep = nlp_flat[states, actions]   # shape (M,H,SA)

    # grad_logpi_per_traj[i] = ∑_h ∇logπ(a_h|s_h)
    grad_logpi_per_traj = grad_per_timestep.sum(axis=1)   # (M,SA)
    
    weighted = coeff[:, None] * grad_logpi_per_traj  # (M,SA)

    grad = weighted.sum(axis=0) / M       # (SA,)

    return grad.reshape(S, A)


def get_pi(theta,H):
    """
    Extract a policy from the given parameters theta.
    """
    # numerically stable softmax
    theta_shifted = theta - np.max(theta, axis=1, keepdims=True)
    softmax = np.exp(theta_shifted) / np.sum(np.exp(theta_shifted), axis=1, keepdims=True)
    pi = MarkovianPolicy([softmax for _ in range(H)])

    return pi
