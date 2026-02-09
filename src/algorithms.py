from src.policies import *
import cvxpy as cp
from scipy.sparse import coo_matrix
from src.utils_for_W_RS_GAIL import *

def RS_BC(DE,mdp,theta):
    """
    Implementation of algorithm RS-BC.
    """
    S,A,H,r = mdp.S,mdp.A,mdp.H,mdp.r

    # Extract states and actions for each stage
    states = DE[:, np.arange(0, 2*H, 2)].astype(int)   # shape (Ntraj, H)
    actions = DE[:, np.arange(1, 2*H, 2)].astype(int)  # shape (Ntraj, H)

    # Construct analogous return_indices array
    return_indices = np.zeros((DE.shape[0], H)).astype(int)

    for h in range(H-1):  # start from the second stage
        return_indices[:,h+1] += np.rint(r[h][states[:,h],actions[:,h]] / theta).astype(int)

    return_indices = np.cumsum(return_indices, axis=1)

    # Visit counts: (H, S, Y, A)
    Y = 1 + int(H/theta)
    N = np.zeros((H, S, Y, A), dtype=float)
    for h in range(H):
        np.add.at(N[h], (states[:,h], return_indices[:,h], actions[:,h]), 1)

    # Compute policy: pi[h][s,g,a] = normalized counts
    N_s = np.sum(N, axis=3, keepdims=True)
    pi = np.divide(N, N_s, out=np.full_like(N, 1.0 / A), where=(N_s != 0))

    return OurNonMarkovianPolicy(pi,theta,r)


def BC(DE,mdp):
    """
    Behavior cloning policy estimator from trajectories.

    Args:
        DE (np.ndarray): shape (N, 2H+1), trajectories
        S (int): number of states
        A (int): number of actions
        H (int): horizon

    Returns:
        pi (np.ndarray): pi[h] is (S,A) policy at stage h
    """
    S,A,H = mdp.S,mdp.A,mdp.H

    # Extract states and actions for each stage
    states = DE[:, np.arange(0, 2*H, 2)].astype(int)   # shape (Ntraj, H)
    actions = DE[:, np.arange(1, 2*H, 2)].astype(int)  # shape (Ntraj, H)

    # Visit counts: (H, S, A)
    N = np.zeros((H, S, A), dtype=float)
    for h in range(H):
        np.add.at(N[h], (states[:,h], actions[:,h]), 1)

    # Compute policy: pi[h][s,a] = normalized counts    
    N_s = np.sum(N, axis=2, keepdims=True)
    pi = np.divide(N, N_s, out=np.full_like(N, 1.0 / A), where=(N_s != 0))

    return MarkovianPolicy(pi)


def MIMIC_MD(DE, mdp, verbose=False):
    """
    Implementation of algorithm MIMIC-MD from Rajaraman et al. "Toward the
    fundamental limits of imitation learning". We use LP formulation from
    Rajaraman et al. "Provably breaking the quadratic error compounding barrier
    in imitation learning, optimally".
    """
    S, A, H, p, s0 = mdp.S, mdp.A, mdp.H, np.array(mdp.p), mdp.s0

    # shuffle DE
    np.random.shuffle(DE)

    # split DE
    size1 = DE.shape[0] // 2
    size2 = DE.shape[0] - size1
    D1, D2 = DE[:size1], DE[size1:]

    # -------- Estimate counts from D1 (like in BC) --------
    states1 = D1[:, np.arange(0, 2*H, 2)].astype(int)
    actions1 = D1[:, np.arange(1, 2*H, 2)].astype(int)
    N_sa = np.zeros((H, S, A), dtype=float)
    for h in range(H):
        np.add.at(N_sa[h], (states1[:, h], actions1[:, h]), 1)

    # -------- Estimate occupancy from D2 --------
    states2 = D2[:, np.arange(0, 2*H, 2)].astype(int)
    actions2 = D2[:, np.arange(1, 2*H, 2)].astype(int)
    M_sa = np.zeros((H, S, A), dtype=float)
    for h in range(H):
        np.add.at(M_sa[h], (states2[:, h], actions2[:, h]), 1)
    M_sa /= size2

    # -------- Optimization problem (Eq. 16) --------
    d = cp.Variable((H, S, A), nonneg=True)

    # Flow constraints (vectorized)
    inflow = cp.sum(cp.multiply(p[:-1, :, :, :], d[:-1, :, :, None]), axis=(1, 2))
    outflow = cp.sum(d[1:, :, :], axis=2)
    flow_constraints = [outflow == inflow]

    # Initial state distribution constraint
    mu0 = np.zeros(S)
    mu0[s0] = 1.0
    init_constraint = [cp.sum(d[0,s,:]) == mu0[s] for s in range(S)]

    # Objective
    objective = cp.Minimize(cp.norm1(d - M_sa))

    # -------- Imitation constraints --------
    imitation_constraints = []
    # Normalized visitation frequencies from D1
    N_s = N_sa.sum(axis=2)  # (H,S)
    # imitation constraints
    for h in range(H):
        for s in range(S):
            if N_s[h,s] > 0:
                pi_emp = N_sa[h, s, :] / N_s[h,s]  # empirical policy
                imitation_constraints.append(d[h, s, :] == pi_emp * cp.sum(d[h, s, :]))

    # Build problem
    constraints = flow_constraints + init_constraint + imitation_constraints
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND)

    # -------- Extract policy --------
    d_hsa = d.value
    d_hsa = np.maximum(d_hsa, 0)  # avoid negatives
    d_hs = np.sum(d_hsa, axis=2, keepdims=True)
    pi = np.divide(d_hsa, d_hs, out=np.full_like(d_hsa, 1.0 / A), where=(d_hs != 0))

    return MarkovianPolicy(pi)


def RS_KT(DE,mdp,theta, verbose=False):
    """
    Implementation of RS-KT algorithm. For practical efficiency, we implement a
    different set Yh for each step h.    
    """
    S, A, H, p, s0, r = mdp.S, mdp.A, mdp.H, np.array(mdp.p), mdp.s0, np.array(mdp.r)

    # before all, discretize r and transform it to indices for simplicity later
    r = np.rint(r / theta).astype(int)

    # compute eta_hat (use est_ret_v2 as in the paper)
    eta_hat = mdp.est_ret_v2(DE,theta)

    # define Y
    Y = [1+int(h/theta) for h in range(H+1)]

    # -------- Optimization problem --------
    d = cp.Variable((S*np.sum(Y[:-1])*A), nonneg=True)  # occupancy measure
    eta = cp.Variable((Y[-1]))  # return distribution
    t = cp.Variable((Y[-1]))  # difference between cumulative return distributions

    ### C1 - Wasserstein definition
    t_def_constraint = [t == cp.cumsum(eta-eta_hat)]

    ### C2 - Flow constraints (vectorized)
    flow_constraints = []

    for h in range(1, H):
        if verbose and h%2==0:
            print('constraints h', h)

        # all (s,g) pairs as a single index
        sg_idx = np.arange(S*Y[h])      # 0..S*Y[h]-1
        s_idx = sg_idx // Y[h]
        g_idx = sg_idx % Y[h]

        # -------- positive part: d[h,s,g,:] with coeff +1
        col0 = S*Y[h-1]*A + (s_idx*Y[h] + g_idx)*A
        cols_pos = (col0[:, None] + np.arange(A)).ravel()
        rows_pos = np.repeat(sg_idx, A)
        data_pos = np.ones_like(rows_pos, dtype=float)

        # -------- negative part: transitions from previous layer
        # shape (S, A, S*Y[h]) after broadcasting
        # Z contains the values of previous returns that give s,g from ss,aa
        Z = g_idx[None, None, :] - r[h-1, :, :, None]   # (S, A, S*Y[h])
        mask = (Z >= 0) & (Z < Y[h-1])  

        # pick valid entries
        ss_idx, aa_idx, sg_lin = np.nonzero(mask)
        z_valid = Z[ss_idx, aa_idx, sg_lin]

        # row index is the sg index
        rows_neg = sg_lin

        # col index = d[h-1, ss, z_valid, aa]
        cols_neg = ss_idx*Y[h-1]*A + z_valid*A + aa_idx

        # coeff = -p[h-1, ss, aa, s_target]
        s_target = s_idx[sg_lin]
        data_neg = -p[h-1, ss_idx, aa_idx, s_target]

        # -------- assemble sparse matrix
        rows = np.concatenate([rows_pos, rows_neg])
        cols = np.concatenate([cols_pos, cols_neg])
        data = np.concatenate([data_pos, data_neg])

        Fh = coo_matrix((data, (rows, cols)), 
                    shape=(S*Y[h], S*(Y[h-1]+Y[h])*A)).tocsr()
        
        low = np.sum(Y[:h-1])*S*A
        if h == 1:
            low = 0
        flow_constraints.append(Fh @ d[low:np.sum(Y[:h+1])*S*A] == 0)

    ### C3 - Relation d and eta (= imitation constraint)
    # lists for sparse matrix
    rows, cols, data = [], [], []

    # add 1 to each eta[g] value
    rows.extend(np.arange(Y[-1]))
    cols.extend(np.arange(Y[-1]))
    data.extend(np.ones(Y[-1]))

    # compute all possible s,a,g triples (where g is next g)
    s_idx = np.repeat(np.arange(S), A*Y[-1])
    a_idx = np.tile(np.repeat(np.arange(A), Y[-1]), S)
    g_idx = np.tile(np.arange(Y[-1]), S*A)

    # compute corresponding previous g
    gp = g_idx - r[-1,s_idx, a_idx]   # vectorized gp = g - r[H-1,s,a]

    # mask only feasible gp
    mask = (gp >= 0) & (gp <= Y[-2]-1)
    if np.any(mask):
        rows.extend(g_idx[mask])
        cols.extend(Y[-1] + s_idx[mask]*Y[-2]*A + gp[mask]*A + a_idx[mask])
        data.extend([-1]*np.sum(mask))

    # construct Y[-1] constraints (one per value of eta)
    # use Y[-1] variables (for eta) and S*Y[-2]*A for d[H-1]
    F = coo_matrix((data, (rows, cols)), shape=(Y[-1], Y[-1]+S*Y[-2]*A)).tocsr()
    k = cp.hstack([eta, d[S*A*np.sum(Y[:-2]):]])  # d[H-1]
    relation_constraints = [F@k == 0]

    ## C4 - Initial state distribution constraint
    G = np.zeros((2,S*A))
    GG = np.array([0,1])
    # line 1: all but s0,g0 are 0
    G[0,:] = 1
    G[0, s0*A:s0*A+A] = 0
    # line 2: s0,g0 sums to 1
    G[1, s0*A:s0*A+A] = 1
    init_constraint = [G @ d[:S*A] == GG] 

    ### Objective
    objective = cp.Minimize(cp.norm1(t))

    # Build problem
    constraints = flow_constraints + init_constraint + relation_constraints + t_def_constraint
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=verbose, canon_backend=cp.SCIPY_CANON_BACKEND)

    # -------- Extract policy --------
    d_hsga = d.value
    d_hsga = np.maximum(d_hsga, 0)  # avoid negatives
    pi = []
    for h in range(H):
        low = np.sum(Y[:h])*S*A
        if h == 0:
            low = 0
        x = d_hsga[low:np.sum(Y[:h+1])*S*A]
        x = np.reshape(x, (S,Y[h],A))
        # normalize
        x_s = np.sum(x, axis=2, keepdims=True)
        pih = np.divide(x, x_s, out=np.full_like(x, 1.0 / A), where=(x_s != 0))
        pi.append(pih)

    return OurNonMarkovianPolicy(pi,theta,mdp.r)


def W_RS_GAIL(DE,mdp,alpha,lam,M,lr,n_iterations,theta0,w0, verbose=False):
    """
    Implementation of the W-RS-GAIL algorithm from paper "Risk-Sensitive
    Generative Adversarial Imitation Learning". We implement a variant.
    Specifically, we do not use either ADAM or a KL-constrained update, but we
    simply update the cost with gradient ascent, and the policy with gradient
    descent, both with the same fixed learning rate lr. Moreover, we do not use
    function approximation for the cost w, and we use SA parameters for policy,
    where the probability of an action is the softmax of the S parameters for
    that state-action pair.

    DE: expert dataset, shape (N,2*H+1)
    mdp: the target mdp
    alpha: the level of the CVaR
    lam: hyperparameter representing the Lagrange multiplier
    M: number of trajectories to generate at each iteration
    lr: fixed learning rate
    n_iterations: total number of iterations
    theta0: initial policy parameters
    w0: initial cost function parameters
    """
    S,A,H = mdp.S,mdp.A,mdp.H
    N = DE.shape[0]  # n expert trajectories

    # set initial parameter values
    theta = np.copy(theta0)
    w = np.copy(w0)

    for j in range(n_iterations):
        if verbose and j % 500 == 0:
            print(j)

        ##### generate M trajectories from pi
        # write policy
        theta_shifted = theta - np.max(theta, axis=1, keepdims=True)
        softmax = np.exp(theta_shifted) / np.sum(np.exp(theta_shifted), axis=1, keepdims=True)
        pi = MarkovianPolicy([softmax for _ in range(H)])
        # play
        D = mdp.play(pi,M)

        ##### estimate VaRs
        # learned policy
        g, var = get_ret_var(D,w,H,alpha,M,mdp)
        # expert's policy
        gE, varE = get_ret_var(DE,w,H,alpha,N,mdp)

        ##### update reward
        # estimate gradient
        grad = compute_grad(D,g,var,M,S,A,H,alpha,lam)
        gradE = compute_grad(DE,gE,varE,N,S,A,H,alpha,lam)
        # update
        grad_w = grad-gradE
        w = w + lr*grad_w

        ##### update policy
        # precompute nablalogpi
        nablalogpi = get_nablalogpi(S,A,softmax)
        # compute gradient of rho
        grad_rho = get_grad_rho(D,w,H,alpha,M,mdp,S,A,nablalogpi)
        # update
        grad_theta = grad_rho
        theta = theta - lr*grad_theta
    
    # get policy with numerically stable softmax
    pi = get_pi(theta,H)

    return pi

