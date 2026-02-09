import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from src.environment import *
from src.policies import *
from src.algorithms import *
import os
import time

def get_piE(S,A,H,markov,det=False):
        """
        Utility for generating an expert's policy at random. markov says if the
        policy is Markovian or not, while det whether it is deterministic or
        stochastic.
        """
        if markov and det:
                # sample the actions
                samples = np.random.choice(A, size=(H,S))
                
                pi = []
                
                for h in range(H):
                        x = np.zeros((S,A))
                        for s in range(S):
                                x[s,samples[h,s]] = 1

                        pi.append(x)
                return MarkovianPolicy(pi)

        elif markov and not det:
                return MarkovianPolicy(
                        [np.random.dirichlet(alpha=(1,)*A, size=S) for _ in range(H)]
                        )
        else:
                return NonMarkovianPolicy(S,A,H,det)


def plot_return_distributions(
        distribs: list,
        names: list,
        eps: float,
        x: int,
        y: int,
        cumsum: bool = True,
        save_fig: bool = False,
        title: str=''
):
    """
    Utility for plotting multiple return distributions.
    """
    # x-axis values: 0, eps, 2*eps, ..., (N-1)*eps
    x = np.arange(x,y) * eps

    fig = plt.figure(figsize=(8, 4))
    if title != '':
        plt.title(title)

    cmap = cm.get_cmap('tab10', len(distribs)+1)  # len(distribs) distinct colors

    for i, (eta, name) in enumerate(zip(distribs,names)):
        color = cmap(i+1)
        if cumsum:
            plt.plot(x, np.cumsum(eta), color=color, label=name)
        else:
            stem = plt.stem(x, eta, label=name)
            stem.markerline.set_color(color)
            stem.stemlines.set_color(color)
            stem.baseline.set_color("black")

    plt.xlabel("Return values")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        fig.savefig('image.pdf', format="pdf", dpi=1200)

    plt.show()


def W1(eta1,eta2,eps):
    """
    Utility for computing the 1-Wasserstein distance between eta1 and eta2.
    """
    F1 = np.cumsum(eta1)
    F2 = np.cumsum(eta2)

    one_norm = np.sum(np.abs(F1-F2))

    return eps*one_norm


def run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose):
        """
        Utility for running the experiments.
        """
        # if folder does ot exists, create it
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # data structure for results (4=algorithms)
        results = np.zeros((4,n_mdps,len(values_N)))

        # loop over MDPs
        for n_mdp in range(n_mdps):
            print('*'*5,'MDP ',n_mdp)

            start = time.time()

            # set seed for reproducibility
            np.random.seed(n_mdp)

            # generate MDP and piE
            mdp = MDP(S,A,H,rho)
            piE = get_piE(S,A,H,markov,det)

            # estimate etaE
            etaE = mdp.est_ret(mdp.play(piE,M),eps)

            for i, N in enumerate(values_N):
                print('N=',N)

                # average results over multiple DE generations
                values_RSBC = []
                values_RSKT = []
                values_BC = []
                values_MIMIC = []

                for _ in range(n_DEs):

                    # collect DE
                    DE = mdp.play(piE,N)

                    # run algorithms
                    piRSBC = RS_BC(DE,mdp,theta)
                    piRSKT = RS_KT(DE,mdp,theta)
                    piBC = BC(DE,mdp)
                    piMIMIC = MIMIC_MD(DE,mdp)

                    # estimate return distributions
                    etaRSBC = mdp.est_ret(mdp.play(piRSBC,M),eps)
                    etaRSKT = mdp.est_ret(mdp.play(piRSKT,M),eps)
                    etaBC = mdp.est_ret(mdp.play(piBC,M),eps)
                    etaMIMIC = mdp.est_ret(mdp.play(piMIMIC,M),eps)

                    # compute distances and append
                    values_RSBC.append(W1(etaE,etaRSBC,eps))
                    values_RSKT.append(W1(etaE,etaRSKT,eps))
                    values_BC.append(W1(etaE,etaBC,eps))
                    values_MIMIC.append(W1(etaE,etaMIMIC,eps))

                # compute averages and update results
                results[0,n_mdp,i] = np.mean(values_RSBC)
                results[1,n_mdp,i] = np.mean(values_RSKT)
                results[2,n_mdp,i] = np.mean(values_BC)
                results[3,n_mdp,i] = np.mean(values_MIMIC)

            # dump the results every 10 mdps
            if folder != '' and (n_mdp % 10 == 0 or n_mdp == n_mdps-1):
                np.save(folder,results)

            # print results
            if verbose and (n_mdp % 3 == 0 or n_mdp == n_mdps-1):
                show_results(results[:,:n_mdp+1,:],values_N)
                
            # print time elapsed
            t = time.time() - start
            print('\n time: ',int(t // 60),'m ',int(t%60),'s')


def run2(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose):
        """
        Utility for running the experiments for Q4.
        """
        # if folder does ot exists, create it
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # data structure for results (3=algorithms)
        results = np.zeros((3,n_mdps,len(values_N)))

        # loop over MDPs
        for n_mdp in range(n_mdps):
            print('*'*5,'MDP ',n_mdp)

            start = time.time()

            # set seed for reproducibility
            np.random.seed(n_mdp)

            # generate MDP and piE
            mdp = MDP(S,A,H,rho)
            piE = get_piE(S,A,H,markov,det)

            # estimate etaE
            etaE = mdp.est_ret(mdp.play(piE,M),eps)

            for i, N in enumerate(values_N):
                print('N=',N)

                # average results over multiple DE generations
                values_RSBC = []
                values_BC = []
                values_hat = []

                for _ in range(n_DEs):

                    # collect DE
                    DE = mdp.play(piE,N)

                    # run algorithms
                    piRSBC = RS_BC(DE,mdp,theta)
                    piBC = BC(DE,mdp)

                    # estimate return distributions
                    etaRSBC = mdp.est_ret(mdp.play(piRSBC,M),eps)
                    etaBC = mdp.est_ret(mdp.play(piBC,M),eps)
                    eta_hat = mdp.est_ret_v2(DE,eps)

                    # compute distances and append
                    values_RSBC.append(W1(etaE,etaRSBC,eps))
                    values_BC.append(W1(etaE,etaBC,eps))
                    values_hat.append(W1(etaE,eta_hat,eps))

                # compute averages and update results
                results[0,n_mdp,i] = np.mean(values_RSBC)
                results[1,n_mdp,i] = np.mean(values_hat)
                results[2,n_mdp,i] = np.mean(values_BC)

            # dump the results every 10 mdps
            if folder != '' and (n_mdp % 10 == 0 or n_mdp == n_mdps-1):
                np.save(folder,results)

            # print results
            if verbose and (n_mdp % 3 == 0 or n_mdp == n_mdps-1):
                show_results(results[:,:n_mdp+1,:],values_N)
                
            # print time elapsed
            t = time.time() - start
            print('\n time: ',int(t // 60),'m ',int(t%60),'s')
            

def show_results(results,values_N,precision=3):
    """
    Utility for showing the results of the experiments.
    """

    for i,N in enumerate(values_N):
        print('N = ',N)
        res_RSBC = 'RS-BC: '
        res_RSBC += str(round(np.mean(results[0,:,i]),precision))
        res_RSBC += u'\u00B1'
        res_RSBC += str(round(np.std(results[0,:,i]),precision))

        res_BC = 'BC: '
        res_BC += str(round(np.mean(results[2,:,i]),precision))
        res_BC += u'\u00B1'
        res_BC += str(round(np.std(results[2,:,i]),precision)) 

        if results.shape[0] == 4:
            res_RSKT = 'RS-KT: '
            res_RSKT += str(round(np.mean(results[1,:,i]),precision))
            res_RSKT += u'\u00B1'
            res_RSKT += str(round(np.std(results[1,:,i]),precision))

            res_MIMIC = 'MIMIC-MD: '
            res_MIMIC += str(round(np.mean(results[3,:,i]),precision))
            res_MIMIC += u'\u00B1'
            res_MIMIC += str(round(np.std(results[3,:,i]),precision))
            if i != len(values_N)-1:
                res_MIMIC += '\n'

            print(res_RSBC)
            print(res_RSKT)
            print(res_BC)
            print(res_MIMIC)
        else:
            res_hat = 'eta_hat: '
            res_hat += str(round(np.mean(results[1,:,i]),precision))
            res_hat += u'\u00B1'
            res_hat += str(round(np.std(results[1,:,i]),precision))
            if i != len(values_N)-1:
                res_hat += '\n'
             
            print(res_RSBC)
            print(res_BC)
            print(res_hat)


def run3(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose):
        """
        Utility for running the experiments comparing RSBC with W-RS-GAIL.
        """
        # hyperparameters WRSGAIL
        alphas = [0.3,0.7]
        lam = 2
        theta0 = np.zeros((S,A))  # the policy parameters
        w0 = np.zeros((S,A))  # the cost/discriminator parameters
        n_trajs = 500
        lr = 5e-4
        n_iterations = 3000

        # if folder does ot exists, create it
        if folder != '' and not os.path.exists(folder):
            os.makedirs(folder)

        # data structure for results (3=algorithms)
        results = np.zeros((3,n_mdps,len(values_N)))

        # loop over MDPs
        for n_mdp in range(n_mdps):
            print('*'*5,'MDP ',n_mdp)

            start = time.time()

            # set seed for reproducibility
            np.random.seed(n_mdp)

            # generate MDP and piE
            mdp = MDP(S,A,H,rho)
            piE = get_piE(S,A,H,markov,det)

            # estimate etaE
            etaE = mdp.est_ret(mdp.play(piE,M),eps)

            for i, N in enumerate(values_N):
                print('N=',N)

                # average results over multiple DE generations
                values_RSBC = np.empty((n_DEs))
                values_WRSGAIL1 = np.empty((n_DEs))
                values_WRSGAIL2 = np.empty((n_DEs))

                for j in range(n_DEs):
                    print('j=',j)

                    # collect DE
                    DE = mdp.play(piE,N)

                    # run algorithms
                    piRSBC = RS_BC(DE,mdp,theta)
                    piWRSGAIL1 = W_RS_GAIL(DE,mdp,alphas[0],lam,n_trajs,lr,n_iterations,theta0,w0)
                    piWRSGAIL2 = W_RS_GAIL(DE,mdp,alphas[1],lam,n_trajs,lr,n_iterations,theta0,w0)

                    # estimate return distributions
                    etaRSBC = mdp.est_ret(mdp.play(piRSBC,M),eps)
                    etaWRSGAIL1 = mdp.est_ret(mdp.play(piWRSGAIL1,M),eps)
                    etaWRSGAIL2 = mdp.est_ret(mdp.play(piWRSGAIL2,M),eps)

                    # compute distances and append
                    values_RSBC[j] = W1(etaE,etaRSBC,eps)
                    values_WRSGAIL1[j] = W1(etaE,etaWRSGAIL1,eps)
                    values_WRSGAIL2[j] = W1(etaE,etaWRSGAIL2,eps)

                # compute averages and update results
                results[0,n_mdp,i] = np.mean(values_RSBC)
                results[1,n_mdp,i] = np.mean(values_WRSGAIL1)
                results[2,n_mdp,i] = np.mean(values_WRSGAIL2)

            # dump the results every 10 mdps
            if folder != '':
                np.save(folder,results)

            # print results
            if verbose and (n_mdp % 3 == 0 or n_mdp == n_mdps-1):
                show_results3(results[:,:n_mdp+1,:],values_N)
                
            # print time elapsed
            t = time.time() - start
            print('\n time: ',int(t // 60),'m ',int(t%60),'s')


def show_results3(results,values_N,precision=3):
    """
    Utility for showing the results of the experiments.
    """
    m = np.mean(results,axis=1)
    std = np.std(results,axis=1)

    for i,N in enumerate(values_N):
        print('#'*10,'N = ',N)

        print('*'*3,' RS-BC:')
        print('W1: ',round(m[0,i],precision),u'\u00B1',round(std[0,i],precision))

        for j,alpha in enumerate([0.3,0.7]):
            print('*'*3,f' W-RS-GAIL, alpha={alpha}')
            print('W1: ',round(m[1+j,i],precision),u'\u00B1',round(std[1+j,i],precision))