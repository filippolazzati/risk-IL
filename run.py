from src.algorithms import *
from src.utils import *

# all runs share the following parameter values
det = False  # if piE is deterministic
n_mdps = 50  # number different mdps
n_DEs = 3  # number of different DE generation
values_N = [20,80,300,1000,10000]  # number of expert's trajectories
eps = 1e-4  # precision for evaluation
M = 1000000  # number of trajectories for evaluation
verbose = True  # print results during run


###################################### Question 1

##### Exp0

folder = 'results/exp0/'
S,A,H = 2,2,5
theta = 5e-2
rho = 3e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


##### Exp3 - larger S,A

folder = 'results/exp3/'
S,A,H = 50,5,5
theta = 5e-2
rho = 3e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


##### Exp6 - larger H

folder = 'results/exp6/'
S,A,H = 2,2,20
theta = 5e-2
rho = 3e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


###################################### Question 2

##### Exp1 - theta=rho

folder = 'results/exp1/'
S,A,H = 2,2,5
theta = 5e-2
rho = 5e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


##### Exp4 - theta>>rho

folder = 'results/exp4/'
S,A,H = 2,2,5
theta = 5e-1
rho = 3e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


##### Exp5 - theta=rho, large S,A

folder = 'results/exp5/'
S,A,H = 20,3,5
theta = 5e-2
rho = 5e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


##### Exp8 - larger H, no apx error

folder = 'results/exp8/'
S,A,H = 2,2,20
theta = 1e-1
rho = 1e-1  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


###################################### Question 3


##### Exp2 - piE Markovian

folder = 'results/exp2/'
S,A,H = 2,2,5
theta = 5e-2
rho = 3e-2  # rE takes on values multiple of rho
markov = True  # if piE is Markovian

# run simulation
run(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


###################################### Question 4


##### Exp7 - eta_hat vs RSBC

folder = 'results/exp7/'
S,A,H = 300,5,5
theta = 5e-2
rho = 3e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

# run simulation
run2(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)


###################################### Question 5


##### Exp9 - W-RS-GAIL vs RSBC

folder = 'results/exp9/'
S,A,H = 2,2,5
theta = 5e-1
rho = 5e-2  # rE takes on values multiple of rho
markov = False  # if piE is Markovian

n_mdps = 50  # number different mdps
n_DEs = 2  # number of different DE generation
values_N = [100,1000]  # number of expert's trajectories
eps = 1e-4  # precision for evaluation
M = 100000  # number of trajectories for evaluation

# run simulation
run3(S,A,H,theta,n_mdps,n_DEs,values_N,rho,markov,det,eps,M,folder,verbose)