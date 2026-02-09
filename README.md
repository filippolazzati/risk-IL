# Imitation Learning as Return Distribution Matching

This repository contains the code for the "Imitation Learning as Return
Distribution Matching" paper. Below, we describe how the repo is organized and
the dependencies.

## Repository Organization

There are two scripts for running and analyzing the results, i.e.,
```run.py``` and ```analysis.ipynb```. Specifically:

- Executing ```run.py``` permits to replicate the simulations in the paper, and it
dumps the results of the simulation in folder ```results```.

- ```analysis.ipynb``` allows to load from file (namely, from the files in
  ```results```) the results of the simulations, and to analyze them.

In addition to these files, notebook ```image.ipynb``` permits to generate the
plot in the Appendix, while folder ```src``` contains the code necessary for
running the experiments. In particular, ```src``` contains four different Python
files:

- ```algorithms.py``` contains the implementation of the four algorithms
  considered, i.e., RS-BC, RS-KT, BC and MIMIC-MD, plus the implementation of
  algorithm W_RS_GAIL from paper "Risk-Sensitive Generative Adversarial
  Imitation Learning".
- ```environment.py``` contains class MDP, that models an MDP, and offers
  methods for collecting trajectories and estimating the return distribution
  from a given dataset of trajectories.
- ```policies.py``` contains three classes: MarkovianPolicy,
  OurNonMarkovianPolicy and NonMarkovianPolicy, that respectively model
  Markovian policies, the class of non-Markovian policies that we consider in
  the paper, and general non-Markovian policies.
- ```utils.py``` contains various utility functions for randomly generating an
  expert's policy, computing the Wasserstein distance between two distributions,
  plotting return distributions, executing the simulations and showing the
  results.

## Dependencies
  
We have developed and executed the code using Python 3.11.9 and the following packages:

- cvxpy 1.7.2
- matplotlib 3.10.5
- numpy 2.3.2
- scipy 1.16.1

with related dependencies.
