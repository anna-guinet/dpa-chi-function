#!/usr/bin/env python

"""SUCCESS PROBA DPA 3-BIT CHI ROW & 3-BIT K """

__author__      = "Anna Guinet"
__email__       = "a.guinet@cs.ru.nl"
__version__     = "3.0"

import numpy as np
import itertools
import pandas as pd
from decimal import *
import math
import random
import matplotlib.pylab as plt
import argparse
import sys
import time

def noise(n=3):
	"""
	Compute a noise power consumption R vector for each possible message mu. 
	
	Parameter:
	n -- integer (defaut 3)
	
	Return:
	R -- list of integers, length 2^n
	"""
	R = []

	# For each message mu
	for j in range(2**n):
		
		# Activity of noise part
		d = random.gauss(0, 1)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def signal_1D(mu, K, kappa, n=3):
	"""
	Generate signal power consumption S values for a secret state key, for all messages mu. 
	
	Parameters:
	mu -- 2D list of bool, size n x 2^n
	K -- string
	kappa -- string
	n -- integer (default 3)
	
	Return:
	S -- list of integers, length 2^n
	"""
	# Transform strings into a list of booleans
	K = list(K)
	K = [bool(int(bit)) for bit in K]

	kappa = list(kappa)
	kappa = [bool(int(bit)) for bit in kappa]

	S = []

	# For each message mu
	for j in range(2**n):

		# Auxiliray
		S_j = int()

		# For each bit of each (K, kappa)
		for i in range(n):

			# Activity of the first storage cell of the register
			d = K[i] ^ kappa[i] ^ mu[j][i] ^ (kappa[(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (kappa[(i+2) % n] ^ mu[j][(i+2) % n])
		
			# Leakage model
			S_j += (-1)**(d)

		S.append(S_j)

	return S

def signal_2D(mu, K, n=3):
	"""
	Generate signal power consumption S referece values for a 3-bit kappa and for a bit of K, for all messages mu.

	Parameters:
	mu -- 2D list of bool, size n x 2^n
	K -- string
	n -- integer (default 3)
	
	Return:
	S -- 2D list of integers, size 2^n x 2^n
	"""

	# Transform string into list of booleans
	K = list(K)
	K = [bool(int(bit)) for bit in K]
	
	# All kappa possibilities
	kappa = list(itertools.product([bool(0), bool(1)], repeat=n))

	# Power consumption of signal part
	S = []

	# For each secret Kappa
	for l in range(2**n):

		S_j = []

		# For each message mu
		for j in range(2**n):

			S_ji = int()
			
			# For each bit i, in each message, for each key 
			for i in range(n):
			
				# Activity vector
				d = K[i] ^ kappa[l][i] ^ mu[j][i] ^ (kappa[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (kappa[l][(i+2) % n] ^ mu[j][(i+2) % n])

				S_ji += (-1)**(d)
				
			S_j.append(S_ji)
		
		S += [S_j]
	
	return 

def signal_3D(mu, n=3):
	"""
	Generate signal power consumption S reference values for 3 bits kappa and for 3 bits of K, for all messages mu.

	Parameters:
	mu -- 2D list of bool, size n x 2^n
	n -- integer (default 3)
	
	Return:
	S -- 3D list of integers, size 2^n x 2^n x 2^n
	"""

	# All K possibilities
	K = list(itertools.product([bool(0), bool(1)], repeat=n))

	# All kappa possibilities
	kappa = list(itertools.product([bool(0), bool(1)], repeat=n))

	# Power consumption of the signal part
	S = []

	# For each secret value K
	for m in range(2**n):

		S_l = []
		
		# For each secret kappa
		for l in range(2**n):

			S_lj = []

			# For each message mu
			for j in range(2**n):

				S_lji = int()

				# For each bit i, in each message, for each (K, kappa) 
				for i in range(n):

					# Activity vector
					d = K[m][i] ^ kappa[l][i] ^ mu[j][i] ^ (kappa[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (kappa[l][(i+2) % n] ^ mu[j][(i+2) % n])

					S_lji += (-1)**(d)

				S_lj.append(S_lji)

			S_l += [S_lj]
			
		S += [S_l]

	return S

def find_idx_scalar(scalar, value):
	"""
	Retrieve indexes of the value in the nested list scalar.
	The index of the sublist corresponds to a K value.
	The index of the element corresponds to a kappa value.

	Parameters:
	scalar -- list, size 2^n x 2^n
	value -- float
	"""

	# Give sublist index and element index in scalar if element == value
	indexes = [[sublist_idx, elem_idx] for sublist_idx,sublist in enumerate(scalar) for elem_idx, elem in enumerate(sublist) if elem==value]
	
	# Flat previous list
	#indexes = list(itertools.chain.from_iterable(indexes))

	return indexes

def find_wr(a, b, value_b):
	"""
	Find the point when for the scalar product of the solution equals the one of a nonsolution.
	
	Parameter:
	a -- Decimal
	b -- Decimal
	value_b -- float
	
	Return:
	y -- Decimal
	"""	
	value = Decimal(24) - Decimal(value_b)

	if (value + b - a) != 0: 
		# w = (b - a)/ (value + b - a)
		w = (b - a) / (value + b - a)

	else:
		w = 0

	return w

def find_rank(wr):
	"""
	Return the list of ranks for the solution kappa.
	
	Parameter:
	wr -- list of float
	
	Return:
	rank -- list of integer
	wr -- list of float
	"""

	# List of ranks
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if wr:
		
		# Count number of rank increment
		rank = [1 + count for count in range(len(wr), 0, -1)]

	rank += [1]
	
	return rank, wr

def pr_success(df, pr_end, pr_sim):
	"""
	Computes probability of success for a given rank.
	More preisely, it counts down from the max w value to the min one the probability of success
	according to the number of simulations (pr_sim).
	Therefore, we need to specify the end probability of success for a rank (pr_end).

	Parameters:
	df -- DataFrame
	pr_end -- integer
	pr_sim -- float
	
	Return:
	df -- DataFrame
	"""
	
	if not df.empty:
		# Add a new pr_sucess column with initial value = 1 for rank 1 and = 0 for lower ranks. 
		df.loc[0, 'pr_success'] = df.loc[0, 'wr']
		df.loc[0, 'pr_success'] = pr_end
	
		# Compute probabilities of success
		for i in range(1, len(df)):
			if df.at[i, 'wr'] < 0:
				df.loc[i, 'pr_success'] = df.loc[i-1, 'pr_success'] - pr_sim
			else:
				df.loc[i, 'pr_success'] = df.loc[i-1, 'pr_success'] + pr_sim

		# Round float in pr_success column
		decimals = 8 
		df['pr_success'] = df['pr_success'].apply(lambda x: round(x, decimals))
		df['wr'] = df['wr'].abs()

	else:
		df['wr'] = 0
		df['pr_success'] = pr_end

	return df

def sim_3bits(K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n -- integer
	K -- string
	kappa -- string
	num_sim -- integer

	Return:
	NaN
	"""
	n = 3

	# Signal message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	getcontext().prec = 10

	# All possible signal power consumption values
	S_ref = signal_3D(mu, n)

	# Save the results of the experiments
	rank_wr_list = []

	""" Find correlation with correct solution with final scalar product values """

	# Scalar product for full signal <S-ref, P = S>
	S = signal_1D(mu, K, kappa, n)
	P_fin = S
	scalar_fin = np.dot(S_ref, P_fin)

	# Find the indexes of the elements equals to a specific scalar product value
	scalar_fin_pos_24 = find_idx_scalar(scalar_fin, 24)
	scalar_fin_pos_16 = find_idx_scalar(scalar_fin, 16)
	scalar_fin_pos_8 = find_idx_scalar(scalar_fin, 8)
	scalar_fin_0 = find_idx_scalar(scalar_fin, 0)
	scalar_fin_neg_8 = find_idx_scalar(scalar_fin, -8)
	scalar_fin_neg_16 = find_idx_scalar(scalar_fin, -16)
	scalar_fin_neg_24 = find_idx_scalar(scalar_fin, -24)

	# Match the vectors with the secret values
	list_K_kappa_idx0 = [['K=000', 'kappa=000', 0, 0],
						 ['K=000', 'kappa=001', 0, 1],
						 ['K=000', 'kappa=010', 0, 2],
						 ['K=000', 'kappa=011', 0, 3],
						 ['K=000', 'kappa=100', 0, 4],
						 ['K=000', 'kappa=101', 0, 5],
						 ['K=000', 'kappa=110', 0, 6],
						 ['K=000', 'kappa=111', 0, 7]]

	list_K_kappa_idx1 = [['K=001', 'kappa=000', 1, 0],
						 ['K=001', 'kappa=001', 1, 1],
						 ['K=001', 'kappa=010', 1, 2],
						 ['K=001', 'kappa=011', 1, 3],
						 ['K=001', 'kappa=100', 1, 4],
						 ['K=001', 'kappa=101', 1, 5],
						 ['K=001', 'kappa=110', 1, 6],
						 ['K=001', 'kappa=111', 1, 7]]

	list_K_kappa_idx2 = [['K=010', 'kappa=000', 2, 0],
						 ['K=010', 'kappa=001', 2, 1],
						 ['K=010', 'kappa=010', 2, 2],
						 ['K=010', 'kappa=011', 2, 3],
						 ['K=010', 'kappa=100', 2, 4],
						 ['K=010', 'kappa=101', 2, 5],
						 ['K=010', 'kappa=110', 2, 6],
						 ['K=010', 'kappa=111', 2, 7]]

	list_K_kappa_idx3 = [['K=011', 'kappa=000', 3, 0],
						 ['K=011', 'kappa=001', 3, 1],
						 ['K=011', 'kappa=010', 3, 2],
						 ['K=011', 'kappa=011', 3, 3],
						 ['K=011', 'kappa=100', 3, 4],
						 ['K=011', 'kappa=101', 3, 5],
						 ['K=011', 'kappa=110', 3, 6],
						 ['K=011', 'kappa=111', 3, 7]]

	list_K_kappa_idx4 = [['K=100', 'kappa=000', 4, 0],
						 ['K=100', 'kappa=001', 4, 1],
						 ['K=100', 'kappa=010', 4, 2],
						 ['K=100', 'kappa=011', 4, 3],
						 ['K=100', 'kappa=100', 4, 4],
						 ['K=100', 'kappa=101', 4, 5],
						 ['K=100', 'kappa=110', 4, 6],
						 ['K=100', 'kappa=111', 4, 7]]

	list_K_kappa_idx5 = [['K=101', 'kappa=000', 5, 0],
						 ['K=101', 'kappa=001', 5, 1],
						 ['K=101', 'kappa=010', 5, 2],
						 ['K=101', 'kappa=011', 5, 3],
						 ['K=101', 'kappa=100', 5, 4],
						 ['K=101', 'kappa=101', 5, 5],
						 ['K=101', 'kappa=110', 5, 6],
						 ['K=101', 'kappa=111', 5, 7]]

	list_K_kappa_idx6 = [['K=110', 'kappa=000', 6, 0],
						 ['K=110', 'kappa=001', 6, 1],
						 ['K=110', 'kappa=010', 6, 2],
						 ['K=110', 'kappa=011', 6, 3],
						 ['K=110', 'kappa=100', 6, 4],
						 ['K=110', 'kappa=101', 6, 5],
						 ['K=110', 'kappa=110', 6, 6],
						 ['K=110', 'kappa=111', 6, 7]]

	list_K_kappa_idx7 = [['K=111', 'kappa=000', 7, 0],
						 ['K=111', 'kappa=001', 7, 1],
						 ['K=111', 'kappa=010', 7, 2],
						 ['K=111', 'kappa=011', 7, 3],
						 ['K=111', 'kappa=100', 7, 4],
						 ['K=111', 'kappa=101', 7, 5],
						 ['K=111', 'kappa=110', 7, 6],
						 ['K=111', 'kappa=111', 7, 7]]

	list_K_kappa_idx = [list_K_kappa_idx0, list_K_kappa_idx1, list_K_kappa_idx2, list_K_kappa_idx3, 
						list_K_kappa_idx4, list_K_kappa_idx5, list_K_kappa_idx6, list_K_kappa_idx7]

	# Find indexes of all correct functions
	correct_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_pos_24 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]

	# Find indexes of all correlated solutions for +16
	correlated_pos_16_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_pos_16 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all correlated solutions for +8
	correlated_pos_8_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_pos_8 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all uncorrelated solutions
	uncorrelated_0_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_0 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all correlated solutions for -8
	correlated_neg_8_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_neg_8 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all correlated solutions for -16
	correlated_neg_16_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_neg_16 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all correlated solutions for -24
	correlated_neg_24_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_neg_24 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]


	""" Initial values of scalar product """

	""" Generate many simulations """
	for j in range(num_sim):

		# Noise power consumption
		R = noise(n)

		# Global power consumption P_init = R
		P_init = [Decimal(r) for r in R]

		# Scalar product <S-ref, P>
		scalar_init = np.dot(S_ref, P_init)

		correct_init0 = scalar_init[correct_idx[0][2]][correct_idx[0][3]]
		
		# Degenerate case
		correct_init1 = None
		if len(correct_idx) == 2:
			correct_init1 = scalar_init[correct_idx[1][2]][correct_idx[1][3]]

		correlated_pos_16_init0 = scalar_init[correlated_pos_16_idx[0][2]][correlated_pos_16_idx[0][3]]
		correlated_pos_16_init1 = scalar_init[correlated_pos_16_idx[1][2]][correlated_pos_16_idx[1][3]]
		correlated_pos_16_init2 = scalar_init[correlated_pos_16_idx[2][2]][correlated_pos_16_idx[2][3]]
		correlated_pos_16_init3 = scalar_init[correlated_pos_16_idx[3][2]][correlated_pos_16_idx[3][3]]

		# Degenerated case
		if len(correlated_pos_16_idx) == 6:
			correlated_pos_16_init4 = scalar_init[correlated_pos_16_idx[4][2]][correlated_pos_16_idx[4][3]]
			correlated_pos_16_init5 = scalar_init[correlated_pos_16_idx[5][2]][correlated_pos_16_idx[5][3]]

		correlated_pos_8_init0 = scalar_init[correlated_pos_8_idx[0][2]][correlated_pos_8_idx[0][3]]
		correlated_pos_8_init1 = scalar_init[correlated_pos_8_idx[1][2]][correlated_pos_8_idx[1][3]]
		correlated_pos_8_init2 = scalar_init[correlated_pos_8_idx[2][2]][correlated_pos_8_idx[2][3]]
		correlated_pos_8_init3 = scalar_init[correlated_pos_8_idx[3][2]][correlated_pos_8_idx[3][3]]
		correlated_pos_8_init4 = scalar_init[correlated_pos_8_idx[4][2]][correlated_pos_8_idx[4][3]]
		correlated_pos_8_init5 = scalar_init[correlated_pos_8_idx[5][2]][correlated_pos_8_idx[5][3]]
		correlated_pos_8_init6 = scalar_init[correlated_pos_8_idx[6][2]][correlated_pos_8_idx[6][3]]
		correlated_pos_8_init7 = scalar_init[correlated_pos_8_idx[7][2]][correlated_pos_8_idx[7][3]]
		correlated_pos_8_init8 = scalar_init[correlated_pos_8_idx[8][2]][correlated_pos_8_idx[8][3]]
		correlated_pos_8_init9 = scalar_init[correlated_pos_8_idx[9][2]][correlated_pos_8_idx[9][3]]
		correlated_pos_8_init10 = scalar_init[correlated_pos_8_idx[10][2]][correlated_pos_8_idx[10][3]]
		correlated_pos_8_init11 = scalar_init[correlated_pos_8_idx[11][2]][correlated_pos_8_idx[11][3]]
		correlated_pos_8_init12 = scalar_init[correlated_pos_8_idx[12][2]][correlated_pos_8_idx[12][3]]
		correlated_pos_8_init13 = scalar_init[correlated_pos_8_idx[13][2]][correlated_pos_8_idx[13][3]]
		correlated_pos_8_init14 = scalar_init[correlated_pos_8_idx[14][2]][correlated_pos_8_idx[14][3]]
		correlated_pos_8_init15 = scalar_init[correlated_pos_8_idx[15][2]][correlated_pos_8_idx[15][3]]
		correlated_pos_8_init16 = scalar_init[correlated_pos_8_idx[16][2]][correlated_pos_8_idx[16][3]]
		correlated_pos_8_init17 = scalar_init[correlated_pos_8_idx[17][2]][correlated_pos_8_idx[17][3]]
		
		# Non degenerated case
		if len(correlated_pos_8_idx) == 19:
			correlated_pos_8_init18 = scalar_init[correlated_pos_8_idx[18][2]][correlated_pos_8_idx[18][3]]

		uncorrelated_0_init0 = scalar_init[uncorrelated_0_idx[0][2]][uncorrelated_0_idx[0][3]]
		uncorrelated_0_init1 = scalar_init[uncorrelated_0_idx[1][2]][uncorrelated_0_idx[1][3]]
		uncorrelated_0_init2 = scalar_init[uncorrelated_0_idx[2][2]][uncorrelated_0_idx[2][3]]
		uncorrelated_0_init3 = scalar_init[uncorrelated_0_idx[3][2]][uncorrelated_0_idx[3][3]]
		uncorrelated_0_init4 = scalar_init[uncorrelated_0_idx[4][2]][uncorrelated_0_idx[4][3]]
		uncorrelated_0_init5 = scalar_init[uncorrelated_0_idx[5][2]][uncorrelated_0_idx[5][3]]
		uncorrelated_0_init6 = scalar_init[uncorrelated_0_idx[6][2]][uncorrelated_0_idx[6][3]]
		uncorrelated_0_init7 = scalar_init[uncorrelated_0_idx[7][2]][uncorrelated_0_idx[7][3]]
		uncorrelated_0_init8 = scalar_init[uncorrelated_0_idx[8][2]][uncorrelated_0_idx[8][3]]
		uncorrelated_0_init9 = scalar_init[uncorrelated_0_idx[9][2]][uncorrelated_0_idx[9][3]]
		uncorrelated_0_init10 = scalar_init[uncorrelated_0_idx[10][2]][uncorrelated_0_idx[10][3]]
		uncorrelated_0_init11 = scalar_init[uncorrelated_0_idx[11][2]][uncorrelated_0_idx[11][3]]

		# Non degenerated case
		if len(uncorrelated_0_idx) == 16:
			uncorrelated_0_init12 = scalar_init[uncorrelated_0_idx[12][2]][uncorrelated_0_idx[12][3]]
			uncorrelated_0_init13 = scalar_init[uncorrelated_0_idx[13][2]][uncorrelated_0_idx[13][3]]
			uncorrelated_0_init14 = scalar_init[uncorrelated_0_idx[14][2]][uncorrelated_0_idx[14][3]]
			uncorrelated_0_init15 = scalar_init[uncorrelated_0_idx[15][2]][uncorrelated_0_idx[15][3]]

		correlated_neg_8_init0 = scalar_init[correlated_neg_8_idx[0][2]][correlated_neg_8_idx[0][3]]
		correlated_neg_8_init1 = scalar_init[correlated_neg_8_idx[1][2]][correlated_neg_8_idx[1][3]]
		correlated_neg_8_init2 = scalar_init[correlated_neg_8_idx[2][2]][correlated_neg_8_idx[2][3]]
		correlated_neg_8_init3 = scalar_init[correlated_neg_8_idx[3][2]][correlated_neg_8_idx[3][3]]
		correlated_neg_8_init4 = scalar_init[correlated_neg_8_idx[4][2]][correlated_neg_8_idx[4][3]]
		correlated_neg_8_init5 = scalar_init[correlated_neg_8_idx[5][2]][correlated_neg_8_idx[5][3]]
		correlated_neg_8_init6 = scalar_init[correlated_neg_8_idx[6][2]][correlated_neg_8_idx[6][3]]
		correlated_neg_8_init7 = scalar_init[correlated_neg_8_idx[7][2]][correlated_neg_8_idx[7][3]]
		correlated_neg_8_init8 = scalar_init[correlated_neg_8_idx[8][2]][correlated_neg_8_idx[8][3]]
		correlated_neg_8_init9 = scalar_init[correlated_neg_8_idx[9][2]][correlated_neg_8_idx[9][3]]
		correlated_neg_8_init10 = scalar_init[correlated_neg_8_idx[10][2]][correlated_neg_8_idx[10][3]]
		correlated_neg_8_init11 = scalar_init[correlated_neg_8_idx[11][2]][correlated_neg_8_idx[11][3]]
		correlated_neg_8_init12 = scalar_init[correlated_neg_8_idx[12][2]][correlated_neg_8_idx[12][3]]
		correlated_neg_8_init13 = scalar_init[correlated_neg_8_idx[13][2]][correlated_neg_8_idx[13][3]]
		correlated_neg_8_init14 = scalar_init[correlated_neg_8_idx[14][2]][correlated_neg_8_idx[14][3]]
		correlated_neg_8_init15 = scalar_init[correlated_neg_8_idx[15][2]][correlated_neg_8_idx[15][3]]
		correlated_neg_8_init16 = scalar_init[correlated_neg_8_idx[16][2]][correlated_neg_8_idx[16][3]]
		correlated_neg_8_init17 = scalar_init[correlated_neg_8_idx[17][2]][correlated_neg_8_idx[17][3]]
		
		# Non degenerated case
		if len(correlated_neg_8_idx) == 19:
			correlated_neg_8_init18 = scalar_init[correlated_neg_8_idx[18][2]][correlated_neg_8_idx[18][3]]
		
		correlated_neg_16_init0 = scalar_init[correlated_neg_16_idx[0][2]][correlated_neg_16_idx[0][3]]
		correlated_neg_16_init1 = scalar_init[correlated_neg_16_idx[1][2]][correlated_neg_16_idx[1][3]]
		correlated_neg_16_init2 = scalar_init[correlated_neg_16_idx[2][2]][correlated_neg_16_idx[2][3]]
		correlated_neg_16_init3 = scalar_init[correlated_neg_16_idx[3][2]][correlated_neg_16_idx[3][3]]

		# Degenerated case
		if len(correlated_neg_16_idx) == 6:
			correlated_neg_16_init4 = scalar_init[correlated_neg_16_idx[4][2]][correlated_neg_16_idx[4][3]]
			correlated_neg_16_init5 = scalar_init[correlated_neg_16_idx[5][2]][correlated_neg_16_idx[5][3]]

		correlated_neg_24_init0 = scalar_init[correlated_neg_24_idx[0][2]][correlated_neg_24_idx[0][3]]

		# Degenerated case
		if len(correlated_neg_24_idx) == 2:
			correlated_neg_24_init1 = scalar_init[correlated_neg_24_idx[1][2]][correlated_neg_24_idx[1][3]]


		""" Find the functions according to the Ks and kappas with correct function """

		wr_correlated_pos_16_0 = find_wr(correct_init0, correlated_pos_16_init0, 16)
		wr_correlated_pos_16_1 = find_wr(correct_init0, correlated_pos_16_init1, 16)
		wr_correlated_pos_16_2 = find_wr(correct_init0, correlated_pos_16_init2, 16)
		wr_correlated_pos_16_3 = find_wr(correct_init0, correlated_pos_16_init3, 16)

		wr_correlated_pos_16 = [wr_correlated_pos_16_0, wr_correlated_pos_16_1, wr_correlated_pos_16_2, wr_correlated_pos_16_3]

		# Degenerated case
		if len(correlated_pos_16_idx) == 6:
			wr_correlated_pos_16_4 = find_wr(correct_init0, correlated_pos_16_init4, 16)
			wr_correlated_pos_16_5 = find_wr(correct_init0, correlated_pos_16_init5, 16)

			wr_correlated_pos_16.append(wr_correlated_pos_16_4)
			wr_correlated_pos_16.append(wr_correlated_pos_16_5)

		# Drop values < 0 and > 1
		wr_correlated_pos_16 = [w for w in wr_correlated_pos_16 if (w > 0) and (w < 1)]


		wr_correlated_pos_8_0 = find_wr(correct_init0, correlated_pos_8_init0, 8)
		wr_correlated_pos_8_1 = find_wr(correct_init0, correlated_pos_8_init1, 8)
		wr_correlated_pos_8_2 = find_wr(correct_init0, correlated_pos_8_init2, 8)
		wr_correlated_pos_8_3 = find_wr(correct_init0, correlated_pos_8_init3, 8)
		wr_correlated_pos_8_4 = find_wr(correct_init0, correlated_pos_8_init4, 8)
		wr_correlated_pos_8_5 = find_wr(correct_init0, correlated_pos_8_init5, 8)
		wr_correlated_pos_8_6 = find_wr(correct_init0, correlated_pos_8_init6, 8)
		wr_correlated_pos_8_7 = find_wr(correct_init0, correlated_pos_8_init7, 8)
		wr_correlated_pos_8_8 = find_wr(correct_init0, correlated_pos_8_init8, 8)
		wr_correlated_pos_8_9 = find_wr(correct_init0, correlated_pos_8_init9, 8)
		wr_correlated_pos_8_10 = find_wr(correct_init0, correlated_pos_8_init10, 8)
		wr_correlated_pos_8_11 = find_wr(correct_init0, correlated_pos_8_init11, 8)
		wr_correlated_pos_8_12 = find_wr(correct_init0, correlated_pos_8_init12, 8)
		wr_correlated_pos_8_13 = find_wr(correct_init0, correlated_pos_8_init13, 8)
		wr_correlated_pos_8_14 = find_wr(correct_init0, correlated_pos_8_init14, 8)
		wr_correlated_pos_8_15 = find_wr(correct_init0, correlated_pos_8_init15, 8)
		wr_correlated_pos_8_16 = find_wr(correct_init0, correlated_pos_8_init16, 8)
		wr_correlated_pos_8_17 = find_wr(correct_init0, correlated_pos_8_init17, 8)

		wr_correlated_pos_8 = [wr_correlated_pos_8_0, wr_correlated_pos_8_1, wr_correlated_pos_8_2, wr_correlated_pos_8_3,
								  wr_correlated_pos_8_4, wr_correlated_pos_8_5, wr_correlated_pos_8_6, wr_correlated_pos_8_7,
								  wr_correlated_pos_8_8, wr_correlated_pos_8_9, wr_correlated_pos_8_10, wr_correlated_pos_8_11,
								  wr_correlated_pos_8_12, wr_correlated_pos_8_13, wr_correlated_pos_8_14, wr_correlated_pos_8_15,
								  wr_correlated_pos_8_16, wr_correlated_pos_8_17]
		
		# Non degenerated case
		if len(correlated_pos_8_idx) == 19:
			wr_correlated_pos_8_18 = find_wr(correct_init0, correlated_pos_8_init18, 8)
			wr_correlated_pos_8.append(wr_correlated_pos_8_18)
			
		# Drop values < 0 and > 1
		wr_correlated_pos_8 = [w for w in wr_correlated_pos_8 if (w > 0) and (w < 1)]


		wr_uncorrelated_0_0 = find_wr(correct_init0, uncorrelated_0_init0, 0)
		wr_uncorrelated_0_1 = find_wr(correct_init0, uncorrelated_0_init1, 0)
		wr_uncorrelated_0_2 = find_wr(correct_init0, uncorrelated_0_init2, 0)
		wr_uncorrelated_0_3 = find_wr(correct_init0, uncorrelated_0_init3, 0)
		wr_uncorrelated_0_4 = find_wr(correct_init0, uncorrelated_0_init4, 0)
		wr_uncorrelated_0_5 = find_wr(correct_init0, uncorrelated_0_init5, 0)
		wr_uncorrelated_0_6 = find_wr(correct_init0, uncorrelated_0_init6, 0)
		wr_uncorrelated_0_7 = find_wr(correct_init0, uncorrelated_0_init7, 0)
		wr_uncorrelated_0_8 = find_wr(correct_init0, uncorrelated_0_init8, 0)
		wr_uncorrelated_0_9 = find_wr(correct_init0, uncorrelated_0_init9, 0)
		wr_uncorrelated_0_10 = find_wr(correct_init0, uncorrelated_0_init10, 0)
		wr_uncorrelated_0_11 = find_wr(correct_init0, uncorrelated_0_init11, 0)

		wr_uncorrelated_0 = [wr_uncorrelated_0_0, wr_uncorrelated_0_1, wr_uncorrelated_0_2, wr_uncorrelated_0_3,
								wr_uncorrelated_0_4, wr_uncorrelated_0_5, wr_uncorrelated_0_6, wr_uncorrelated_0_7,
								wr_uncorrelated_0_8, wr_uncorrelated_0_9, wr_uncorrelated_0_10, wr_uncorrelated_0_11]

		# Non degenerated case
		if len(uncorrelated_0_idx) == 16:
			wr_uncorrelated_0_12 = find_wr(correct_init0, uncorrelated_0_init12, 0)
			wr_uncorrelated_0_13 = find_wr(correct_init0, uncorrelated_0_init13, 0)
			wr_uncorrelated_0_14 = find_wr(correct_init0, uncorrelated_0_init14, 0)
			wr_uncorrelated_0_15 = find_wr(correct_init0, uncorrelated_0_init15, 0)

			wr_uncorrelated_0.append(wr_uncorrelated_0_12)
			wr_uncorrelated_0.append(wr_uncorrelated_0_13)
			wr_uncorrelated_0.append(wr_uncorrelated_0_14)
			wr_uncorrelated_0.append(wr_uncorrelated_0_15)

		# Drop values < 0 and > 1
		wr_uncorrelated_0 = [w for w in wr_uncorrelated_0 if (w > 0) and (w < 1)]


		wr_correlated_neg_8_0 = find_wr(correct_init0, correlated_neg_8_init0, -8)
		wr_correlated_neg_8_1 = find_wr(correct_init0, correlated_neg_8_init1, -8)
		wr_correlated_neg_8_2 = find_wr(correct_init0, correlated_neg_8_init2, -8)
		wr_correlated_neg_8_3 = find_wr(correct_init0, correlated_neg_8_init3, -8)
		wr_correlated_neg_8_4 = find_wr(correct_init0, correlated_neg_8_init4, -8)
		wr_correlated_neg_8_5 = find_wr(correct_init0, correlated_neg_8_init5, -8)
		wr_correlated_neg_8_6 = find_wr(correct_init0, correlated_neg_8_init6, -8)
		wr_correlated_neg_8_7 = find_wr(correct_init0, correlated_neg_8_init7, -8)
		wr_correlated_neg_8_8 = find_wr(correct_init0, correlated_neg_8_init8, -8)
		wr_correlated_neg_8_9 = find_wr(correct_init0, correlated_neg_8_init9, -8)
		wr_correlated_neg_8_10 = find_wr(correct_init0, correlated_neg_8_init10, -8)
		wr_correlated_neg_8_11 = find_wr(correct_init0, correlated_neg_8_init11, -8)
		wr_correlated_neg_8_12 = find_wr(correct_init0, correlated_neg_8_init12, -8)
		wr_correlated_neg_8_13 = find_wr(correct_init0, correlated_neg_8_init13, -8)
		wr_correlated_neg_8_14 = find_wr(correct_init0, correlated_neg_8_init14, -8)
		wr_correlated_neg_8_15 = find_wr(correct_init0, correlated_neg_8_init15, -8)
		wr_correlated_neg_8_16 = find_wr(correct_init0, correlated_neg_8_init16, -8)
		wr_correlated_neg_8_17 = find_wr(correct_init0, correlated_neg_8_init17, -8)

		wr_correlated_neg_8 = [wr_correlated_neg_8_0, wr_correlated_neg_8_1, wr_correlated_neg_8_2, wr_correlated_neg_8_3,
								  wr_correlated_neg_8_4, wr_correlated_neg_8_5, wr_correlated_neg_8_6, wr_correlated_neg_8_7,
								  wr_correlated_neg_8_8, wr_correlated_neg_8_9, wr_correlated_neg_8_10, wr_correlated_neg_8_11,
								  wr_correlated_neg_8_12, wr_correlated_neg_8_13, wr_correlated_neg_8_14, wr_correlated_neg_8_15,
								  wr_correlated_neg_8_16, wr_correlated_neg_8_17]
		
		# Non degenerated case
		if len(correlated_neg_8_idx) == 19:
			wr_correlated_neg_8_18 = find_wr(correct_init0, correlated_neg_8_init18, -8)
			wr_correlated_neg_8.append(wr_correlated_neg_8_18)
			
		# Drop values < 0 and > 1
		wr_correlated_neg_8 = [w for w in wr_correlated_neg_8 if (w > 0) and (w < 1)]


		wr_correlated_neg_16_0 = find_wr(correct_init0, correlated_neg_16_init0, -16)
		wr_correlated_neg_16_1 = find_wr(correct_init0, correlated_neg_16_init1, -16)
		wr_correlated_neg_16_2 = find_wr(correct_init0, correlated_neg_16_init2, -16)
		wr_correlated_neg_16_3 = find_wr(correct_init0, correlated_neg_16_init3, -16)

		wr_correlated_neg_16 = [wr_correlated_neg_16_0, wr_correlated_neg_16_1, wr_correlated_neg_16_2, wr_correlated_neg_16_3]

		# Degenerated case
		if len(correlated_neg_16_idx) == 6:
			wr_correlated_neg_16_4 = find_wr(correct_init0, correlated_neg_16_init4, -16)
			wr_correlated_neg_16_5 = find_wr(correct_init0, correlated_neg_16_init5, -16)

			wr_correlated_neg_16.append(wr_correlated_neg_16_4)
			wr_correlated_neg_16.append(wr_correlated_neg_16_5)

		# Drop values < 0 and > 1
		wr_correlated_neg_16 = [w for w in wr_correlated_neg_16 if (w > 0) and (w < 1)]


		wr_correlated_neg_24_0 = find_wr(correct_init0, correlated_neg_24_init0, -24)
		wr_correlated_neg_24 = [wr_correlated_neg_24_0]

		# Degenerated case
		if len(correlated_neg_24_idx) == 2:
			wr_correlated_neg_24_1 = find_wr(correct_init0, correlated_neg_24_init1, -24)
			wr_correlated_neg_24.append(wr_correlated_neg_24_1)

		# Drop values < 0 and > 1
		wr_correlated_neg_24 = [w for w in wr_correlated_neg_24 if (w > 0) and (w < 1)]

		""" Find rank of correct kappa """
		wr = wr_correlated_pos_16
		wr += wr_correlated_pos_8
		wr += wr_uncorrelated_0
		wr += wr_correlated_neg_8
		wr += wr_correlated_neg_16
		wr += wr_correlated_neg_24

		# Transform Decimal into float
		getcontext().prec = 5
		wr = [float(w) for w in wr]

		# Sort by w
		wr = sorted(wr)
				
		rank, wr = find_rank(wr)
		
		# Expand the lists for plotting
		rank = [r for r in zip(rank, rank)]
		wr = [i for i in zip(wr, wr)]
				
		# Un-nest the previous lists
		rank = list(itertools.chain.from_iterable(rank))
		wr = list(itertools.chain.from_iterable(wr))
				
		# Extract abscissas for intersection points
		wr.insert(0, 0)
		wr.append(1)
			
		# Cut the edges
		rank = rank[1:-1]
		wr = wr[1:-1]

		""" Save data """
		# w_i = (-1)^i * w_i
		# To determine the intervals of the rank values:
		# - negative value is the beginning of the interval from the origin
		# - positive value is the end of the interval from the origin
		wr = [((-1)**i)*wr[i] for i in range(len(wr))]
			
		# Group to save and manipulate later on
		rank_wr = [i for i in zip(rank, wr)]
			
		# Save result for the specific noise vector R
		rank_wr_list += rank_wr
			
	return rank_wr_list

def sim_3bits_to_csv(K, kappa, num_sim):
	"""
	Plot probabilities of success for all possibilities for several simulations. 
	
	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	
	Return:
	NaN
	"""
	
	rank_wr_list = sim_3bits(K, kappa, num_sim)

	""" Save data in a DataFrame """

	# Prepare dataset by sorting by rank first
	rank_wr_list = sorted(rank_wr_list)
	rank_wr_df = pd.DataFrame(rank_wr_list, columns=['rank', 'wr'])

	# Drop rows with zero values for w
	rank_wr_df = rank_wr_df[(rank_wr_df != 0).all(1)]

	""" Create DataFrame by ranks and compute their probabilities of success """

	# Create DataFrame for each rank
	rank_wr_df1 = rank_wr_df.loc[rank_wr_df['rank'] == 1]
	rank_wr_df2 = rank_wr_df.loc[rank_wr_df['rank'] == 2]
	rank_wr_df3 = rank_wr_df.loc[rank_wr_df['rank'] == 3]
	rank_wr_df4 = rank_wr_df.loc[rank_wr_df['rank'] == 4]
	rank_wr_df5 = rank_wr_df.loc[rank_wr_df['rank'] == 5]
	rank_wr_df6 = rank_wr_df.loc[rank_wr_df['rank'] == 6]
	rank_wr_df7 = rank_wr_df.loc[rank_wr_df['rank'] == 7]
	rank_wr_df8 = rank_wr_df.loc[rank_wr_df['rank'] == 8]
	rank_wr_df9 = rank_wr_df.loc[rank_wr_df['rank'] == 9]
	rank_wr_df10 = rank_wr_df.loc[rank_wr_df['rank'] == 10]
	rank_wr_df11 = rank_wr_df.loc[rank_wr_df['rank'] == 11]
	rank_wr_df12 = rank_wr_df.loc[rank_wr_df['rank'] == 12]
	rank_wr_df13 = rank_wr_df.loc[rank_wr_df['rank'] == 13]
	rank_wr_df14 = rank_wr_df.loc[rank_wr_df['rank'] == 14]
	rank_wr_df15 = rank_wr_df.loc[rank_wr_df['rank'] == 15]
	rank_wr_df16 = rank_wr_df.loc[rank_wr_df['rank'] == 16]
	rank_wr_df17 = rank_wr_df.loc[rank_wr_df['rank'] == 17]
	rank_wr_df18 = rank_wr_df.loc[rank_wr_df['rank'] == 18]
	rank_wr_df19 = rank_wr_df.loc[rank_wr_df['rank'] == 19]
	rank_wr_df20 = rank_wr_df.loc[rank_wr_df['rank'] == 20]
	rank_wr_df21 = rank_wr_df.loc[rank_wr_df['rank'] == 21]
	rank_wr_df22 = rank_wr_df.loc[rank_wr_df['rank'] == 22]
	rank_wr_df23 = rank_wr_df.loc[rank_wr_df['rank'] == 23]
	rank_wr_df24 = rank_wr_df.loc[rank_wr_df['rank'] == 24]
	rank_wr_df25 = rank_wr_df.loc[rank_wr_df['rank'] == 25]
	rank_wr_df26 = rank_wr_df.loc[rank_wr_df['rank'] == 26]
	rank_wr_df27 = rank_wr_df.loc[rank_wr_df['rank'] == 27]
	rank_wr_df28 = rank_wr_df.loc[rank_wr_df['rank'] == 28]
	rank_wr_df29 = rank_wr_df.loc[rank_wr_df['rank'] == 29]
	rank_wr_df30 = rank_wr_df.loc[rank_wr_df['rank'] == 30]
	rank_wr_df31 = rank_wr_df.loc[rank_wr_df['rank'] == 31]
	rank_wr_df32 = rank_wr_df.loc[rank_wr_df['rank'] == 32]
	rank_wr_df33 = rank_wr_df.loc[rank_wr_df['rank'] == 33]
	rank_wr_df34 = rank_wr_df.loc[rank_wr_df['rank'] == 34]
	rank_wr_df35 = rank_wr_df.loc[rank_wr_df['rank'] == 35]
	rank_wr_df36 = rank_wr_df.loc[rank_wr_df['rank'] == 36]
	rank_wr_df37 = rank_wr_df.loc[rank_wr_df['rank'] == 37]
	rank_wr_df38 = rank_wr_df.loc[rank_wr_df['rank'] == 38]
	rank_wr_df39 = rank_wr_df.loc[rank_wr_df['rank'] == 39]
	rank_wr_df40 = rank_wr_df.loc[rank_wr_df['rank'] == 40]
	rank_wr_df41 = rank_wr_df.loc[rank_wr_df['rank'] == 41]
	rank_wr_df42 = rank_wr_df.loc[rank_wr_df['rank'] == 42]
	rank_wr_df43 = rank_wr_df.loc[rank_wr_df['rank'] == 43]
	rank_wr_df44 = rank_wr_df.loc[rank_wr_df['rank'] == 44]
	rank_wr_df45 = rank_wr_df.loc[rank_wr_df['rank'] == 45]
	rank_wr_df46 = rank_wr_df.loc[rank_wr_df['rank'] == 46]
	rank_wr_df47 = rank_wr_df.loc[rank_wr_df['rank'] == 47]
	rank_wr_df48 = rank_wr_df.loc[rank_wr_df['rank'] == 48]
	rank_wr_df49 = rank_wr_df.loc[rank_wr_df['rank'] == 49]
	rank_wr_df50 = rank_wr_df.loc[rank_wr_df['rank'] == 50]
	rank_wr_df51 = rank_wr_df.loc[rank_wr_df['rank'] == 51]
	rank_wr_df52 = rank_wr_df.loc[rank_wr_df['rank'] == 52]
	rank_wr_df53 = rank_wr_df.loc[rank_wr_df['rank'] == 53]
	rank_wr_df54 = rank_wr_df.loc[rank_wr_df['rank'] == 54]
	rank_wr_df55 = rank_wr_df.loc[rank_wr_df['rank'] == 55]
	rank_wr_df56 = rank_wr_df.loc[rank_wr_df['rank'] == 56]
	rank_wr_df57 = rank_wr_df.loc[rank_wr_df['rank'] == 57]
	rank_wr_df58 = rank_wr_df.loc[rank_wr_df['rank'] == 58]
	rank_wr_df59 = rank_wr_df.loc[rank_wr_df['rank'] == 59]
	rank_wr_df60 = rank_wr_df.loc[rank_wr_df['rank'] == 60]
	rank_wr_df61 = rank_wr_df.loc[rank_wr_df['rank'] == 61]
	rank_wr_df62 = rank_wr_df.loc[rank_wr_df['rank'] == 62]
	rank_wr_df63 = rank_wr_df.loc[rank_wr_df['rank'] == 63]
	rank_wr_df64 = rank_wr_df.loc[rank_wr_df['rank'] == 64]

	# Sort by absolute value, descending order
	if not rank_wr_df1.empty:
		rank_wr_df1 = rank_wr_df1.iloc[(-rank_wr_df1['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df2.empty:
		rank_wr_df2 = rank_wr_df2.iloc[(-rank_wr_df2['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df3.empty:
		rank_wr_df3 = rank_wr_df3.iloc[(-rank_wr_df3['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df4.empty:
		rank_wr_df4 = rank_wr_df4.iloc[(-rank_wr_df4['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df5.empty:
		rank_wr_df5 = rank_wr_df5.iloc[(-rank_wr_df5['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df6.empty:
		rank_wr_df6 = rank_wr_df6.iloc[(-rank_wr_df6['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df7.empty:
		rank_wr_df7 = rank_wr_df7.iloc[(-rank_wr_df7['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df8.empty:
		rank_wr_df8 = rank_wr_df8.iloc[(-rank_wr_df8['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df9.empty:
		rank_wr_df9 = rank_wr_df9.iloc[(-rank_wr_df9['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df10.empty:
		rank_wr_df10 = rank_wr_df10.iloc[(-rank_wr_df10['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df11.empty:
		rank_wr_df11 = rank_wr_df11.iloc[(-rank_wr_df11['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df12.empty:
		rank_wr_df12 = rank_wr_df12.iloc[(-rank_wr_df12['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df13.empty:
		rank_wr_df13 = rank_wr_df13.iloc[(-rank_wr_df13['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df14.empty:
		rank_wr_df14 = rank_wr_df14.iloc[(-rank_wr_df14['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df15.empty:
		rank_wr_df15 = rank_wr_df15.iloc[(-rank_wr_df15['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df16.empty:
		rank_wr_df16 = rank_wr_df16.iloc[(-rank_wr_df16['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df17.empty:
		rank_wr_df17 = rank_wr_df17.iloc[(-rank_wr_df17['wr'].abs()).argsort()].reset_index(drop=True)
		
	if not rank_wr_df18.empty:
		rank_wr_df18 = rank_wr_df18.iloc[(-rank_wr_df18['wr'].abs()).argsort()].reset_index(drop=True)
		
	if not rank_wr_df19.empty:
		rank_wr_df19 = rank_wr_df19.iloc[(-rank_wr_df19['wr'].abs()).argsort()].reset_index(drop=True)
		
	if not rank_wr_df20.empty:
		rank_wr_df20 = rank_wr_df20.iloc[(-rank_wr_df20['wr'].abs()).argsort()].reset_index(drop=True)
		
	if not rank_wr_df21.empty:
		rank_wr_df21 = rank_wr_df21.iloc[(-rank_wr_df21['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df22.empty:
		rank_wr_df22 = rank_wr_df22.iloc[(-rank_wr_df22['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df23.empty:
		rank_wr_df23 = rank_wr_df23.iloc[(-rank_wr_df23['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df24.empty:
		rank_wr_df24 = rank_wr_df24.iloc[(-rank_wr_df24['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df25.empty:
		rank_wr_df25 = rank_wr_df25.iloc[(-rank_wr_df25['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df26.empty:
		rank_wr_df26 = rank_wr_df26.iloc[(-rank_wr_df26['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df27.empty:
		rank_wr_df27 = rank_wr_df27.iloc[(-rank_wr_df27['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df28.empty:
		rank_wr_df28 = rank_wr_df28.iloc[(-rank_wr_df28['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df29.empty:
		rank_wr_df29 = rank_wr_df29.iloc[(-rank_wr_df29['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df30.empty:
		rank_wr_df30 = rank_wr_df30.iloc[(-rank_wr_df30['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df31.empty:
		rank_wr_df31 = rank_wr_df31.iloc[(-rank_wr_df31['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df32.empty:
		rank_wr_df32 = rank_wr_df32.iloc[(-rank_wr_df32['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df33.empty:
		rank_wr_df33 = rank_wr_df33.iloc[(-rank_wr_df33['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df34.empty:
		rank_wr_df34 = rank_wr_df34.iloc[(-rank_wr_df34['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df35.empty:
		rank_wr_df35 = rank_wr_df35.iloc[(-rank_wr_df35['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df36.empty:
		rank_wr_df36 = rank_wr_df36.iloc[(-rank_wr_df36['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df37.empty:
		rank_wr_df37 = rank_wr_df37.iloc[(-rank_wr_df37['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df38.empty:
		rank_wr_df38 = rank_wr_df38.iloc[(-rank_wr_df38['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df39.empty:
		rank_wr_df39 = rank_wr_df39.iloc[(-rank_wr_df39['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df40.empty:
		rank_wr_df40 = rank_wr_df40.iloc[(-rank_wr_df40['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df41.empty:
		rank_wr_df41 = rank_wr_df41.iloc[(-rank_wr_df41['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df42.empty:
		rank_wr_df42 = rank_wr_df42.iloc[(-rank_wr_df42['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df43.empty:
		rank_wr_df43 = rank_wr_df43.iloc[(-rank_wr_df43['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df44.empty:
		rank_wr_df44 = rank_wr_df44.iloc[(-rank_wr_df44['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df45.empty:
		rank_wr_df45 = rank_wr_df45.iloc[(-rank_wr_df45['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df46.empty:
		rank_wr_df46 = rank_wr_df46.iloc[(-rank_wr_df46['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df47.empty:
		rank_wr_df47 = rank_wr_df47.iloc[(-rank_wr_df47['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df48.empty:
		rank_wr_df48 = rank_wr_df48.iloc[(-rank_wr_df48['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df49.empty:
		rank_wr_df49 = rank_wr_df49.iloc[(-rank_wr_df49['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df50.empty:
		rank_wr_df50 = rank_wr_df50.iloc[(-rank_wr_df50['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df51.empty:
		rank_wr_df51 = rank_wr_df51.iloc[(-rank_wr_df51['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df52.empty:
		rank_wr_df52 = rank_wr_df52.iloc[(-rank_wr_df52['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df53.empty:
		rank_wr_df53 = rank_wr_df53.iloc[(-rank_wr_df53['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df54.empty:	
		rank_wr_df54 = rank_wr_df54.iloc[(-rank_wr_df54['wr'].abs()).argsort()].reset_index(drop=True)
	
	if not rank_wr_df55.empty:
		rank_wr_df55 = rank_wr_df55.iloc[(-rank_wr_df55['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df56.empty:
		rank_wr_df56 = rank_wr_df56.iloc[(-rank_wr_df56['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df57.empty:
		rank_wr_df57 = rank_wr_df57.iloc[(-rank_wr_df57['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df58.empty:
		rank_wr_df58 = rank_wr_df58.iloc[(-rank_wr_df58['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df59.empty:
		rank_wr_df59 = rank_wr_df59.iloc[(-rank_wr_df59['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df60.empty:
		rank_wr_df60 = rank_wr_df60.iloc[(-rank_wr_df60['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df61.empty:
		rank_wr_df61 = rank_wr_df61.iloc[(-rank_wr_df61['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df62.empty:
		rank_wr_df62 = rank_wr_df62.iloc[(-rank_wr_df62['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df63.empty:
		rank_wr_df63 = rank_wr_df63.iloc[(-rank_wr_df63['wr'].abs()).argsort()].reset_index(drop=True)

	if not rank_wr_df64.empty:
		rank_wr_df64 = rank_wr_df64.iloc[(-rank_wr_df64['wr'].abs()).argsort()].reset_index(drop=True)

	# Probability of a simulation
	pr_sim = 1 / num_sim
	
	# Compute probabilities of success
	rank_wr_df1 = pr_success(rank_wr_df1, 1, pr_sim)
	rank_wr_df2 = pr_success(rank_wr_df2, 0, pr_sim)
	rank_wr_df3 = pr_success(rank_wr_df3, 0, pr_sim)
	rank_wr_df4 = pr_success(rank_wr_df4, 0, pr_sim)
	rank_wr_df5 = pr_success(rank_wr_df5, 0, pr_sim)
	rank_wr_df6 = pr_success(rank_wr_df6, 0, pr_sim)
	rank_wr_df7 = pr_success(rank_wr_df7, 0, pr_sim)
	rank_wr_df8 = pr_success(rank_wr_df8, 0, pr_sim)
	rank_wr_df9 = pr_success(rank_wr_df9, 0, pr_sim)
	rank_wr_df10 = pr_success(rank_wr_df10, 0, pr_sim)
	rank_wr_df11 = pr_success(rank_wr_df11, 0, pr_sim)
	rank_wr_df12 = pr_success(rank_wr_df12, 0, pr_sim)
	rank_wr_df13 = pr_success(rank_wr_df13, 0, pr_sim)
	rank_wr_df14 = pr_success(rank_wr_df14, 0, pr_sim)
	rank_wr_df15 = pr_success(rank_wr_df15, 0, pr_sim)
	rank_wr_df16 = pr_success(rank_wr_df16, 0, pr_sim)
	rank_wr_df17 = pr_success(rank_wr_df17, 0, pr_sim)
	rank_wr_df18 = pr_success(rank_wr_df18, 0, pr_sim)
	rank_wr_df19 = pr_success(rank_wr_df19, 0, pr_sim)
	rank_wr_df20 = pr_success(rank_wr_df20, 0, pr_sim)
	rank_wr_df21 = pr_success(rank_wr_df21, 0, pr_sim)
	rank_wr_df22 = pr_success(rank_wr_df22, 0, pr_sim)
	rank_wr_df23 = pr_success(rank_wr_df23, 0, pr_sim)
	rank_wr_df24 = pr_success(rank_wr_df24, 0, pr_sim)
	rank_wr_df25 = pr_success(rank_wr_df25, 0, pr_sim)
	rank_wr_df26 = pr_success(rank_wr_df26, 0, pr_sim)
	rank_wr_df27 = pr_success(rank_wr_df27, 0, pr_sim)
	rank_wr_df28 = pr_success(rank_wr_df28, 0, pr_sim)
	rank_wr_df29 = pr_success(rank_wr_df29, 0, pr_sim)
	rank_wr_df30 = pr_success(rank_wr_df30, 0, pr_sim)
	rank_wr_df31 = pr_success(rank_wr_df31, 0, pr_sim)
	rank_wr_df32 = pr_success(rank_wr_df32, 0, pr_sim)
	rank_wr_df33 = pr_success(rank_wr_df33, 0, pr_sim)
	rank_wr_df34 = pr_success(rank_wr_df34, 0, pr_sim)
	rank_wr_df35 = pr_success(rank_wr_df35, 0, pr_sim)
	rank_wr_df36 = pr_success(rank_wr_df36, 0, pr_sim)
	rank_wr_df37 = pr_success(rank_wr_df37, 0, pr_sim)
	rank_wr_df38 = pr_success(rank_wr_df38, 0, pr_sim)
	rank_wr_df39 = pr_success(rank_wr_df39, 0, pr_sim)
	rank_wr_df40 = pr_success(rank_wr_df40, 0, pr_sim)
	rank_wr_df41 = pr_success(rank_wr_df41, 0, pr_sim)
	rank_wr_df42 = pr_success(rank_wr_df42, 0, pr_sim)
	rank_wr_df43 = pr_success(rank_wr_df43, 0, pr_sim)
	rank_wr_df44 = pr_success(rank_wr_df44, 0, pr_sim)
	rank_wr_df45 = pr_success(rank_wr_df45, 0, pr_sim)
	rank_wr_df46 = pr_success(rank_wr_df46, 0, pr_sim)
	rank_wr_df47 = pr_success(rank_wr_df47, 0, pr_sim)
	rank_wr_df48 = pr_success(rank_wr_df48, 0, pr_sim)
	rank_wr_df49 = pr_success(rank_wr_df49, 0, pr_sim)
	rank_wr_df50 = pr_success(rank_wr_df50, 0, pr_sim)
	rank_wr_df51 = pr_success(rank_wr_df51, 0, pr_sim)
	rank_wr_df52 = pr_success(rank_wr_df52, 0, pr_sim)
	rank_wr_df53 = pr_success(rank_wr_df53, 0, pr_sim)
	rank_wr_df54 = pr_success(rank_wr_df54, 0, pr_sim)
	rank_wr_df55 = pr_success(rank_wr_df55, 0, pr_sim)
	rank_wr_df56 = pr_success(rank_wr_df56, 0, pr_sim)
	rank_wr_df57 = pr_success(rank_wr_df57, 0, pr_sim)
	rank_wr_df58 = pr_success(rank_wr_df58, 0, pr_sim)
	rank_wr_df59 = pr_success(rank_wr_df59, 0, pr_sim)
	rank_wr_df60 = pr_success(rank_wr_df60, 0, pr_sim)
	rank_wr_df61 = pr_success(rank_wr_df61, 0, pr_sim)
	rank_wr_df62 = pr_success(rank_wr_df62, 0, pr_sim)
	rank_wr_df63 = pr_success(rank_wr_df63, 0, pr_sim)
	rank_wr_df64 = pr_success(rank_wr_df64, 0, pr_sim)


	frames =  [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4, rank_wr_df5, rank_wr_df6, rank_wr_df7, rank_wr_df8,
			   rank_wr_df9, rank_wr_df10, rank_wr_df11, rank_wr_df12, rank_wr_df13, rank_wr_df14, rank_wr_df15, rank_wr_df16,
			   rank_wr_df17, rank_wr_df18, rank_wr_df19, rank_wr_df20, rank_wr_df21, rank_wr_df22, rank_wr_df23, rank_wr_df24,
			   rank_wr_df25, rank_wr_df26, rank_wr_df27, rank_wr_df28, rank_wr_df29, rank_wr_df30, rank_wr_df31, rank_wr_df32,
			   rank_wr_df33, rank_wr_df34, rank_wr_df35, rank_wr_df36, rank_wr_df37, rank_wr_df38, rank_wr_df39, rank_wr_df40,
			   rank_wr_df41, rank_wr_df42, rank_wr_df43, rank_wr_df44, rank_wr_df45, rank_wr_df46, rank_wr_df47, rank_wr_df48,
			   rank_wr_df49, rank_wr_df50, rank_wr_df51, rank_wr_df52, rank_wr_df53, rank_wr_df54, rank_wr_df55, rank_wr_df56,
			   rank_wr_df57, rank_wr_df58, rank_wr_df59, rank_wr_df60, rank_wr_df61, rank_wr_df62, rank_wr_df63, rank_wr_df64]

	""" Write in CSV files """

	num_rank = 1
	for df in frames:
		file_name = r'./csv/3bits_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (num_sim, K, kappa, num_rank)
		df.to_csv(file_name, encoding='utf-8', index=False)
		num_rank += 1

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3:
		print('\n**ERROR**')
		print('Required length of K and kappa: 3\n')
	else:

		# Time option
		start_t = time.perf_counter()
		
		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		sim_3bits_to_csv(K, kappa, num_sim)

		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))
