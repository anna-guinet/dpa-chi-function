#!/usr/bin/env python

"""DPA SIMULATION 3-BIT CHI ROW & 3-BIT K"""

__author__      = "Anna Guinet"
__email__       = "a.guinet@cs.ru.nl"
__version__     = "3.0"

import numpy as np
import itertools
from decimal import *
import math
import matplotlib.pylab as plt
import random
import argparse
import time
import sys

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
	
	return S

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

def fct_scalar(w, a, value):
	"""
	Function for the scalar product of any possibility for a given value of correlation.
	
	Parameter:
	w -- float
	a -- Decimal
	value -- Decimal
	
	Return:
	y -- Decimal
	"""
	# y = (value - a) * w + a
	y = (Decimal(value) - a) * Decimal(w) + a
		
	return y / Decimal(24)

def find_wr(a, b, value_b):
	"""
	Find the point when for the scalar product of the solution guess equals the scalar product of an nonsol guess.
	
	Parameter:
	a -- Decimal
	b -- Decimal
	value_b -- float
	
	Return:
	y -- Decimal
	"""	
	value = Decimal(24) - Decimal(value_b)

	# w = (b - a)/ (value + b - a)
	w = (b - a) / (value + b - a)

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

def sim_3bits(n, K, kappa, num_sim):
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
	# Signal message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	getcontext().prec = 10

	# Noise power consumption
	R = noise(n)

	# All possible signal power consumption values
	S_ref = signal_3D(mu, n)

	""" Find correlation with solution with final scalar product values """

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

	# Find indexes of all solution functions
	solution_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_pos_24 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]

	# Find indexes of all common solutions for +16
	common_pos_16_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_pos_16 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all common solutions for +8
	common_pos_8_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_pos_8 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all uncommon solutions
	uncommon_0_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_0 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all common solutions for -8
	common_neg_8_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_neg_8 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all common solutions for -16
	common_neg_16_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_neg_16 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	# Find indexes of all common solutions for -24
	common_neg_24_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in scalar_fin_neg_24 if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]


	""" Initial values of scalar product """

	# Global power consumption P_init
	P_init = [Decimal(r) for r in R]

	# Scalar product <S-ref, P>
	scalar_init = np.dot(S_ref, P_init)

	solution_init0 = scalar_init[solution_idx[0][2]][solution_idx[0][3]]

	# Degenerated case
	solution_init1 = None
	if len(solution_idx) == 2:
		solution_init1 = scalar_init[solution_idx[1][2]][solution_idx[1][3]]

	common_pos_16_init0 = scalar_init[common_pos_16_idx[0][2]][common_pos_16_idx[0][3]]
	common_pos_16_init1 = scalar_init[common_pos_16_idx[1][2]][common_pos_16_idx[1][3]]
	common_pos_16_init2 = scalar_init[common_pos_16_idx[2][2]][common_pos_16_idx[2][3]]
	common_pos_16_init3 = scalar_init[common_pos_16_idx[3][2]][common_pos_16_idx[3][3]]

	# Degenerated case
	if len(common_pos_16_idx) == 6:
		common_pos_16_init4 = scalar_init[common_pos_16_idx[4][2]][common_pos_16_idx[4][3]]
		common_pos_16_init5 = scalar_init[common_pos_16_idx[5][2]][common_pos_16_idx[5][3]]

	common_pos_8_init0 = scalar_init[common_pos_8_idx[0][2]][common_pos_8_idx[0][3]]
	common_pos_8_init1 = scalar_init[common_pos_8_idx[1][2]][common_pos_8_idx[1][3]]
	common_pos_8_init2 = scalar_init[common_pos_8_idx[2][2]][common_pos_8_idx[2][3]]
	common_pos_8_init3 = scalar_init[common_pos_8_idx[3][2]][common_pos_8_idx[3][3]]
	common_pos_8_init4 = scalar_init[common_pos_8_idx[4][2]][common_pos_8_idx[4][3]]
	common_pos_8_init5 = scalar_init[common_pos_8_idx[5][2]][common_pos_8_idx[5][3]]
	common_pos_8_init6 = scalar_init[common_pos_8_idx[6][2]][common_pos_8_idx[6][3]]
	common_pos_8_init7 = scalar_init[common_pos_8_idx[7][2]][common_pos_8_idx[7][3]]
	common_pos_8_init8 = scalar_init[common_pos_8_idx[8][2]][common_pos_8_idx[8][3]]
	common_pos_8_init9 = scalar_init[common_pos_8_idx[9][2]][common_pos_8_idx[9][3]]
	common_pos_8_init10 = scalar_init[common_pos_8_idx[10][2]][common_pos_8_idx[10][3]]
	common_pos_8_init11 = scalar_init[common_pos_8_idx[11][2]][common_pos_8_idx[11][3]]
	common_pos_8_init12 = scalar_init[common_pos_8_idx[12][2]][common_pos_8_idx[12][3]]
	common_pos_8_init13 = scalar_init[common_pos_8_idx[13][2]][common_pos_8_idx[13][3]]
	common_pos_8_init14 = scalar_init[common_pos_8_idx[14][2]][common_pos_8_idx[14][3]]
	common_pos_8_init15 = scalar_init[common_pos_8_idx[15][2]][common_pos_8_idx[15][3]]
	common_pos_8_init16 = scalar_init[common_pos_8_idx[16][2]][common_pos_8_idx[16][3]]
	common_pos_8_init17 = scalar_init[common_pos_8_idx[17][2]][common_pos_8_idx[17][3]]
	
	# Non degenerated case
	if len(common_pos_8_idx) == 19:
		common_pos_8_init18 = scalar_init[common_pos_8_idx[18][2]][common_pos_8_idx[18][3]]

	uncommon_0_init0 = scalar_init[uncommon_0_idx[0][2]][uncommon_0_idx[0][3]]
	uncommon_0_init1 = scalar_init[uncommon_0_idx[1][2]][uncommon_0_idx[1][3]]
	uncommon_0_init2 = scalar_init[uncommon_0_idx[2][2]][uncommon_0_idx[2][3]]
	uncommon_0_init3 = scalar_init[uncommon_0_idx[3][2]][uncommon_0_idx[3][3]]
	uncommon_0_init4 = scalar_init[uncommon_0_idx[4][2]][uncommon_0_idx[4][3]]
	uncommon_0_init5 = scalar_init[uncommon_0_idx[5][2]][uncommon_0_idx[5][3]]
	uncommon_0_init6 = scalar_init[uncommon_0_idx[6][2]][uncommon_0_idx[6][3]]
	uncommon_0_init7 = scalar_init[uncommon_0_idx[7][2]][uncommon_0_idx[7][3]]
	uncommon_0_init8 = scalar_init[uncommon_0_idx[8][2]][uncommon_0_idx[8][3]]
	uncommon_0_init9 = scalar_init[uncommon_0_idx[9][2]][uncommon_0_idx[9][3]]
	uncommon_0_init10 = scalar_init[uncommon_0_idx[10][2]][uncommon_0_idx[10][3]]
	uncommon_0_init11 = scalar_init[uncommon_0_idx[11][2]][uncommon_0_idx[11][3]]

	# Non degenerated case
	if len(uncommon_0_idx) == 16:
		uncommon_0_init12 = scalar_init[uncommon_0_idx[12][2]][uncommon_0_idx[12][3]]
		uncommon_0_init13 = scalar_init[uncommon_0_idx[13][2]][uncommon_0_idx[13][3]]
		uncommon_0_init14 = scalar_init[uncommon_0_idx[14][2]][uncommon_0_idx[14][3]]
		uncommon_0_init15 = scalar_init[uncommon_0_idx[15][2]][uncommon_0_idx[15][3]]

	common_neg_8_init0 = scalar_init[common_neg_8_idx[0][2]][common_neg_8_idx[0][3]]
	common_neg_8_init1 = scalar_init[common_neg_8_idx[1][2]][common_neg_8_idx[1][3]]
	common_neg_8_init2 = scalar_init[common_neg_8_idx[2][2]][common_neg_8_idx[2][3]]
	common_neg_8_init3 = scalar_init[common_neg_8_idx[3][2]][common_neg_8_idx[3][3]]
	common_neg_8_init4 = scalar_init[common_neg_8_idx[4][2]][common_neg_8_idx[4][3]]
	common_neg_8_init5 = scalar_init[common_neg_8_idx[5][2]][common_neg_8_idx[5][3]]
	common_neg_8_init6 = scalar_init[common_neg_8_idx[6][2]][common_neg_8_idx[6][3]]
	common_neg_8_init7 = scalar_init[common_neg_8_idx[7][2]][common_neg_8_idx[7][3]]
	common_neg_8_init8 = scalar_init[common_neg_8_idx[8][2]][common_neg_8_idx[8][3]]
	common_neg_8_init9 = scalar_init[common_neg_8_idx[9][2]][common_neg_8_idx[9][3]]
	common_neg_8_init10 = scalar_init[common_neg_8_idx[10][2]][common_neg_8_idx[10][3]]
	common_neg_8_init11 = scalar_init[common_neg_8_idx[11][2]][common_neg_8_idx[11][3]]
	common_neg_8_init12 = scalar_init[common_neg_8_idx[12][2]][common_neg_8_idx[12][3]]
	common_neg_8_init13 = scalar_init[common_neg_8_idx[13][2]][common_neg_8_idx[13][3]]
	common_neg_8_init14 = scalar_init[common_neg_8_idx[14][2]][common_neg_8_idx[14][3]]
	common_neg_8_init15 = scalar_init[common_neg_8_idx[15][2]][common_neg_8_idx[15][3]]
	common_neg_8_init16 = scalar_init[common_neg_8_idx[16][2]][common_neg_8_idx[16][3]]
	common_neg_8_init17 = scalar_init[common_neg_8_idx[17][2]][common_neg_8_idx[17][3]]
	
	# Non degenerated case
	if len(common_neg_8_idx) == 19:
		common_neg_8_init18 = scalar_init[common_neg_8_idx[18][2]][common_neg_8_idx[18][3]]
	
	common_neg_16_init0 = scalar_init[common_neg_16_idx[0][2]][common_neg_16_idx[0][3]]
	common_neg_16_init1 = scalar_init[common_neg_16_idx[1][2]][common_neg_16_idx[1][3]]
	common_neg_16_init2 = scalar_init[common_neg_16_idx[2][2]][common_neg_16_idx[2][3]]
	common_neg_16_init3 = scalar_init[common_neg_16_idx[3][2]][common_neg_16_idx[3][3]]

	# Degenerated case
	if len(common_neg_16_idx) == 6:
		common_neg_16_init4 = scalar_init[common_neg_16_idx[4][2]][common_neg_16_idx[4][3]]
		common_neg_16_init5 = scalar_init[common_neg_16_idx[5][2]][common_neg_16_idx[5][3]]

	common_neg_24_init0 = scalar_init[common_neg_24_idx[0][2]][common_neg_24_idx[0][3]]

	# Degenerated case
	if len(common_neg_24_idx) == 2:
		common_neg_24_init1 = scalar_init[common_neg_24_idx[1][2]][common_neg_24_idx[1][3]]


	""" Find the functions according to the Ks and kappas with solution function """

	# Display a figure with two subplots
	# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": [1.5,1]})
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5),  gridspec_kw={"width_ratios": [2,1]})

	# Make subplots close to each other and hide x ticks for all but bottom plot.
	# fig.subplots_adjust(hspace=0.1)
	# plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
	
	# List of all intersections with the solution kappa function
	wr = []

	# interval of w for abscissas
	interval = [0, 1]

	solution0 = [fct_scalar(w, solution_init0, 24) for w in interval]
	ax1.plot(interval, solution0, '-', color='tab:red', markersize=1, label='%s %s' % (solution_idx[0][0], solution_idx[0][1]))

	# Degenerate case
	if (solution_init1 != None):
		solution1 = [fct_scalar(w, solution_init1, 24) for w in interval]
		ax1.plot(interval, solution1, '-', color='tab:red', markersize=1, label='%s %s' % (solution_idx[1][0], solution_idx[1][1]))


	common_pos_16_0 = [fct_scalar(w, common_pos_16_init0, 16) for w in interval]
	common_pos_16_1 = [fct_scalar(w, common_pos_16_init1, 16) for w in interval]
	common_pos_16_2 = [fct_scalar(w, common_pos_16_init2, 16) for w in interval]
	common_pos_16_3 = [fct_scalar(w, common_pos_16_init3, 16) for w in interval]

	ax1.plot(interval, common_pos_16_0, '-', color='tab:purple', markersize=1, label='%s %s' % (common_pos_16_idx[0][0], common_pos_16_idx[0][1]))
	ax1.plot(interval, common_pos_16_1, '-', color='tab:purple', markersize=1, label='%s %s' % (common_pos_16_idx[1][0], common_pos_16_idx[1][1]))
	ax1.plot(interval, common_pos_16_2, '-', color='tab:purple', markersize=1, label='%s %s' % (common_pos_16_idx[2][0], common_pos_16_idx[2][1]))
	ax1.plot(interval, common_pos_16_3, '-', color='tab:purple', markersize=1, label='%s %s' % (common_pos_16_idx[3][0], common_pos_16_idx[3][1]))

	# Degenerated case
	if len(common_pos_16_idx) == 6:
		common_pos_16_4 = [fct_scalar(w, common_pos_16_init4, 16) for w in interval]
		common_pos_16_5 = [fct_scalar(w, common_pos_16_init5, 16) for w in interval]
		ax1.plot(interval, common_pos_16_4, '-', color='tab:purple', markersize=1, label='%s %s' % (common_pos_16_idx[4][0], common_pos_16_idx[4][1]))
		ax1.plot(interval, common_pos_16_5, '-', color='tab:purple', markersize=1, label='%s %s' % (common_pos_16_idx[5][0], common_pos_16_idx[5][1]))


	common_pos_8_0 = [fct_scalar(w, common_pos_8_init0, 8) for w in interval]
	common_pos_8_1 = [fct_scalar(w, common_pos_8_init1, 8) for w in interval]
	common_pos_8_2 = [fct_scalar(w, common_pos_8_init2, 8) for w in interval]
	common_pos_8_3 = [fct_scalar(w, common_pos_8_init3, 8) for w in interval]
	common_pos_8_4 = [fct_scalar(w, common_pos_8_init4, 8) for w in interval]
	common_pos_8_5 = [fct_scalar(w, common_pos_8_init5, 8) for w in interval]
	common_pos_8_6 = [fct_scalar(w, common_pos_8_init6, 8) for w in interval]
	common_pos_8_7 = [fct_scalar(w, common_pos_8_init7, 8) for w in interval]
	common_pos_8_8 = [fct_scalar(w, common_pos_8_init8, 8) for w in interval]
	common_pos_8_9 = [fct_scalar(w, common_pos_8_init9, 8) for w in interval]
	common_pos_8_10 = [fct_scalar(w, common_pos_8_init10, 8) for w in interval]
	common_pos_8_11 = [fct_scalar(w, common_pos_8_init11, 8) for w in interval]
	common_pos_8_12 = [fct_scalar(w, common_pos_8_init12, 8) for w in interval]
	common_pos_8_13 = [fct_scalar(w, common_pos_8_init13, 8) for w in interval]
	common_pos_8_14 = [fct_scalar(w, common_pos_8_init14, 8) for w in interval]
	common_pos_8_15 = [fct_scalar(w, common_pos_8_init15, 8) for w in interval]
	common_pos_8_16 = [fct_scalar(w, common_pos_8_init16, 8) for w in interval]
	common_pos_8_17 = [fct_scalar(w, common_pos_8_init17, 8) for w in interval]
	
	ax1.plot(interval, common_pos_8_0, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[0][0], common_pos_8_idx[0][1]))
	ax1.plot(interval, common_pos_8_1, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[1][0], common_pos_8_idx[1][1]))
	ax1.plot(interval, common_pos_8_2, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[2][0], common_pos_8_idx[2][1]))
	ax1.plot(interval, common_pos_8_3, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[3][0], common_pos_8_idx[3][1]))
	ax1.plot(interval, common_pos_8_4, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[4][0], common_pos_8_idx[4][1]))
	ax1.plot(interval, common_pos_8_5, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[5][0], common_pos_8_idx[5][1]))
	ax1.plot(interval, common_pos_8_6, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[6][0], common_pos_8_idx[6][1]))
	ax1.plot(interval, common_pos_8_7, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[7][0], common_pos_8_idx[7][1]))
	ax1.plot(interval, common_pos_8_8, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[8][0], common_pos_8_idx[8][1]))
	ax1.plot(interval, common_pos_8_9, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[9][0], common_pos_8_idx[9][1]))
	ax1.plot(interval, common_pos_8_10, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[10][0], common_pos_8_idx[10][1]))
	ax1.plot(interval, common_pos_8_11, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[11][0], common_pos_8_idx[11][1]))
	ax1.plot(interval, common_pos_8_12, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[12][0], common_pos_8_idx[12][1]))
	ax1.plot(interval, common_pos_8_13, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[13][0], common_pos_8_idx[13][1]))
	ax1.plot(interval, common_pos_8_14, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[14][0], common_pos_8_idx[14][1]))
	ax1.plot(interval, common_pos_8_15, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[15][0], common_pos_8_idx[15][1]))
	ax1.plot(interval, common_pos_8_16, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[16][0], common_pos_8_idx[16][1]))
	ax1.plot(interval, common_pos_8_17, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[17][0], common_pos_8_idx[17][1]))

	# Non degenerated case
	if len(common_pos_8_idx) == 19:
		common_pos_8_18 = [fct_scalar(w, common_pos_8_init18, 8) for w in interval]
		ax1.plot(interval, common_pos_8_18, '-', color='tab:orange', markersize=1, label='%s %s' % (common_pos_8_idx[18][0], common_pos_8_idx[18][1]))
		

	uncommon_0_0 = [fct_scalar(w, uncommon_0_init0, 0) for w in interval]
	uncommon_0_1 = [fct_scalar(w, uncommon_0_init1, 0) for w in interval]
	uncommon_0_2 = [fct_scalar(w, uncommon_0_init2, 0) for w in interval]
	uncommon_0_3 = [fct_scalar(w, uncommon_0_init3, 0) for w in interval]
	uncommon_0_4 = [fct_scalar(w, uncommon_0_init4, 0) for w in interval]
	uncommon_0_5 = [fct_scalar(w, uncommon_0_init5, 0) for w in interval]
	uncommon_0_6 = [fct_scalar(w, uncommon_0_init6, 0) for w in interval]
	uncommon_0_7 = [fct_scalar(w, uncommon_0_init7, 0) for w in interval]
	uncommon_0_8 = [fct_scalar(w, uncommon_0_init8, 0) for w in interval]
	uncommon_0_9 = [fct_scalar(w, uncommon_0_init9, 0) for w in interval]
	uncommon_0_10 = [fct_scalar(w, uncommon_0_init10, 0) for w in interval]
	uncommon_0_11 = [fct_scalar(w, uncommon_0_init11, 0) for w in interval]

	ax1.plot(interval, uncommon_0_0, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[0][0], uncommon_0_idx[0][1]))
	ax1.plot(interval, uncommon_0_1, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[1][0], uncommon_0_idx[1][1]))
	ax1.plot(interval, uncommon_0_2, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[2][0], uncommon_0_idx[2][1]))
	ax1.plot(interval, uncommon_0_3, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[3][0], uncommon_0_idx[3][1]))
	ax1.plot(interval, uncommon_0_4, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[4][0], uncommon_0_idx[4][1]))
	ax1.plot(interval, uncommon_0_5, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[5][0], uncommon_0_idx[5][1]))
	ax1.plot(interval, uncommon_0_6, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[6][0], uncommon_0_idx[6][1]))
	ax1.plot(interval, uncommon_0_7, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[7][0], uncommon_0_idx[7][1]))
	ax1.plot(interval, uncommon_0_8, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[8][0], uncommon_0_idx[8][1]))
	ax1.plot(interval, uncommon_0_9, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[9][0], uncommon_0_idx[9][1]))
	ax1.plot(interval, uncommon_0_10, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[10][0], uncommon_0_idx[10][1]))
	ax1.plot(interval, uncommon_0_11, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[11][0], uncommon_0_idx[11][1]))

	# Non degenerated case
	if len(uncommon_0_idx) == 16:
		uncommon_0_12 = [fct_scalar(w, uncommon_0_init12, 0) for w in interval]
		uncommon_0_13 = [fct_scalar(w, uncommon_0_init13, 0) for w in interval]
		uncommon_0_14 = [fct_scalar(w, uncommon_0_init14, 0) for w in interval]
		uncommon_0_15 = [fct_scalar(w, uncommon_0_init15, 0) for w in interval]
		ax1.plot(interval, uncommon_0_12, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[12][0], uncommon_0_idx[12][1]))
		ax1.plot(interval, uncommon_0_13, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[13][0], uncommon_0_idx[13][1]))
		ax1.plot(interval, uncommon_0_14, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[14][0], uncommon_0_idx[14][1]))
		ax1.plot(interval, uncommon_0_15, '-', color='tab:blue', markersize=1, label='%s %s' % (uncommon_0_idx[15][0], uncommon_0_idx[15][1]))


	common_neg_8_0 = [fct_scalar(w, common_neg_8_init0, -8) for w in interval]
	common_neg_8_1 = [fct_scalar(w, common_neg_8_init1, -8) for w in interval]
	common_neg_8_2 = [fct_scalar(w, common_neg_8_init2, -8) for w in interval]
	common_neg_8_3 = [fct_scalar(w, common_neg_8_init3, -8) for w in interval]
	common_neg_8_4 = [fct_scalar(w, common_neg_8_init4, -8) for w in interval]
	common_neg_8_5 = [fct_scalar(w, common_neg_8_init5, -8) for w in interval]
	common_neg_8_6 = [fct_scalar(w, common_neg_8_init6, -8) for w in interval]
	common_neg_8_7 = [fct_scalar(w, common_neg_8_init7, -8) for w in interval]
	common_neg_8_8 = [fct_scalar(w, common_neg_8_init8, -8) for w in interval]
	common_neg_8_9 = [fct_scalar(w, common_neg_8_init9, -8) for w in interval]
	common_neg_8_10 = [fct_scalar(w, common_neg_8_init10, -8) for w in interval]
	common_neg_8_11 = [fct_scalar(w, common_neg_8_init11, -8) for w in interval]
	common_neg_8_12 = [fct_scalar(w, common_neg_8_init12, -8) for w in interval]
	common_neg_8_13 = [fct_scalar(w, common_neg_8_init13, -8) for w in interval]
	common_neg_8_14 = [fct_scalar(w, common_neg_8_init14, -8) for w in interval]
	common_neg_8_15 = [fct_scalar(w, common_neg_8_init15, -8) for w in interval]
	common_neg_8_16 = [fct_scalar(w, common_neg_8_init16, -8) for w in interval]
	common_neg_8_17 = [fct_scalar(w, common_neg_8_init17, -8) for w in interval]
	
	ax1.plot(interval, common_neg_8_0, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[0][0], common_neg_8_idx[0][1]))
	ax1.plot(interval, common_neg_8_1, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[1][0], common_neg_8_idx[1][1]))
	ax1.plot(interval, common_neg_8_2, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[2][0], common_neg_8_idx[2][1]))
	ax1.plot(interval, common_neg_8_3, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[3][0], common_neg_8_idx[3][1]))
	ax1.plot(interval, common_neg_8_4, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[4][0], common_neg_8_idx[4][1]))
	ax1.plot(interval, common_neg_8_5, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[5][0], common_neg_8_idx[5][1]))
	ax1.plot(interval, common_neg_8_6, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[6][0], common_neg_8_idx[6][1]))
	ax1.plot(interval, common_neg_8_7, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[7][0], common_neg_8_idx[7][1]))
	ax1.plot(interval, common_neg_8_8, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[8][0], common_neg_8_idx[8][1]))
	ax1.plot(interval, common_neg_8_9, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[9][0], common_neg_8_idx[9][1]))
	ax1.plot(interval, common_neg_8_10, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[10][0], common_neg_8_idx[10][1]))
	ax1.plot(interval, common_neg_8_11, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[11][0], common_neg_8_idx[11][1]))
	ax1.plot(interval, common_neg_8_12, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[12][0], common_neg_8_idx[12][1]))
	ax1.plot(interval, common_neg_8_13, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[13][0], common_neg_8_idx[13][1]))
	ax1.plot(interval, common_neg_8_14, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[14][0], common_neg_8_idx[14][1]))
	ax1.plot(interval, common_neg_8_15, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[15][0], common_neg_8_idx[15][1]))
	ax1.plot(interval, common_neg_8_16, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[16][0], common_neg_8_idx[16][1]))
	ax1.plot(interval, common_neg_8_17, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[17][0], common_neg_8_idx[17][1]))


	# Non degenerated case
	if len(common_neg_8_idx) == 19:
		common_neg_8_18 = [fct_scalar(w, common_neg_8_init18, -8) for w in interval]
		ax1.plot(interval, common_neg_8_18, '-', color='tab:green', markersize=1, label='%s %s' % (common_neg_8_idx[18][0], common_neg_8_idx[18][1]))


	common_neg_16_0 = [fct_scalar(w, common_neg_16_init0, -16) for w in interval]
	common_neg_16_1 = [fct_scalar(w, common_neg_16_init1, -16) for w in interval]
	common_neg_16_2 = [fct_scalar(w, common_neg_16_init2, -16) for w in interval]
	common_neg_16_3 = [fct_scalar(w, common_neg_16_init3, -16) for w in interval]

	ax1.plot(interval, common_neg_16_0, '-', color='tab:brown', markersize=1, label='%s %s' % (common_neg_16_idx[0][0], common_neg_16_idx[0][1]))
	ax1.plot(interval, common_neg_16_1, '-', color='tab:brown', markersize=1, label='%s %s' % (common_neg_16_idx[1][0], common_neg_16_idx[1][1]))
	ax1.plot(interval, common_neg_16_2, '-', color='tab:brown', markersize=1, label='%s %s' % (common_neg_16_idx[2][0], common_neg_16_idx[2][1]))
	ax1.plot(interval, common_neg_16_3, '-', color='tab:brown', markersize=1, label='%s %s' % (common_neg_16_idx[3][0], common_neg_16_idx[3][1]))

	# Degenerated case
	if len(common_neg_16_idx) == 6:
		common_neg_16_4 = [fct_scalar(w, common_neg_16_init4, -16) for w in interval]
		common_neg_16_5 = [fct_scalar(w, common_neg_16_init5, -16) for w in interval]
		ax1.plot(interval, common_neg_16_4, '-', color='tab:brown', markersize=1, label='%s %s' % (common_neg_16_idx[4][0], common_neg_16_idx[4][1]))
		ax1.plot(interval, common_neg_16_5, '-', color='tab:brown', markersize=1, label='%s %s' % (common_neg_16_idx[5][0], common_neg_16_idx[5][1]))


	common_neg_24_0 = [fct_scalar(w, common_neg_24_init0, -24) for w in interval]
	ax1.plot(interval, common_neg_24_0, '-', color='tab:pink', markersize=1, label='%s %s' % (common_neg_24_idx[0][0], common_neg_24_idx[0][1]))

	# Degenerated case
	if len(common_neg_24_idx) == 2:
		common_neg_24_1 = [fct_scalar(w, common_neg_24_init1, -24) for w in interval]
		ax1.plot(interval, common_neg_24_1, '-', color='tab:pink', markersize=1, label='%s %s' % (common_neg_24_idx[1][0], common_neg_24_idx[1][1]))


	""" Find the functions according to the Ks and kappas with solution function """

	wr_common_pos_16_0 = find_wr(solution_init0, common_pos_16_init0, 16)
	wr_common_pos_16_1 = find_wr(solution_init0, common_pos_16_init1, 16)
	wr_common_pos_16_2 = find_wr(solution_init0, common_pos_16_init2, 16)
	wr_common_pos_16_3 = find_wr(solution_init0, common_pos_16_init3, 16)

	wr_common_pos_16 = [wr_common_pos_16_0, wr_common_pos_16_1, wr_common_pos_16_2, wr_common_pos_16_3]

	# Degenerated case
	if len(common_pos_16_idx) == 6:
		wr_common_pos_16_4 = find_wr(solution_init0, common_pos_16_init4, 16)
		wr_common_pos_16_5 = find_wr(solution_init0, common_pos_16_init5, 16)

		wr_common_pos_16.append(wr_common_pos_16_4)
		wr_common_pos_16.append(wr_common_pos_16_5)

	# Drop values < 0 and > 1
	wr_common_pos_16 = [w for w in wr_common_pos_16 if (w > 0) and (w < 1)]

	# Plot the intersection values
	# for w in wr_common_pos_16:
		# ax1.axvline(x=w, linestyle='--', color='tab:purple', markersize=1, label='wr_common_pos_16')

	wr_common_pos_8_0 = find_wr(solution_init0, common_pos_8_init0, 8)
	wr_common_pos_8_1 = find_wr(solution_init0, common_pos_8_init1, 8)
	wr_common_pos_8_2 = find_wr(solution_init0, common_pos_8_init2, 8)
	wr_common_pos_8_3 = find_wr(solution_init0, common_pos_8_init3, 8)
	wr_common_pos_8_4 = find_wr(solution_init0, common_pos_8_init4, 8)
	wr_common_pos_8_5 = find_wr(solution_init0, common_pos_8_init5, 8)
	wr_common_pos_8_6 = find_wr(solution_init0, common_pos_8_init6, 8)
	wr_common_pos_8_7 = find_wr(solution_init0, common_pos_8_init7, 8)
	wr_common_pos_8_8 = find_wr(solution_init0, common_pos_8_init8, 8)
	wr_common_pos_8_9 = find_wr(solution_init0, common_pos_8_init9, 8)
	wr_common_pos_8_10 = find_wr(solution_init0, common_pos_8_init10, 8)
	wr_common_pos_8_11 = find_wr(solution_init0, common_pos_8_init11, 8)
	wr_common_pos_8_12 = find_wr(solution_init0, common_pos_8_init12, 8)
	wr_common_pos_8_13 = find_wr(solution_init0, common_pos_8_init13, 8)
	wr_common_pos_8_14 = find_wr(solution_init0, common_pos_8_init14, 8)
	wr_common_pos_8_15 = find_wr(solution_init0, common_pos_8_init15, 8)
	wr_common_pos_8_16 = find_wr(solution_init0, common_pos_8_init16, 8)
	wr_common_pos_8_17 = find_wr(solution_init0, common_pos_8_init17, 8)

	wr_common_pos_8 = [wr_common_pos_8_0, wr_common_pos_8_1, wr_common_pos_8_2, wr_common_pos_8_3,
							  wr_common_pos_8_4, wr_common_pos_8_5, wr_common_pos_8_6, wr_common_pos_8_7,
							  wr_common_pos_8_8, wr_common_pos_8_9, wr_common_pos_8_10, wr_common_pos_8_11,
							  wr_common_pos_8_12, wr_common_pos_8_13, wr_common_pos_8_14, wr_common_pos_8_15,
							  wr_common_pos_8_16, wr_common_pos_8_17]
	
	# Non degenerated case
	if len(common_pos_8_idx) == 19:
		wr_common_pos_8_18 = find_wr(solution_init0, common_pos_8_init18, 8)
		wr_common_pos_8.append(wr_common_pos_8_18)
		
	# Drop values < 0 and > 1
	wr_common_pos_8 = [w for w in wr_common_pos_8 if (w > 0) and (w < 1)]

	# Plot the intersection values
	# for w in wr_common_pos_8:
		# ax1.axvline(x=w, linestyle='--', color='tab:orange', markersize=1, label='wr_common_pos_8')


	wr_uncommon_0_0 = find_wr(solution_init0, uncommon_0_init0, 0)
	wr_uncommon_0_1 = find_wr(solution_init0, uncommon_0_init1, 0)
	wr_uncommon_0_2 = find_wr(solution_init0, uncommon_0_init2, 0)
	wr_uncommon_0_3 = find_wr(solution_init0, uncommon_0_init3, 0)
	wr_uncommon_0_4 = find_wr(solution_init0, uncommon_0_init4, 0)
	wr_uncommon_0_5 = find_wr(solution_init0, uncommon_0_init5, 0)
	wr_uncommon_0_6 = find_wr(solution_init0, uncommon_0_init6, 0)
	wr_uncommon_0_7 = find_wr(solution_init0, uncommon_0_init7, 0)
	wr_uncommon_0_8 = find_wr(solution_init0, uncommon_0_init8, 0)
	wr_uncommon_0_9 = find_wr(solution_init0, uncommon_0_init9, 0)
	wr_uncommon_0_10 = find_wr(solution_init0, uncommon_0_init10, 0)
	wr_uncommon_0_11 = find_wr(solution_init0, uncommon_0_init11, 0)

	wr_uncommon_0 = [wr_uncommon_0_0, wr_uncommon_0_1, wr_uncommon_0_2, wr_uncommon_0_3,
							wr_uncommon_0_4, wr_uncommon_0_5, wr_uncommon_0_6, wr_uncommon_0_7,
							wr_uncommon_0_8, wr_uncommon_0_9, wr_uncommon_0_10, wr_uncommon_0_11]

	# Non degenerated case
	if len(uncommon_0_idx) == 16:
		wr_uncommon_0_12 = find_wr(solution_init0, uncommon_0_init12, 0)
		wr_uncommon_0_13 = find_wr(solution_init0, uncommon_0_init13, 0)
		wr_uncommon_0_14 = find_wr(solution_init0, uncommon_0_init14, 0)
		wr_uncommon_0_15 = find_wr(solution_init0, uncommon_0_init15, 0)

		wr_uncommon_0.append(wr_uncommon_0_12)
		wr_uncommon_0.append(wr_uncommon_0_13)
		wr_uncommon_0.append(wr_uncommon_0_14)
		wr_uncommon_0.append(wr_uncommon_0_15)

	# Drop values < 0 and > 1
	wr_uncommon_0 = [w for w in wr_uncommon_0 if (w > 0) and (w < 1)]

	# Plot the intersection values
	# for w in wr_uncommon_0:
		# ax1.axvline(x=w, linestyle='--', color='tab:blue', markersize=1, label='wr_uncommon_0')


	wr_common_neg_8_0 = find_wr(solution_init0, common_neg_8_init0, -8)
	wr_common_neg_8_1 = find_wr(solution_init0, common_neg_8_init1, -8)
	wr_common_neg_8_2 = find_wr(solution_init0, common_neg_8_init2, -8)
	wr_common_neg_8_3 = find_wr(solution_init0, common_neg_8_init3, -8)
	wr_common_neg_8_4 = find_wr(solution_init0, common_neg_8_init4, -8)
	wr_common_neg_8_5 = find_wr(solution_init0, common_neg_8_init5, -8)
	wr_common_neg_8_6 = find_wr(solution_init0, common_neg_8_init6, -8)
	wr_common_neg_8_7 = find_wr(solution_init0, common_neg_8_init7, -8)
	wr_common_neg_8_8 = find_wr(solution_init0, common_neg_8_init8, -8)
	wr_common_neg_8_9 = find_wr(solution_init0, common_neg_8_init9, -8)
	wr_common_neg_8_10 = find_wr(solution_init0, common_neg_8_init10, -8)
	wr_common_neg_8_11 = find_wr(solution_init0, common_neg_8_init11, -8)
	wr_common_neg_8_12 = find_wr(solution_init0, common_neg_8_init12, -8)
	wr_common_neg_8_13 = find_wr(solution_init0, common_neg_8_init13, -8)
	wr_common_neg_8_14 = find_wr(solution_init0, common_neg_8_init14, -8)
	wr_common_neg_8_15 = find_wr(solution_init0, common_neg_8_init15, -8)
	wr_common_neg_8_16 = find_wr(solution_init0, common_neg_8_init16, -8)
	wr_common_neg_8_17 = find_wr(solution_init0, common_neg_8_init17, -8)

	wr_common_neg_8 = [wr_common_neg_8_0, wr_common_neg_8_1, wr_common_neg_8_2, wr_common_neg_8_3,
							  wr_common_neg_8_4, wr_common_neg_8_5, wr_common_neg_8_6, wr_common_neg_8_7,
							  wr_common_neg_8_8, wr_common_neg_8_9, wr_common_neg_8_10, wr_common_neg_8_11,
							  wr_common_neg_8_12, wr_common_neg_8_13, wr_common_neg_8_14, wr_common_neg_8_15,
							  wr_common_neg_8_16, wr_common_neg_8_17]
	
	# Non degenerated case
	if len(common_neg_8_idx) == 19:
		wr_common_neg_8_18 = find_wr(solution_init0, common_neg_8_init18, -8)
		wr_common_neg_8.append(wr_common_neg_8_18)
		
	# Drop values < 0 and > 1
	wr_common_neg_8 = [w for w in wr_common_neg_8 if (w > 0) and (w < 1)]

	# Plot the intersection values
	# for w in wr_common_neg_8:
		# ax1.axvline(x=w, linestyle='--', color='tab:green', markersize=1, label='wr_common_neg_8')


	wr_common_neg_16_0 = find_wr(solution_init0, common_neg_16_init0, -16)
	wr_common_neg_16_1 = find_wr(solution_init0, common_neg_16_init1, -16)
	wr_common_neg_16_2 = find_wr(solution_init0, common_neg_16_init2, -16)
	wr_common_neg_16_3 = find_wr(solution_init0, common_neg_16_init3, -16)

	wr_common_neg_16 = [wr_common_neg_16_0, wr_common_neg_16_1, wr_common_neg_16_2, wr_common_neg_16_3]

	# Degenerated case
	if len(common_neg_16_idx) == 6:
		wr_common_neg_16_4 = find_wr(solution_init0, common_neg_16_init4, -16)
		wr_common_neg_16_5 = find_wr(solution_init0, common_neg_16_init5, -16)

		wr_common_neg_16.append(wr_common_neg_16_4)
		wr_common_neg_16.append(wr_common_neg_16_5)

	# Drop values < 0 and > 1
	wr_common_neg_16 = [w for w in wr_common_neg_16 if (w > 0) and (w < 1)]

	# Plot the intersection values
	# for w in wr_common_neg_16:
		# ax1.axvline(x=w, linestyle='--', color='tab:brown', markersize=1, label='wr_common_neg_16')


	wr_common_neg_24_0 = find_wr(solution_init0, common_neg_24_init0, -24)
	wr_common_neg_24 = [wr_common_neg_24_0]

	# Degenerated case
	if len(common_neg_24_idx) == 2:
		wr_common_neg_24_1 = find_wr(solution_init0, common_neg_24_init1, -24)
		wr_common_neg_24.append(wr_common_neg_24_1)

	# Drop values < 0 and > 1
	wr_common_neg_24 = [w for w in wr_common_neg_24 if (w > 0) and (w < 1)]

	# Plot the intersection values
	# for w in wr_common_neg_24:
		# ax1.axvline(x=w, linestyle='--', color='tab:pink', markersize=1, label='wr_common_neg_24')


	wr = wr_common_pos_16
	wr += wr_common_pos_8
	wr += wr_uncommon_0
	wr += wr_common_neg_8
	wr += wr_common_neg_16
	wr += wr_common_neg_24

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

	# Insert edges for plotting
	wr.insert(0, 0)
	wr.append(1)

	""" Tests loop

	# Results of scalar products
	w_list = []

	# Signal power consumption for a key
	S = signal_1D(mu, K, kappa, n)

	# Parameters for loop
	scalar_K_000_kappa_000 = []
	scalar_K_000_kappa_001 = []
	scalar_K_000_kappa_010 = []
	scalar_K_000_kappa_011 = []
	scalar_K_000_kappa_100 = []
	scalar_K_000_kappa_101 = []
	scalar_K_000_kappa_110 = []
	scalar_K_000_kappa_111 = []

	scalar_K_001_kappa_000 = []
	scalar_K_001_kappa_001 = []
	scalar_K_001_kappa_010 = []
	scalar_K_001_kappa_011 = []
	scalar_K_001_kappa_100 = []
	scalar_K_001_kappa_101 = []
	scalar_K_001_kappa_110 = []
	scalar_K_001_kappa_111 = []

	scalar_K_010_kappa_000 = []
	scalar_K_010_kappa_001 = []
	scalar_K_010_kappa_010 = []
	scalar_K_010_kappa_011 = []
	scalar_K_010_kappa_100 = []
	scalar_K_010_kappa_101 = []
	scalar_K_010_kappa_110 = []
	scalar_K_010_kappa_111 = []

	scalar_K_011_kappa_000 = []
	scalar_K_011_kappa_001 = []
	scalar_K_011_kappa_010 = []
	scalar_K_011_kappa_011 = []
	scalar_K_011_kappa_100 = []
	scalar_K_011_kappa_101 = []
	scalar_K_011_kappa_110 = []
	scalar_K_011_kappa_111 = []

	scalar_K_100_kappa_000 = []
	scalar_K_100_kappa_001 = []
	scalar_K_100_kappa_010 = []
	scalar_K_100_kappa_011 = []
	scalar_K_100_kappa_100 = []
	scalar_K_100_kappa_101 = []
	scalar_K_100_kappa_110 = []
	scalar_K_100_kappa_111 = []

	scalar_K_101_kappa_000 = []
	scalar_K_101_kappa_001 = []
	scalar_K_101_kappa_010 = []
	scalar_K_101_kappa_011 = []
	scalar_K_101_kappa_100 = []
	scalar_K_101_kappa_101 = []
	scalar_K_101_kappa_110 = []
	scalar_K_101_kappa_111 = []

	scalar_K_110_kappa_000 = []
	scalar_K_110_kappa_001 = []
	scalar_K_110_kappa_010 = []
	scalar_K_110_kappa_011 = []
	scalar_K_110_kappa_100 = []
	scalar_K_110_kappa_101 = []
	scalar_K_110_kappa_110 = []
	scalar_K_110_kappa_111 = []

	scalar_K_111_kappa_000 = []
	scalar_K_111_kappa_001 = []
	scalar_K_111_kappa_010 = []
	scalar_K_111_kappa_011 = []
	scalar_K_111_kappa_100 = []
	scalar_K_111_kappa_101 = []
	scalar_K_111_kappa_110 = []
	scalar_K_111_kappa_111 = []


	for j in range(101):

		# Weight of signal part
		w = j / 100
		
		scalar = []
			
		# Global power consumption P = (1 - w) * R + w * S
		getcontext().prec = 5
		P = [float((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S)]

		# Scalar product <S-ref, P>
		scalar = np.dot(S_ref, P)

		w_list.append(w)

		scalar_K_000_kappa_000.append(scalar[0][0])
		scalar_K_000_kappa_001.append(scalar[0][1])
		scalar_K_000_kappa_010.append(scalar[0][2])
		scalar_K_000_kappa_011.append(scalar[0][3])
		scalar_K_000_kappa_100.append(scalar[0][4])
		scalar_K_000_kappa_101.append(scalar[0][5])
		scalar_K_000_kappa_110.append(scalar[0][6])
		scalar_K_000_kappa_111.append(scalar[0][7])

		scalar_K_001_kappa_000.append(scalar[1][0])
		scalar_K_001_kappa_001.append(scalar[1][1])
		scalar_K_001_kappa_010.append(scalar[1][2])
		scalar_K_001_kappa_011.append(scalar[1][3])
		scalar_K_001_kappa_100.append(scalar[1][4])
		scalar_K_001_kappa_101.append(scalar[1][5])
		scalar_K_001_kappa_110.append(scalar[1][6])
		scalar_K_001_kappa_111.append(scalar[1][7])

		scalar_K_010_kappa_000.append(scalar[2][0])
		scalar_K_010_kappa_001.append(scalar[2][1])
		scalar_K_010_kappa_010.append(scalar[2][2])
		scalar_K_010_kappa_011.append(scalar[2][3])
		scalar_K_010_kappa_100.append(scalar[2][4])
		scalar_K_010_kappa_101.append(scalar[2][5])
		scalar_K_010_kappa_110.append(scalar[2][6])
		scalar_K_010_kappa_111.append(scalar[2][7])

		scalar_K_011_kappa_000.append(scalar[3][0])
		scalar_K_011_kappa_001.append(scalar[3][1])
		scalar_K_011_kappa_010.append(scalar[3][2])
		scalar_K_011_kappa_011.append(scalar[3][3])
		scalar_K_011_kappa_100.append(scalar[3][4])
		scalar_K_011_kappa_101.append(scalar[3][5])
		scalar_K_011_kappa_110.append(scalar[3][6])
		scalar_K_011_kappa_111.append(scalar[3][7])

		scalar_K_100_kappa_000.append(scalar[4][0])
		scalar_K_100_kappa_001.append(scalar[4][1])
		scalar_K_100_kappa_010.append(scalar[4][2])
		scalar_K_100_kappa_011.append(scalar[4][3])
		scalar_K_100_kappa_100.append(scalar[4][4])
		scalar_K_100_kappa_101.append(scalar[4][5])
		scalar_K_100_kappa_110.append(scalar[0][6])
		scalar_K_100_kappa_111.append(scalar[4][7])

		scalar_K_101_kappa_000.append(scalar[5][0])
		scalar_K_101_kappa_001.append(scalar[5][1])
		scalar_K_101_kappa_010.append(scalar[5][2])
		scalar_K_101_kappa_011.append(scalar[5][3])
		scalar_K_101_kappa_100.append(scalar[5][4])
		scalar_K_101_kappa_101.append(scalar[5][5])
		scalar_K_101_kappa_110.append(scalar[5][6])
		scalar_K_101_kappa_111.append(scalar[5][7])

		scalar_K_110_kappa_000.append(scalar[6][0])
		scalar_K_110_kappa_001.append(scalar[6][1])
		scalar_K_110_kappa_010.append(scalar[6][2])
		scalar_K_110_kappa_011.append(scalar[6][3])
		scalar_K_110_kappa_100.append(scalar[6][4])
		scalar_K_110_kappa_101.append(scalar[6][5])
		scalar_K_110_kappa_110.append(scalar[6][6])
		scalar_K_110_kappa_111.append(scalar[6][7])

		scalar_K_111_kappa_000.append(scalar[7][0])
		scalar_K_111_kappa_001.append(scalar[7][1])
		scalar_K_111_kappa_010.append(scalar[7][2])
		scalar_K_111_kappa_011.append(scalar[7][3])
		scalar_K_111_kappa_100.append(scalar[7][4])
		scalar_K_111_kappa_101.append(scalar[7][5])
		scalar_K_111_kappa_110.append(scalar[7][6])
		scalar_K_111_kappa_111.append(scalar[7][7])
	
	# ax1.plot(w_list, scalar_K_000_kappa_000, '-', color='tab:blue',  markersize=1, label='K=000 kappa=000')
	# ax1.plot(w_list, scalar_K_000_kappa_001, '-', color='tab:orange',  markersize=1, label='K=000 kappa=001')
	# ax1.plot(w_list, scalar_K_000_kappa_010, '-', color='tab:green',  markersize=1, label='K=000 kappa=010')
	# ax1.plot(w_list, scalar_K_000_kappa_011, '-', color='tab:red',  markersize=1, label='K=000 kappa=011')
	# ax1.plot(w_list, scalar_K_000_kappa_100, '-', color='tab:pink',  markersize=1, label='K=000 kappa=100')
	# ax1.plot(w_list, scalar_K_000_kappa_101, '-', color='tab:brown',  markersize=1, label='K=000 kappa=101')
	# ax1.plot(w_list, scalar_K_000_kappa_110, '-', color='tab:grey',  markersize=1, label='K=000 kappa=110')
	# ax1.plot(w_list, scalar_K_000_kappa_111, '-', color='tab:purple',  markersize=1, label='K=000 kappa=111')

	# ax1.plot(w_list, scalar_K_001_kappa_000, '-', color='tab:blue',  markersize=1, label='K=001 kappa=000')
	ax1.plot(w_list, scalar_K_001_kappa_001, ':', color='black',  markersize=1, label='K=001 kappa=001')
	# ax1.plot(w_list, scalar_K_001_kappa_010, '-', color='tab:green',  markersize=1, label='K=001 kappa=010')
	# ax1.plot(w_list, scalar_K_001_kappa_011, '-', color='tab:red',  markersize=1, label='K=001 kappa=011')
	# ax1.plot(w_list, scalar_K_001_kappa_100, '-', color='tab:pink',  markersize=1, label='K=001 kappa=100')
	# ax1.plot(w_list, scalar_K_001_kappa_101, '-', color='tab:brown',  markersize=1, label='K=001 kappa=101')
	# ax1.plot(w_list, scalar_K_001_kappa_110, '-', color='tab:grey',  markersize=1, label='K=001 kappa=110')
	# ax1.plot(w_list, scalar_K_001_kappa_111, '-', color='tab:purple',  markersize=1, label='K=001 kappa=111')

	# ax1.plot(w_list, scalar_K_010_kappa_000, '-', color='tab:blue',  markersize=1, label='K=010 kappa=000')
	# ax1.plot(w_list, scalar_K_010_kappa_001, '-', color='tab:orange',  markersize=1, label='K=010 kappa=001')
	# ax1.plot(w_list, scalar_K_010_kappa_010, '-', color='tab:green',  markersize=1, label='K=010 kappa=010')
	# ax1.plot(w_list, scalar_K_010_kappa_011, '-', color='tab:red',  markersize=1, label='K=010 kappa=011')
	# ax1.plot(w_list, scalar_K_010_kappa_100, '-', color='tab:pink',  markersize=1, label='K=010 kappa=100')
	# ax1.plot(w_list, scalar_K_010_kappa_101, '-', color='tab:brown',  markersize=1, label='K=010 kappa=101')
	# ax1.plot(w_list, scalar_K_010_kappa_110, '-', color='tab:grey',  markersize=1, label='K=010 kappa=110')
	# ax1.plot(w_list, scalar_K_010_kappa_111, '-', color='tab:purple',  markersize=1, label='K=010 kappa=111')

	# ax1.plot(w_list, scalar_K_011_kappa_000, '-', color='tab:blue',  markersize=1, label='K=011 kappa=000')
	# ax1.plot(w_list, scalar_K_011_kappa_001, '-', color='tab:orange',  markersize=1, label='K=011 kappa=001')
	# ax1.plot(w_list, scalar_K_011_kappa_010, '-', color='tab:green',  markersize=1, label='K=011 kappa=010')
	ax1.plot(w_list, scalar_K_011_kappa_011, '-.', color='black',  markersize=1, label='K=011 kappa=011')
	# ax1.plot(w_list, scalar_K_011_kappa_100, '-', color='tab:pink',  markersize=1, label='K=011 kappa=100')
	# ax1.plot(w_list, scalar_K_011_kappa_101, '-', color='tab:brown',  markersize=1, label='K=011 kappa=101')
	# ax1.plot(w_list, scalar_K_011_kappa_110, '-', color='tab:grey',  markersize=1, label='K=011 kappa=110')
	# ax1.plot(w_list, scalar_K_011_kappa_111, '-', color='tab:purple',  markersize=1, label='K=011 kappa=111')

	# ax1.plot(w_list, scalar_K_100_kappa_000, '-', color='tab:blue',  markersize=1, label='K=100 kappa=000')
	# ax1.plot(w_list, scalar_K_100_kappa_001, '-', color='tab:orange',  markersize=1, label='K=100 kappa=001')
	# ax1.plot(w_list, scalar_K_100_kappa_010, '-', color='tab:green',  markersize=1, label='K=100 kappa=010')
	# ax1.plot(w_list, scalar_K_100_kappa_011, '-', color='tab:red',  markersize=1, label='K=100 kappa=011')
	# ax1.plot(w_list, scalar_K_100_kappa_100, '-', color='tab:pink',  markersize=1, label='K=100 kappa=100')
	# ax1.plot(w_list, scalar_K_100_kappa_101, '-', color='tab:brown',  markersize=1, label='K=100 kappa=101')
	# ax1.plot(w_list, scalar_K_100_kappa_110, '-', color='tab:grey',  markersize=1, label='K=100 kappa=110')
	# ax1.plot(w_list, scalar_K_100_kappa_111, '-', color='tab:purple',  markersize=1, label='K=100 kappa=111')

	# ax1.plot(w_list, scalar_K_101_kappa_000, '-', color='tab:blue',  markersize=1, label='K=101 kappa=000')
	# ax1.plot(w_list, scalar_K_101_kappa_001, '-', color='tab:orange',  markersize=1, label='K=101 kappa=001')
	# ax1.plot(w_list, scalar_K_101_kappa_010, '-', color='tab:green',  markersize=1, label='K=101 kappa=010')
	# ax1.plot(w_list, scalar_K_101_kappa_011, '-', color='tab:red',  markersize=1, label='K=101 kappa=011')
	# ax1.plot(w_list, scalar_K_101_kappa_100, '-', color='tab:pink',  markersize=1, label='K=101 kappa=100')
	# ax1.plot(w_list, scalar_K_101_kappa_101, '-', color='tab:brown',  markersize=1, label='K=101 kappa=101')
	# ax1.plot(w_list, scalar_K_101_kappa_110, '-', color='tab:grey',  markersize=1, label='K=101 kappa=110')
	# ax1.plot(w_list, scalar_K_101_kappa_111, '-', color='tab:purple',  markersize=1, label='K=101 kappa=111')

	# ax1.plot(w_list, scalar_K_110_kappa_000, '-', color='tab:blue',  markersize=1, label='K=110 kappa=000')
	ax1.plot(w_list, scalar_K_110_kappa_001, '--', color='black',  markersize=1, label='K=110 kappa=001')
	# ax1.plot(w_list, scalar_K_110_kappa_010, '-', color='tab:green',  markersize=1, label='K=110 kappa=010')
	# ax1.plot(w_list, scalar_K_110_kappa_011, '-', color='tab:red',  markersize=1, label='K=110 kappa=011')
	# ax1.plot(w_list, scalar_K_110_kappa_100, '-', color='tab:pink',  markersize=1, label='K=110 kappa=100')
	# ax1.plot(w_list, scalar_K_110_kappa_101, '-', color='tab:brown',  markersize=1, label='K=110 kappa=101')
	# ax1.plot(w_list, scalar_K_110_kappa_110, '--', color='black',  markersize=1, label='K=110 kappa=110')
	# ax1.plot(w_list, scalar_K_110_kappa_111, '-', color='tab:purple',  markersize=1, label='K=110 kappa=111')

	# ax1.plot(w_list, scalar_K_111_kappa_000, '-', color='tab:blue',  markersize=1, label='K=111 kappa=000')
	# ax1.plot(w_list, scalar_K_111_kappa_001, '-', color='tab:orange',  markersize=1, label='K=111 kappa=001')
	# ax1.plot(w_list, scalar_K_111_kappa_010, '-', color='tab:green',  markersize=1, label='K=111 kappa=010')
	# ax1.plot(w_list, scalar_K_111_kappa_011, '-', color='tab:red',  markersize=1, label='K=111 kappa=011')
	# ax1.plot(w_list, scalar_K_111_kappa_100, '-', color='tab:pink',  markersize=1, label='K=111 kappa=100')
	# ax1.plot(w_list, scalar_K_111_kappa_101, '-', color='tab:brown',  markersize=1, label='K=111 kappa=101')
	# ax1.plot(w_list, scalar_K_111_kappa_110, '-', color='tab:grey',  markersize=1, label='K=111 kappa=110')
	# ax1.plot(w_list, scalar_K_111_kappa_111, '-', color='tab:purple',  markersize=1, label='K=111 kappa=111')

	"""

	ax2.plot(wr, rank, color='tab:red')
	
	# ax1.legend(loc='upper right')
	ax1.set_title(r'Three-bit strategy for K=%s & kappa=%s' %(K, kappa))
	ax1.set_ylabel('Scalar product <S_ref,P>')
	ax1.set_xlabel('Weight w')
	ax2.set_ylabel('Rank (K, kappa) solution', rotation=-90, labelpad=12)
	ax2.set_xlabel('Weight w')
	ax2.set_ylim([65, -5])
	ax2.yaxis.set_label_position("right")
	ax2.yaxis.tick_right()
	
	fig.tight_layout()
	# plt.show()
	plt.savefig('./plot/sim_3bits_K=%s_kappa=%s_#%s.png' % (K, kappa, num_sim))
	plt.close(fig)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3:
		print('\n**ERROR**')
		print('Required length of K and kappa: 3')

	else:
		# Length of signal part
		n = 3

		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		# Time option
		start_t = time.perf_counter()
		
		for j in range(num_sim):
			sim_3bits(n, K, kappa, j)

		# Time option
		end_t = time.perf_counter()
		print('\ntime', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))


