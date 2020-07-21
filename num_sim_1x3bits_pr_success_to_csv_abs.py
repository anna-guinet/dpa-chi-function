#!/usr/bin/env python

"""SUCCESS PROBA DPA 3-BIT CHI ROW & 1-BIT K """

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

def gen_key(i, n=3):
	""" 
	Create key values where key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	Eg: for i = 1, key = [[ 0, 0, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ], [ 1, 0, 1 ]]

	Note: We do not take into account key_i as it adds only symmetry the signal power consumption 
	(and the output of the scalar product by extension). In fact, in the algorithm, 
	we set the i-th bit as 0. 

	Parameters:
	i -- integer in [0,n-1]
	n -- integer (default 3)

	Return:
	key -- list of tuples of length 4
	"""
	# All possible values for n-1 bits
	key = list(itertools.product([bool(0), bool(1)], repeat=(n-1)))

	# Transform nested tuples into nested lists
	key = [list(j) for j in key]

	# Insert 'False' as the i-th position
	key_extend = [j.insert(i % n, bool(0)) for j in key]
	
	return key
	
def signal_2D(mu, i, n=3):
	"""
	Generate signal power consumption S values for a 3-bit kappa and a 1-bit K, for all messages mu.

	'key' is a value such as key = kappa, except for bit i: key_i = kappa_i XOR K

	Parameters:
	mu -- 2D list of bool, size n x 2^n
	i -- integer in [0,n-1]
	n -- integer (default 3)

	Return:
	S -- 2D list of integers, size 2^n x 4
	"""

	# All possible values for the secret key
	key = gen_key(i, n)

	S = []

	# For each key
	for l in range(2**(n-1)):

		S_l = []

		# For each message mu
		for j in range(2**n):

			# Activity of the first storage cell of the register
			d = key[l][i] ^ mu[j][i] ^ (key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n])

			# Power consumption model
			S_lj = (-1)**(d)

			S_l.append(S_lj)

		S += [S_l]

	return S
	
def gen_S_ref(mu, n=3):
	"""
	Generate all possible signal power consumption for every bit i of first chi's row. 

	Parameters:
	mu -- 2D list of bool, size n x 2^n
	n -- integer (default 3)

	Return:
	S_ref -- 3D list of integers, size 2^n x 2^(n-1) x n
	"""
	S_ref = []

	for i in range(n):
		S_ref_i = signal_2D(mu, i, n)
		S_ref += [S_ref_i]

	return S_ref

def noise(n=3):	
	"""
	Compute a noise power consumption R vector for each possible message mu. 

	Parameters:
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

	# Absolute value
	df['wr'] = df['wr'].abs()

	return df

def find_abs_w0(a0, a1, a2):
	"""
	Find the fiber points for each function on 1 bit of register

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal

	Return:
	list_tup_w0 -- list of Decimal
	"""	

	list_a = [a0, a1, a2]
	list_tup_w0 = []

	for a in list_a:
		if (a != 8):
			w0 = a / (a - Decimal(8))

			if (w0 > 0) & (w0 < 1):
				list_tup_w0.append((w0, a))
			else:
				list_tup_w0.append((0, a)) # by default 0 for the loop afterwards
		else:
			list_tup_w0.append((0, a)) # by default 0 for the loop afterwards
	
	return list_tup_w0

def reorder_common_init(solution_init, common_init):
	"""
	Reorder common_init and common_idx list according to the sorted solution_init.
	Common kappa's scalar product values changed order because of the sorting on w0 values. 

	Parameters:
	solution_init -- list of Decimal
	common_init -- list of lists

	Return:
	common_init_new -- list of Decimal
	"""	
	common_init_new = []

	for i in range(3):
		for sublist in common_init:
			if solution_init[i] in sublist:
				sublist_new = []

				# Add non-common initial scalar product values in new sublist
				sublist_new = [elem for elem in sublist if elem != solution_init[i]]

				# If ever non-common initial scalar product == common initial scalar product
				if len(sublist_new) == 1:
					sublist_new.append(solution_init[i])
				if len(sublist_new) == 0:
					sublist_new.append(solution_init[i])
					sublist_new.append(solution_init[i])

				# Insert common initial scalar product value at the same index than sorted solution_init list
				sublist_new.insert(i, solution_init[i])

				common_init_new += [sublist_new]

	return common_init_new

def find_wr1_nonsol(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the solution equals the one of an nonsolution.

	Fct_nonsol(w) = fct_solution(w), for which w in [0, w0_0] with 0 < w0_0 < w0_1 < w0_2 < 1 ?

	fct_solution(w) = (alpha1 - 24) w - alpha1
	fct_nonsol(w) = -beta w + beta

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal
	d -- Decimal

	Return:
	w1 -- Decimal
	"""
	b = abs(b)
	c = abs(c)
	d = abs(d)

	w1 = Decimal()

	alpha1 = a0 + a1 + a2
	beta = b + c + d

	if (alpha1 + beta - Decimal(24)) != 0:
		w1 = (alpha1 + beta) / (alpha1 + beta - Decimal(24))
	else:
		w1 = None

	return w1

def find_wr2_nonsol(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the solution equals the one of an nonsolution.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_0, w0_1] with 0 =< w0_0 < w0_1 < w0_2 < 1 ?

	fct_solution(w) = (alpha2 - 8) w - alpha2
	fct_nonsol(w) = -beta w + beta

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal
	d -- Decimal

	Return:
	w2 -- Decimal
	"""
	b = abs(b)
	c = abs(c)
	d = abs(d)
	
	w2 = Decimal()

	alpha2 = - a0 + a1 + a2
	beta = b + c + d
	
	if (alpha2 + beta - Decimal(8)) != 0:
		w2 = (alpha2 + beta) / (alpha2 + beta - Decimal(8))
	else:
		w2 = None

	return w2

def find_wr3_nonsol(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the solution equals the one of an nonsolution.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_1, w0_2] with 0 =< w0_1 < w0_2 < 1 ?

	fct_solution(w) = (8 - alpha3) w + alpha3
	fct_nonsol(w) = -beta w + beta

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal
	d -- Decimal

	Return:
	w3 -- Decimal
	"""
	b = abs(b)
	c = abs(c)
	d = abs(d)

	w3 = Decimal()

	alpha3 = a0 + a1 - a2
	beta = b + c + d

	if (alpha3 - beta - Decimal(8)) != 0:
		w3 = (alpha3 - beta) / (alpha3 - beta - Decimal(8))
	else:
		w3 = None

	return w3

def find_wr4_nonsol(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the solution equals the one of an nonsolution.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_2, 1] with 0 =< w0_2 < 1?

	fct_solution(w) = (24 - alpha1) w + alpha1
	fct_nonsol(w) = -beta w + beta

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal
	d -- Decimal

	Return:
	w4 -- Decimal
	"""
	b = abs(b)
	c = abs(c)
	d = abs(d)
	
	w4 = Decimal()

	alpha1 = a0 + a1 + a2
	beta = b + c + d
	
	if (alpha1 - beta - Decimal(24)) != 0:
		w4 = (alpha1 - beta) / (alpha1 - beta - Decimal(24))
	else:
		w4 = None

	return w4
	
def find_wr1_correlated(a0, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the solution equals 
	the scalar product of an correlated hypothesis on a0.

	Fct_correlated(w) = fct_solution(w), for which w in [0, w0_0] with 0 < w0_0 < w0_1 < w0_2 < 1 ?

	fct_solution(w) = (alpha1 - 24) w - alpha1
	fct_correlated(w) = (beta1 - 8) w - beta1

	/ ! \ Also valid for correlation on a1 and a2

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal

	Return:
	w1 -- Decimal
	"""	
	b = abs(b)
	c = abs(c)

	alpha1 = a0 + a1 + a2
	beta1 = a0 - b - c

	w1 = Decimal()
	
	if (alpha1 - beta1 - Decimal(16)) != 0:
		w1 = (alpha1 - beta1) / (alpha1 - beta1 - Decimal(16))
	else:
		w1 = None

	return w1

def find_wr2_correlated(a0, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the solution equals 
	the scalar product of an correlated hypothesis on a0.

	Fct_correlated(w) = fct_solution(w), for which w in [w0_0, w0_1] with 0 =< w0_0 < w0_1 < w0_2 < 1 ?

	fct_solution(w) = (alpha2 - 8) w - alpha2
	fct_correlated(w) = (8 - beta2) w + beta2 for correlation on a0

	/ ! \ Only valid for correlation on a0 (and not a1 or a1)

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal

	Return:
	w2 -- Decimal
	"""	
	b = abs(b)
	c = abs(c)

	alpha2 = - a0 + a1 + a2
	beta2 = a0 + b + c

	w2 = Decimal()
	
	if (alpha2 + beta2 - Decimal(16)) != 0:
		w2 = (alpha2 + beta2) / (alpha2 + beta2 - Decimal(16))
	else:
		w2 = None

	return w2

def find_wr3_correlated(a2, a0, a1, b, c):
	"""
	Find the point when for the scalar product of the solution equals 
	the scalar product of an correlated hypothesis on a0.

	Fct_correlated(w) = fct_solution(w), for which w in [w0_1, w0_2] with 0 =< w0_1 < w0_2 < 1 ?

	fct_solution(w) = (8 - alpha3) w + alpha3
	fct_correlated(w) = (beta1 - 8) w - beta1 for correlation on a2

	/ ! \ Only valid for correlation on a2 (and not a0 or a1)

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal

	Return:
	w3 -- Decimal
	"""	
	b = abs(b)
	c = abs(c)

	alpha3 = a0 + a1 - a2
	beta1 = a2 - b - c

	w3 = Decimal()
	
	if (alpha3 + beta1 - Decimal(16)) != 0:
		w3 = (alpha3 + beta1) / (alpha3 + beta1 - Decimal(16))
	else:
		w3 = None

	return w3

def find_wr4_correlated(a0, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the solution equals 
	the scalar product of an correlated hypothesis on a0.

	Fct_correlated(w) = fct_solution(w), for which w in [w0_2, 1] with 0 =< w0_2 < 1 ?

	fct_solution(w) = (24 - alpha1) w + alpha1
	fct_correlated(w) = (8 - beta2) w + beta2

	/ ! \ Also valid for correlation on a1 and on a2

	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal

	Return:
	w4 -- Decimal
	"""	
	b = abs(b)
	c = abs(c)

	alpha1 = a0 + a1 + a2
	beta2 = a0 + b + c

	w4 = Decimal()
	
	if (alpha1 - beta2 - Decimal(16)) != 0:
		w4 = (alpha1 - beta2) / (alpha1 - beta2 - Decimal(16))
	else:
		w4 = None

	return w4

def find_rank(wr):
	"""
	Return the list of ranks for the solution kappa.
	
	Parameter:
	wr -- list of Decimal
	
	Return:
	rank -- list of integer
	"""
	# List of ranks
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if wr:

		# Count number of rank increment ('wr4' or 'wr3') and rank decrement ('wr2' or 'wr1')
		counter_1 = sum(1 for tuple in wr if tuple[1] == ('wr4') or tuple[1] == ('wr3'))
		counter_2 = sum(-1 for tuple in wr if tuple[1] == ('wr2') or tuple[1] == ('wr1'))
		num_wr = counter_1 + counter_2
		
		rank_init = 1 + num_wr
		
		rank.append(rank_init)
		
		for tuple in wr:
		
			# If 'wr4'or 'wr3', rank increases
			if tuple[1] == ('wr4') or tuple[1] == ('wr3'):
				rank.append(rank[-1] - 1)
				
			# If 'wr2' or 'wr1', rank decreases
			if tuple[1] == ('wr2') or tuple[1] == ('wr1'):
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, wr

def xor_secret(K, kappa, n=3):
	""" 
	Compute power consumption at bit i = kappa_i XOR K_i
	
	Parameters:
	kappa -- string
	K -- string
	n -- integer
	
	Return:
	p -- list of float, size n
	"""

	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = list(K)
	K = [bool(int(j)) for j in K]
	
	# Initialize new kappa values
	kappa_copy = kappa.copy()
	K_copy = K.copy()
	
	p = []
	
	for i in range(n):
		# XOR at indice i
		d = K_copy[i] ^ kappa_copy[i]
		
		# Power consumption at i-th bit
		p_i = (-1)**d
		
		p.append(p_i)

	return p

def sim_1x3bits(K, kappa, num_sim):
	"""
	Compute simulation scalar products and return a list of ranks and intersection values with solution scalar product
	for several simulations. 
	
	Parameter:
	num_sim -- integer
	
	Return:
	rank_wr_list -- list of tuples
	"""

	# Length of signal part
	n = 3

	# Signal message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	getcontext().prec = 10

	# All possible signal power consumption values
	S_ref = gen_S_ref(mu, n)
	
	# Save the results of the experiments
	rank_wr_list = []
	
	""" Generate many simulations """
	for j in range(num_sim):
	
		# Noise power consumption
		R = noise(n)

		""" Initial values of scalar product """
		scalar_init = []
			
		# i-th bit of register state studied
		for i in range(n):
				
			# Global power consumption P_init = R
			P_init_i = [Decimal(r) for r in R]

			# Scalar product <S-ref, P>
			scalar_init_i = np.dot(S_ref[i], P_init_i)
			scalar_init += [scalar_init_i]
			
		# Scalar product for i = 0
		# s0 : ['_00', '_01', '_10', '_11']
		s0 = scalar_init[0].tolist()
			
		init_x00 = s0[0]
		init_x01 = s0[1]
		init_x10 = s0[2]
		init_x11 = s0[3]
			
		# Scalar product for i = 1
		# s1 : ['0_0', '0_1', '1_0', '1_1']		
		s1 = scalar_init[1].tolist()
			
		init_0x0 = s1[0]
		init_0x1 = s1[1]
		init_1x0 = s1[2]
		init_1x1 = s1[3]

		# Scalar product for i = 2
		# s2 : ['00_', '01_', '10_', '11_']
		s2 = scalar_init[2].tolist()
			
		init_00x = s2[0]
		init_01x = s2[1]
		init_10x = s2[2]
		init_11x = s2[3]
			
		init_scalar = [[init_x00, init_x01, init_x10, init_x11],
					   [init_0x0, init_0x1, init_1x0, init_1x1],
					   [init_00x, init_01x, init_10x, init_11x]]

		""" To find functions parameters """
		list_kappa_idx = [['000', 0, 0, 0],
						  ['001', 1, 1, 0],
						  ['010', 2, 0, 1],
						  ['011', 3, 1, 1],
						  ['100', 0, 2, 2],
						  ['101', 1, 3, 2],
						  ['110', 2, 2, 3],
						  ['111', 3, 3, 3]]

		solution_idx = [sublist for sublist in list_kappa_idx if sublist[0] == kappa]
		solution_idx = list(itertools.chain.from_iterable(solution_idx)) # Un-nest the previous list
			
		common0_idx = [sublist for sublist in list_kappa_idx if (sublist[1] == solution_idx[1]) and (sublist[0] != kappa)]
		common0_idx = list(itertools.chain.from_iterable(common0_idx)) # Un-nest the previous list

		common1_idx = [sublist for sublist in list_kappa_idx if (sublist[2] == solution_idx[2]) and (sublist[0] != kappa)]
		common1_idx = list(itertools.chain.from_iterable(common1_idx)) # Un-nest the previous list
			
		common2_idx = [sublist for sublist in list_kappa_idx if (sublist[3] == solution_idx[3]) and (sublist[0] != kappa)]
		common2_idx = list(itertools.chain.from_iterable(common2_idx)) # Un-nest the previous list

		nonsol_idx = [sublist for sublist in list_kappa_idx if (sublist[1] != solution_idx[1]) and (sublist[2] != solution_idx[2]) and (sublist[3] != solution_idx[3])]
		nonsol0_idx = nonsol_idx[0]
		nonsol1_idx = nonsol_idx[1]
		nonsol2_idx = nonsol_idx[2]
		nonsol3_idx = nonsol_idx[3]

		""" To find initial scalar product values """
		
		solution_init = [init_scalar[i][solution_idx[i+1]] for i in range(3)]

		common0_init = [init_scalar[i][common0_idx[i+1]] for i in range(3)]
		common1_init = [init_scalar[i][common1_idx[i+1]] for i in range(3)]
		common2_init = [init_scalar[i][common2_idx[i+1]] for i in range(3)]
			
		nonsol0_init = [init_scalar[i][nonsol0_idx[i+1]] for i in range(3)]
		nonsol1_init = [init_scalar[i][nonsol1_idx[i+1]] for i in range(3)]
		nonsol2_init = [init_scalar[i][nonsol2_idx[i+1]] for i in range(3)]
		nonsol3_init = [init_scalar[i][nonsol3_idx[i+1]] for i in range(3)]
		
		# Determine the sign of initial solution value depending on the power concumption of the register at bit i
		p_secret = xor_secret(K, kappa, n)
		solution_init = [c * p for c, p in zip(solution_init, p_secret)]
		common0_init[0] = common0_init[0] * p_secret[0]
		common1_init[1] = common1_init[1] * p_secret[1]
		common2_init[2] = common2_init[2] * p_secret[2]

		# Find w0 for each bit of the register
		# tup = (w0, a)
		list_tup_w0 = find_abs_w0(solution_init[0], solution_init[1], solution_init[2])

		# Sort result by increasing order on w0
		list_tup_w0 = sorted(list_tup_w0, key=lambda x: x[0])

		# Sorted list of solution_init by increasing order on w0
		solution_init = [tup[1] for tup in list_tup_w0]		

		# Sorted list of (common_init, common_idx) according to solution_init
		common_init = [common0_init, common1_init, common2_init]
		common_init = reorder_common_init(solution_init, common_init)
		common0_init = common_init[0]
		common1_init = common_init[1]
		common2_init = common_init[2]

		# Sorted list of w0 by increasing order
		list_w0 = [tup[0] for tup in list_tup_w0]


		""" Find the intersections with solution kappa function """

		# List of all intersections with the solution kappa function
		wr = []

		interval = [list_w0[2], 1]

		# intersections between solution function and nonsol functions	
		wr4_nonsol0 = find_wr4_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol0_init[0], nonsol0_init[1], nonsol0_init[2])
		wr4_nonsol1 = find_wr4_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol1_init[0], nonsol1_init[1], nonsol1_init[2])
		wr4_nonsol2 = find_wr4_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol2_init[0], nonsol2_init[1], nonsol2_init[2])
		wr4_nonsol3 = find_wr4_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol3_init[0], nonsol3_init[1], nonsol3_init[2])

		if (wr4_nonsol0 != None) and (wr4_nonsol0 > interval[0]) and (wr4_nonsol0 < interval[1]):
			wr.append((wr4_nonsol0, 'wr4'))

		if (wr4_nonsol1 != None) and (wr4_nonsol1 > interval[0]) and (wr4_nonsol1 < interval[1]):
			wr.append((wr4_nonsol1, 'wr4'))

		if (wr4_nonsol2 != None) and (wr4_nonsol2 > interval[0]) and (wr4_nonsol2 < interval[1]):
			wr.append((wr4_nonsol2, 'wr4'))

		if (wr4_nonsol3 != None) and (wr4_nonsol3 > interval[0]) and (wr4_nonsol3 < interval[1]):
			wr.append((wr4_nonsol3, 'wr4'))

		# intersections between solution function and correlated functions	
		wr4_common0 = find_wr4_correlated(common0_init[0], solution_init[1], solution_init[2], common0_init[1], common0_init[2])
		wr4_common1 = find_wr4_correlated(common1_init[1], solution_init[2], solution_init[0], common1_init[2], common1_init[0])
		wr4_common2 = find_wr4_correlated(common2_init[2], solution_init[0], solution_init[1], common2_init[0], common2_init[1])

		if (wr4_common0 != None) and (wr4_common0 > interval[0]) and (wr4_common0 < interval[1]):
			wr.append((wr4_common0, 'wr4'))

		if (wr4_common1 != None) and (wr4_common1 > interval[0]) and (wr4_common1 < interval[1]):
			wr.append((wr4_common1, 'wr4'))

		if (wr4_common2 != None) and (wr4_common2 > interval[0]) and (wr4_common2 < interval[1]):
			wr.append((wr4_common2, 'wr4'))

		# ------------------------------------------------------------------------------------------- #

		if list_w0[2] != 0:

			interval = [list_w0[1], list_w0[2]]

			# intersections between solution function and nonsol functions	
			wr3_nonsol0 = find_wr3_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol0_init[0], nonsol0_init[1], nonsol0_init[2])
			wr3_nonsol1 = find_wr3_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol1_init[0], nonsol1_init[1], nonsol1_init[2])
			wr3_nonsol2 = find_wr3_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol2_init[0], nonsol2_init[1], nonsol2_init[2])
			wr3_nonsol3 = find_wr3_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol3_init[0], nonsol3_init[1], nonsol3_init[2])

			if (wr3_nonsol0 != None) and (wr3_nonsol0 > interval[0]) and (wr3_nonsol0 < interval[1]):
				wr.append((wr3_nonsol0, 'wr3'))

			if (wr3_nonsol1 != None) and (wr3_nonsol1 > interval[0]) and (wr3_nonsol1 < interval[1]):
				wr.append((wr3_nonsol1, 'wr3'))

			if (wr3_nonsol2 != None) and (wr3_nonsol2 > interval[0]) and (wr3_nonsol2 < interval[1]):
				wr.append((wr3_nonsol2, 'wr3'))

			if (wr3_nonsol3 != None) and (wr3_nonsol3 > interval[0]) and (wr3_nonsol3 < interval[1]):
				wr.append((wr3_nonsol3, 'wr3'))

			# intersection between solution function and correlated function on a2
			wr3_common2 = find_wr3_correlated(common2_init[2], solution_init[0], solution_init[1], common2_init[0], common2_init[1])

			if (wr3_common2 != None) and (wr3_common2 > interval[0]) and (wr3_common2 < interval[1]):
				wr.append((wr3_common2, 'wr3'))

			# ------------------------------------------------------------------------------------------- #

			if list_w0[1] != 0:
			
				interval = [list_w0[0], list_w0[1]]

				# intersections between solution function and nonsol functions	
				wr2_nonsol0 = find_wr2_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol0_init[0], nonsol0_init[1], nonsol0_init[2])
				wr2_nonsol1 = find_wr2_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol1_init[0], nonsol1_init[1], nonsol1_init[2])
				wr2_nonsol2 = find_wr2_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol2_init[0], nonsol2_init[1], nonsol2_init[2])
				wr2_nonsol3 = find_wr2_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol3_init[0], nonsol3_init[1], nonsol3_init[2])

				if (wr2_nonsol0 != None) and (wr2_nonsol0 > interval[0]) and (wr2_nonsol0 < interval[1]):
					wr.append((wr2_nonsol0, 'wr2'))

				if (wr2_nonsol1 != None) and (wr2_nonsol1 > interval[0]) and (wr2_nonsol1 < interval[1]):
					wr.append((wr2_nonsol1, 'wr2'))

				if (wr2_nonsol2 != None) and (wr2_nonsol2 > interval[0]) and (wr2_nonsol2 < interval[1]):
					wr.append((wr2_nonsol2, 'wr2'))

				if (wr2_nonsol3 != None) and (wr2_nonsol3 > interval[0]) and (wr2_nonsol3 < interval[1]):
					wr.append((wr2_nonsol3, 'wr2'))

				# intersection between solution function and correlated function on a0
				wr2_common0 = find_wr2_correlated(common0_init[0], solution_init[1], solution_init[2], common0_init[1], common0_init[2])

				if (wr2_common0 != None) and (wr2_common0 > interval[0]) and (wr2_common0 < interval[1]):
					wr.append((wr2_common0, 'wr2'))

				# ------------------------------------------------------------------------------------------- #

				if list_w0[0] != 0:
			
					interval = [0, list_w0[0]]

					# intersections between solution function and nonsol functions
					wr1_nonsol0 = find_wr1_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol0_init[0], nonsol0_init[1], nonsol0_init[2])
					wr1_nonsol1 = find_wr1_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol1_init[0], nonsol1_init[1], nonsol1_init[2])
					wr1_nonsol2 = find_wr1_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol2_init[0], nonsol2_init[1], nonsol2_init[2])
					wr1_nonsol3 = find_wr1_nonsol(solution_init[0], solution_init[1], solution_init[2], nonsol3_init[0], nonsol3_init[1], nonsol3_init[2])

					if (wr1_nonsol0 != None) and (wr1_nonsol0 > interval[0]) and (wr1_nonsol0 < interval[1]):
						wr.append((wr1_nonsol0, 'wr1'))

					if (wr1_nonsol1 != None) and (wr1_nonsol1 > interval[0]) and (wr1_nonsol1 < interval[1]):
						wr.append((wr1_nonsol1, 'wr1'))

					if (wr1_nonsol2 != None) and (wr1_nonsol2 > interval[0]) and (wr1_nonsol2 < interval[1]):
						wr.append((wr1_nonsol2, 'wr1'))

					if (wr1_nonsol3 != None) and (wr1_nonsol3 > interval[0]) and (wr1_nonsol3 < interval[1]):
						wr.append((wr1_nonsol3, 'wr1'))

					# intersections between solution function and correlated functions	
					wr1_common0 = find_wr1_correlated(common0_init[0], solution_init[1], solution_init[2], common0_init[1], common0_init[2])
					wr1_common1 = find_wr1_correlated(common1_init[1], solution_init[2], solution_init[0], common1_init[2], common1_init[0])
					wr1_common2 = find_wr1_correlated(common2_init[2], solution_init[0], solution_init[1], common2_init[0], common2_init[1])

					if (wr1_common0 != None) and (wr1_common0 > interval[0]) and (wr1_common0 < interval[1]):
						wr.append((wr1_common0, 'wr1'))

					if (wr1_common1 != None) and (wr1_common1 > interval[0]) and (wr1_common1 < interval[1]):
						wr.append((wr1_common1, 'wr1'))

					if (wr1_common2 != None) and (wr1_common2 > interval[0]) and (wr1_common2 < interval[1]):
						wr.append((wr1_common2, 'wr1'))


		""" Find rank of solution kappa """
		
		# Transform Decimal into float
		getcontext().prec = 5
		wr = [(float(w), string) for w, string in wr]
		
		# Sort by w
		wr.sort(key=lambda x: x[0])
			
		rank, wr = find_rank(wr)
	
		# Expand the lists for plotting
		rank = [r for r in zip(rank, rank)]
		wr = [i for i in zip(wr, wr)]
			
		# Un-nest the previous lists
		rank = list(itertools.chain.from_iterable(rank))
		wr = list(itertools.chain.from_iterable(wr))
			
		# Extract abscissas for intersection points
		wr = [tuple[0] for tuple in wr]
		wr.insert(0, 0)
		wr.append(1)

		# Transform Decimal into float values
		# wr = [round(float(i), 8) for i in wr]
		
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

def sim_1x3bits_to_csv(K, kappa, num_sim):
	"""
	Plot probabilities of success for all possibilities for several simulations. 
	
	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	
	Return:
	NaN
	"""
	rank_wr_list = sim_1x3bits(K, kappa, m, num_sim)
	
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

	# Sort by absolute value, descending order
	rank_wr_df1 = rank_wr_df1.iloc[(-rank_wr_df1['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df2 = rank_wr_df2.iloc[(-rank_wr_df2['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df3 = rank_wr_df3.iloc[(-rank_wr_df3['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df4 = rank_wr_df4.iloc[(-rank_wr_df4['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df5 = rank_wr_df5.iloc[(-rank_wr_df5['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df6 = rank_wr_df6.iloc[(-rank_wr_df6['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df7 = rank_wr_df7.iloc[(-rank_wr_df7['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df8 = rank_wr_df8.iloc[(-rank_wr_df8['wr'].abs()).argsort()].reset_index(drop=True)

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

	""" Write in CSV files """

	num_rank = 1
	for df in [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4,
			   rank_wr_df5, rank_wr_df6, rank_wr_df7, rank_wr_df8]:
		file_name = r'./csv/1x3bits_abs_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (num_sim, K, kappa, num_rank)
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

		sim_1x3bits_to_csv(K, kappa, num_sim)
		
		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))