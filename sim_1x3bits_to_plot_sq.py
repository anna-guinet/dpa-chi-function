#!/usr/bin/env python

"""DPA SIMULATION 3-BIT CHI ROW & 3-BIT K"""

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

def signal_1D(mu, i, key, n=3):
	"""
	Generate signal power consumption S values for a secret state key, for all messages mu. 
	
	Parameters:
	mu -- 2D list of bool, size n x 2^n
	i -- integer in [0,n-1]
	key -- list of bool, length n
	n -- integer (default 3)
	
	Return:
	S -- list of integers, length 2^n
	"""
	S = []

	# For each message mu value
	for j in range(2**n):
			
		# Activity of the first storage cell of the register
		d = key[i] ^ mu[j][i] ^ ((key[(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[(i+2) % n] ^ mu[j][(i+2) % n]))
		
		# Power consumption model
		S_j = (-1)**(d)
	
		S.append(S_j)

	return S

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

def gen_key_i(i, kappa='000', K='000'):
	""" 
	Create key value where key = kappa, except for bit i: key_i = kappa_i XOR K
	
	Parameters:
	kappa -- string (default '000')
	K -- string (default '000')
	i -- integer in [0,n-1]
	
	Return:
	key -- list of bool
	"""
	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = list(K)
	K = [bool(int(j)) for j in K]
	
	# Initialize new key value
	key = kappa.copy()
	
	# XOR at indice i
	key[i] = K[i] ^ kappa[i]
	
	return key

def gen_S(mu, n=3, kappa='000', K='000'):
	"""
	Generate all signal power consumption for every bit i of first chi's row, 
	for a couple (kappa, K). 
	
	Parameters:
	mu -- 2D list of bool, size n x 2^n
	n -- integer (default 3)
	kappa -- string (default '000')
	K -- string (default '000')
	
	Return:
	S -- 2D list of integers, size 2^n x n
	"""
	S = []
	
	for i in range(n):
		key = gen_key_i(i, kappa, K) # Generate the key with XOR at the i-th position
		S_i = signal_1D(mu, i, key, n)
		S += [S_i]
	
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

def fct_sq_solution(w, a0, a1, a2):
	"""
	Function for the scalar product of the solution guess.
	
	Parameter:
	w -- float
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	
	Return:
	y -- Decimal
	"""
	sq_a0 = a0**2
	sq_a1 = a1**2
	sq_a2 = a2**2
	
	w = Decimal(w)
	sq_w = Decimal(w)**2
	
	# alpha = (8 - a0)^2 + (8 - a1)^2 + (8 - a2)^2
	alpha = (Decimal(8) - a0)**2 + (Decimal(8) - a1)**2 + (Decimal(8) - a2)**2 
	
	# beta = 2 (a0 (8 - a0) + a1 (8 - a1) + a2 (8 - a2))
	beta = Decimal(2)*(a0 * (Decimal(8) - a0) + a1 * (Decimal(8) - a1) + a2 * (Decimal(8) - a2))
	
	# gamma = a0^2 + a1^2 + a2^2
	gamma = sq_a0 + sq_a1 + sq_a2
		
	# y = alpha w^2 + beta w + gamma
	y = (alpha * sq_w) + (beta * w) + gamma
		
	return y / Decimal(192)

def fct_sq_common(w, a, b, c):
	"""
	Function for the scalar product of a guess which has two kappa bits in common with the solution one (a).
	
	Parameter:
	w -- float
	a -- Decimal
	b -- Decimal
	c -- Decimal
	
	Return:
	y -- Decimal
	"""	

	w = Decimal(w)
	
	y_a = (Decimal(8) - a) * w + Decimal(a)
	y_b = - b * w + b
	y_c = - c * w + c

	y = y_a**2 + y_b**2 + y_c**2

	return y / Decimal(192)

def fct_sq_nonsolution(w, b, c, d):
	"""
	Function for the scalar product of an nonsolution guess.
	
	Parameter:
	x -- float
	b -- Decimal
	c -- Decimal
	d -- Decimal
	
	Return:
	y -- Decimal
	"""
	w = Decimal(w)

	y_b = - b * w + b
	y_c = - c * w + c
	y_d = - d * w + d
	
	y = y_b**2 + y_c**2 + y_d**2
		
	return y / Decimal(192)

def find_sq_min(a0, a1, a2):
	"""
	Find the minimum point for the scalar product of the solution guess.
	
	Parameters:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	
	Return:
	w_min -- Decimal
	"""		
	# alpha = (8 - a0)^2 + (8 - a1)^2 + (8 - a2)^2
	alpha = (Decimal(8) - a0)**2 + (Decimal(8) - a1)**2 + (Decimal(8) - a2)**2 
	
	# beta = 2 (a0 (8 - a0) + a1 (8 - a1) + a2 (8 - a2))
	beta = Decimal(2) * ((a0 * (Decimal(8) - a0)) + (a1 * (Decimal(8) - a1)) + (a2 * (Decimal(8) - a2)))
	
	# gamma = a0^2 + a1^2 + a2^2
	gamma = a0**2 + a1**2 + a2**2

	w_min = Decimal()
	
	if (alpha != 0):

		# w_min = - beta / 2 alpha
		w_min = (- beta) / (Decimal(2) * alpha)
		
	else:
		w_min = None

	return w_min

def find_wr_nonsolution(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the solution guess equals the scalar product of an nonsolution guess.
	
	F(w) = (alpha - delta) w^2 + (beta +  2 delta) w + (gamma - delta) = 0 for which w? 
	
	Parameter:
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal
	d -- Decimal
	
	Return:
	w1 -- Decimal
	w2 -- Decimal
	"""
	sq_a0 = a0**2
	sq_a1 = a1**2
	sq_a2 = a2**2
	sq_b = b**2
	sq_c = c**2
	sq_d = d**2
	
	# alpha = (8 - a0)^2 + (8 - a1)^2 + (8 - a2)^2
	alpha = (Decimal(8) - a0)**2 + (Decimal(8) - a1)**2 + (Decimal(8) - a2)**2 
	
	# beta = 2 (a0 (8 - a0) + a1 (8 - a1) + a2 (8 - a2))
	beta = Decimal(2)*(a0 * (Decimal(8) - a0) + a1 * (Decimal(8) - a1) + a2 * (Decimal(8) - a2))
	
	# gamma = a0^2 + a1^2 + a2^2
	gamma = sq_a0 + sq_a1 + sq_a2
	
	# delta = b^2 + c^2 + d^2
	delta = sq_b + sq_c + sq_d
	
	# discriminant = (beta + 2 delta)^2 - 4 (alpha - delta) (gamma - delta)
	discriminant = (beta + (Decimal(2) * delta))**2 - (Decimal(4) * (alpha - delta) * (gamma - delta))
	
	w1 = Decimal()
	w2 = Decimal()
	
	if (alpha != delta) and (discriminant >= 0):
		
		sqrt_discriminant = Decimal(discriminant).sqrt()
	
		# w1 = - beta - 2 delta + sqrt_discriminant / 2 (alpha - delta)
		w1 = (- beta - (Decimal(2) * delta) + sqrt_discriminant) / (Decimal(2) * (alpha - delta))
		
		# w2 = - beta - 2 delta - sqrt_discriminant / 2 (alpha - delta)
		w2 = (- beta - (Decimal(2) * delta) - sqrt_discriminant) / (Decimal(2) * (alpha - delta))
		
	else:
		w1 = None
		w2 = None

	return w1, w2
	
def find_wr_common(a, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the solution guess equals the scalar product of an common guess on a.
	
	F(w) = (alpha - zeta) w^2 + (beta - epsilon) w + (gamma - delta) = 0 for which w? 
	
	Parameter:
	a -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	b -- Decimal
	c -- Decimal
	
	Return:
	w1 -- Decimal
	w2 -- Decimal
	"""	
	sq_a = a**2
	sq_a1 = a1**2
	sq_a2 = a2**2
	sq_b = b**2
	sq_c = c**2
	
	# alpha = (8 - a)^2 + (8 - a1)^2 + (8 - a2)^2
	alpha = (Decimal(8) - a)**2 + (Decimal(8) - a1)**2 + (Decimal(8) - a2)**2 
	
	# beta = 2 (a (8 - a) + a1 (8 - a1) + a2 (8 - a2))
	beta = Decimal(2)*(a * (Decimal(8) - a) + a1 * (Decimal(8) - a1) + a2 * (Decimal(8) - a2))
	
	# gamma = a^2 + a1^2 + a2^2
	gamma = sq_a + sq_a1 + sq_a2
	
	# delta = a^2 + b^2 + c^2
	delta = sq_a + sq_b + sq_c
	
	# epsilon = 2 (a (8 - a) - b^2 - c^2)
	epsilon = Decimal(2) * (a * (Decimal(8) - a) - sq_b - sq_c)
	
	# zeta = (8 - a)^2 + b^2 + c^2
	zeta = (Decimal(8) - a)**2 + sq_b + sq_c
	
	# discriminant = (beta - epsilon)^2 - 4 (alpha - zeta) (gamma - delta)
	discriminant = (beta - epsilon)**2 - (Decimal(4) * (alpha - zeta) * (gamma - delta))

	w1 = Decimal()
	w2 = Decimal()
	
	if (alpha != zeta) and (discriminant >= 0):
		
		sqrt_discriminant = Decimal(discriminant).sqrt()
	
		# w1 = - (beta - epsilon) + sqrt_discriminant / 2 (alpha - zeta)
		w1 = (- beta + epsilon + sqrt_discriminant) / (Decimal(2) * (alpha - zeta))
		
		# w2 = - (beta - epsilon) - sqrt_discriminant / 2 (alpha - zeta)
		w2 = (- beta + epsilon - sqrt_discriminant) / (Decimal(2) * (alpha - zeta))
		
	else:
		w1 = None
		w2 = None

	return w1, w2

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

		# Count number of rank increment (_1) and rank decrement (_2)
		counter_1 = sum(1 for tuple in wr if tuple[1].endswith('_1'))
		counter_2 = sum(-1 for tuple in wr if tuple[1].endswith('_2'))
		num_wr = counter_1 + counter_2
		
		rank_init = 1 + num_wr
		
		rank.append(rank_init)
		
		for tuple in wr:
		
			# If solution 1, rank increases
			if tuple[1].endswith('_1'):
				rank.append(rank[-1] - 1)
				
			# If solution 2, rank decreases
			if tuple[1].endswith('_2'):
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, wr

def xor_secret_i(K, kappa, i):
	""" 
	Compute power consumption at bit i = kappa_i XOR K_i
	
	Parameters:
	kappa -- string
	K -- string
	i -- integer
	
	Return:
	p_i -- float
	"""

	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = list(K)
	K = [bool(int(j)) for j in K]
	
	# Initialize new kappa values
	kappa_copy = kappa.copy()
	K_copy = K.copy()
	
	# XOR at indice i
	d = K_copy[i] ^ kappa_copy[i]
	
	# Power consumption at i-th bit
	p_i = (-1)**d

	return p_i
	
def sim_1x3bits(n, K, kappa, num_sim):
	"""
	Compute simulation' scalar products and plot outcomes. 
	
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
	S_ref = gen_S_ref(mu, n)

	""" Initial values of scalar product """
	scalar_init = []

	# Global power consumption P_init
	P_init = [Decimal(r) for r in R]

	# i-th bit of register state studied
	for i in range(n):

		# Scalar product <S-ref, P>
		scalar_init_i = np.dot(S_ref[i], P_init)
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

	nonsolution_idx = [sublist for sublist in list_kappa_idx if (sublist[1] != solution_idx[1]) and (sublist[2] != solution_idx[2]) and (sublist[3] != solution_idx[3])]
	nonsolution0_idx = nonsolution_idx[0]
	nonsolution1_idx = nonsolution_idx[1]
	nonsolution2_idx = nonsolution_idx[2]
	nonsolution3_idx = nonsolution_idx[3]
	
	""" To find initial scalar product values """
	solution_init = [init_scalar[i][solution_idx[i+1]] for i in range(3)]
	
	common0_init = [init_scalar[i][common0_idx[i+1]] for i in range(3)]
	common1_init = [init_scalar[i][common1_idx[i+1]] for i in range(3)]
	common2_init = [init_scalar[i][common2_idx[i+1]] for i in range(3)]
	
	nonsolution0_init = [init_scalar[i][nonsolution0_idx[i+1]] for i in range(3)]
	nonsolution1_init = [init_scalar[i][nonsolution1_idx[i+1]] for i in range(3)]
	nonsolution2_init = [init_scalar[i][nonsolution2_idx[i+1]] for i in range(3)]
	nonsolution3_init = [init_scalar[i][nonsolution3_idx[i+1]] for i in range(3)]

	
	""" Find the functions according to the kappas """
	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,6), gridspec_kw={"height_ratios": [2,1]}, sharex=True)
	
	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
	
	# Abscissas
	precision = 1000
	w_list = [(w / precision) for w in range(precision)]
	
	# Determine the sign of initial solution value depending on the activity of the register at bit i
	p_0 = xor_secret_i(K, kappa, 0)
	p_1 = xor_secret_i(K, kappa, 1)
	p_2 = xor_secret_i(K, kappa, 2)
	solution_init[0] = solution_init[0] * p_0
	solution_init[1] = solution_init[1] * p_1
	solution_init[2] = solution_init[2] * p_2

	# solution function
	solution_sq = [fct_sq_solution(w, solution_init[0], solution_init[1], solution_init[2]) for w in w_list]
	ax1.plot(w_list, solution_sq, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[0])
	
	# common functions
	common0_sq = [fct_sq_common(w, solution_init[0], common0_init[1], common0_init[2]) for w in w_list]
	common1_sq = [fct_sq_common(w, solution_init[1], common1_init[2], common1_init[0]) for w in w_list]
	common2_sq = [fct_sq_common(w, solution_init[2], common2_init[0], common2_init[1]) for w in w_list]
	ax1.plot(w_list, common0_sq, '-', color='tab:brown', markersize=1, label='%s' % common0_idx[0])
	ax1.plot(w_list, common1_sq, '-', color='tab:olive', markersize=1, label='%s' % common1_idx[0])
	ax1.plot(w_list, common2_sq, '-', color='tab:purple', markersize=1, label='%s' % common2_idx[0])
	
	# nonsolution functions
	nonsolution0_sq = [fct_sq_nonsolution(w, nonsolution0_init[0], nonsolution0_init[1], nonsolution0_init[2]) for w in w_list]
	nonsolution1_sq = [fct_sq_nonsolution(w, nonsolution1_init[0], nonsolution1_init[1], nonsolution1_init[2]) for w in w_list]
	nonsolution2_sq = [fct_sq_nonsolution(w, nonsolution2_init[0], nonsolution2_init[1], nonsolution2_init[2]) for w in w_list]
	nonsolution3_sq = [fct_sq_nonsolution(w, nonsolution3_init[0], nonsolution3_init[1], nonsolution3_init[2]) for w in w_list]
	ax1.plot(w_list, nonsolution0_sq, '-', color='tab:red', markersize=1, label='%s' % nonsolution0_idx[0])
	ax1.plot(w_list, nonsolution1_sq, '-', color='tab:orange', markersize=1, label='%s' % nonsolution1_idx[0])
	ax1.plot(w_list, nonsolution2_sq, '-', color='tab:pink', markersize=1, label='%s' % nonsolution2_idx[0])
	ax1.plot(w_list, nonsolution3_sq, '-', color='tab:green', markersize=1, label='%s' % nonsolution3_idx[0])

	""" Loop for several simulations with squared scalar product for verification purpose
	# Results of scalar products
	scalar_000_sq = []
	scalar_001_sq = []
	scalar_010_sq = []
	scalar_011_sq = []
	scalar_100_sq = []
	scalar_101_sq = []
	scalar_110_sq = []
	scalar_111_sq = []
	
	w_list = []
	
	# Signal power consumption for a key
	S = gen_S(mu, n, kappa, K)
	
	for j in range(10000):

		# Weight of signal part
		w = j / 10000
		
		scalar_sq = []
	
		# i-th bit of register state studied
		for i in range(n):
			
			# Global power consumption P = (1 - w) * R + w S
			#P_i = [(r + w s) for r, s in zip(R, S[i])]	
			P_i = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S[i])]

			# Squared scalar product <S-ref, P>^2
			scalar_i = np.dot(S_ref[i], P_i)
			scalar_sq_i = scalar_i ** 2
			scalar_sq += [scalar_sq_i]
		
		# Scalar product for i = 0
		# s0 : ['_00', '_01', '_10', '_11', '_00', '_01', '_10', '_11']
		s0_sq = scalar_sq[0].tolist()
		s0_sq = s0_sq + s0_sq # Concatenate

		# Scalar product for i = 1
		# s1 : [(('0_0', '0_1'), ('0_0', '0_1')), (('1_0', '1_1'), ('1_0', '1_1'))]		
		s1_sq = [(s,s) for s in zip(scalar_sq[1][::2], scalar_sq[1][1::2])]
		s1_sq = list(itertools.chain.from_iterable(s1_sq)) 
		s1_sq = list(itertools.chain.from_iterable(s1_sq)) # Un-nest the previous list twice

		# Scalar product for i = 2
		# s2 : [('00_', '00_'), ('01_', '01_'), ('10_', '10_'), ('11_', '11_')]
		s2_sq = [s for s in zip(scalar_sq[2], scalar_sq[2])]
		s2_sq = list(itertools.chain.from_iterable(s2_sq)) # Un-nest the previous list

		# Sum of sqaured scalar products for all kappa values
		scalar_sq = [s0_sq[mu] + s1_sq[mu] + s2_sq[mu] for mu in range(2**n)]
		
		w_list.append(w)
		
		scalar_000_sq.append(scalar_sq[0])
		scalar_001_sq.append(scalar_sq[1])
		scalar_010_sq.append(scalar_sq[2])
		scalar_011_sq.append(scalar_sq[3])
		scalar_100_sq.append(scalar_sq[4])
		scalar_101_sq.append(scalar_sq[5])
		scalar_110_sq.append(scalar_sq[6])
		scalar_111_sq.append(scalar_sq[7])
	
	ax1.plot(w_list, scalar_000_sq, ':', color='black', markersize=4, label='000')
	ax1.plot(w_list, scalar_001_sq, ':', color='black',  markersize=4, label='001')
	ax1.plot(w_list, scalar_010_sq, ':', color='black', markersize=4, label='010')
	ax1.plot(w_list, scalar_011_sq, '-.', color='black', markersize=4, label='011')
	ax1.plot(w_list, scalar_100_sq, ':', color='black', markersize=4, label='100')
	ax1.plot(w_list, scalar_101_sq, '-.', color='black',  markersize=4, label='101')
	ax1.plot(w_list, scalar_110_sq, '-.', color='black', markersize=4, label='110')
	ax1.plot(w_list, scalar_111_sq, '-', color='black', markersize=4, label='111')
	"""

		
	""" Find the intersections with the solution kappa function """
	# List of all intersections
	wr = []
	
	# Minimum of solution function
	w_min = find_sq_min(solution_init[0], solution_init[1], solution_init[2])
	
	if (w_min != None) and (w_min > 0) and (w_min < 1):
		ax1.axvline(x=w_min, linestyle=':', color='tab:blue', markersize=1, label='w_min')
	
	# Intersections solution function and common functions
	wr_kappa0_1, wr_kappa0_2 = find_wr_common(solution_init[0], solution_init[1], solution_init[2], common0_init[1], common0_init[2])
	wr_kappa1_1, wr_kappa1_2 = find_wr_common(solution_init[1], solution_init[2], solution_init[0], common1_init[2], common1_init[0])
	wr_kappa2_1, wr_kappa2_2 = find_wr_common(solution_init[2], solution_init[0], solution_init[1], common2_init[0], common2_init[1])
	
	if (wr_kappa0_1 != None) and (wr_kappa0_1 > 0) and (wr_kappa0_1 < 1):
		wr.append((wr_kappa0_1, 'wr_kappa0_1'))
		# ax1.axvline(x=wr_kappa0_1, linestyle='--', color='tab:brown', markersize=1)
			
	if (wr_kappa0_2 != None) and (wr_kappa0_2 > 0) and (wr_kappa0_2 < 1):
		wr.append((wr_kappa0_2, 'wr_kappa0_2'))
		# ax1.axvline(x=wr_kappa0_2, linestyle=':', color='tab:brown', markersize=1)
			
	if (wr_kappa1_1 != None) and (wr_kappa1_1 > 0) and (wr_kappa1_1 < 1):
		wr.append((wr_kappa1_1, 'wr_kappa1_1'))
		# ax1.axvline(x=wr_kappa1_1, linestyle='--', color='tab:olive', markersize=1)
		
	if (wr_kappa1_2 != None) and (wr_kappa1_2 > 0) and (wr_kappa1_2 < 1):
		wr.append((wr_kappa1_2, 'wr_kappa1_2'))
		# ax1.axvline(x=wr_kappa1_2, linestyle=':', color='tab:olive', markersize=1)
		
	if (wr_kappa2_1 != None) and (wr_kappa2_1 > 0) and (wr_kappa2_1 < 1):
		wr.append((wr_kappa2_1, 'wr_kappa2_1'))
		# ax1.axvline(x=wr_kappa2_1, linestyle='--', color='tab:purple', markersize=1)
			
	if (wr_kappa2_2 != None) and (wr_kappa2_2 > 0) and (wr_kappa2_2 < 1):
		wr.append((wr_kappa2_2, 'wr_kappa2_2'))
		# ax1.axvline(x=wr_kappa2_2, linestyle=':', color='tab:purple', markersize=1)
		
	# Intersections solution function and nonsolution functions	
	wr_nonsolution0_1, wr_nonsolution0_2 = find_wr_nonsolution(solution_init[0], solution_init[1], solution_init[2], nonsolution0_init[0], nonsolution0_init[1], nonsolution0_init[2])
	wr_nonsolution1_1, wr_nonsolution1_2 = find_wr_nonsolution(solution_init[0], solution_init[1], solution_init[2], nonsolution1_init[0], nonsolution1_init[1], nonsolution1_init[2])
	wr_nonsolution2_1, wr_nonsolution2_2 = find_wr_nonsolution(solution_init[0], solution_init[1], solution_init[2], nonsolution2_init[0], nonsolution2_init[1], nonsolution2_init[2])
	wr_nonsolution3_1, wr_nonsolution3_2 = find_wr_nonsolution(solution_init[0], solution_init[1], solution_init[2], nonsolution3_init[0], nonsolution3_init[1], nonsolution3_init[2])

	if (wr_nonsolution0_1 != None) and (wr_nonsolution0_1 > 0) and (wr_nonsolution0_1 < 1):
		wr.append((wr_nonsolution0_1, 'wr_nonsolution0_1'))
		# ax1.axvline(x=wr_nonsolution0_1, linestyle='--', color='tab:red', markersize=1)
		
	if (wr_nonsolution0_2 != None) and (wr_nonsolution0_2 > 0) and (wr_nonsolution0_2 < 1):
		wr.append((wr_nonsolution0_2, 'wr_nonsolution0_2'))
		# ax1.axvline(x=wr_nonsolution0_2, linestyle=':', color='tab:red', markersize=1)
		
	if (wr_nonsolution1_1 != None) and (wr_nonsolution1_1 > 0) and (wr_nonsolution1_1 < 1):
		wr.append((wr_nonsolution1_1, 'wr_nonsolution1_1'))
		# ax1.axvline(x=wr_nonsolution1_1, linestyle='--', color='tab:orange', markersize=1)
		
	if (wr_nonsolution1_2 != None) and (wr_nonsolution1_2 > 0) and (wr_nonsolution1_2 < 1):
		wr.append((wr_nonsolution1_2, 'wr_nonsolution1_2'))
		# ax1.axvline(x=wr_nonsolution1_2, linestyle=':', color='tab:orange', markersize=1)
		
	if (wr_nonsolution2_1 != None) and (wr_nonsolution2_1 > 0) and (wr_nonsolution2_1 < 1):
		wr.append((wr_nonsolution2_1, 'wr_nonsolution2_1'))
		# ax1.axvline(x=wr_nonsolution2_1, linestyle='--', color='tab:pink', markersize=1)
		
	if (wr_nonsolution2_2 != None) and (wr_nonsolution2_2 > 0) and (wr_nonsolution2_2 < 1):
		wr.append((wr_nonsolution2_2, 'wr_nonsolution2_2'))
		# ax1.axvline(x=wr_nonsolution2_2, linestyle=':', color='tab:pink', markersize=1)
		
	if (wr_nonsolution3_1 != None) and (wr_nonsolution3_1 > 0) and (wr_nonsolution3_1 < 1):
		wr.append((wr_nonsolution3_1, 'wr_nonsolution3_1'))
		# ax1.axvline(x=wr_nonsolution3_1, linestyle='--', color='tab:green', markersize=1)
		
	if (wr_nonsolution3_2 != None) and (wr_nonsolution3_2 > 0) and (wr_nonsolution3_2 < 1):
		wr.append((wr_nonsolution3_2, 'wr_nonsolution3_2'))
		# ax1.axvline(x=wr_nonsolution3_2, linestyle=':', color='tab:green', markersize=1)
	

	""" Find rank of solution kappa	"""
	
	wr.sort(key=lambda x: x[0])
	
	rank, wr = find_rank(wr)
	
	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr = [i for i in zip(wr, wr)]
		
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr = list(itertools.chain.from_iterable(wr))
		
	# Extract abscissas for wrsection points
	wr = [tuple[0] for tuple in wr]
	wr.insert(0, 0)
	wr.append(1)

	ax2.plot(wr, rank, color='tab:blue')

	ax1.legend(loc='upper left', ncol=3, fancybox=True)
	ax1.set_title(r'Combining three scalar products for K=%s and kappa=%s' %(K, kappa))
	ax1.set_ylabel('Scalar product <S_ref,P>^2')
	ax2.set_ylabel('Rank kappa solution')
	ax2.set_xlabel('Weight w')
	# ax2.invert_yaxis()
	ax2.set_ylim([9, 0])
	

	fig.tight_layout()
	# plt.show()
	plt.savefig('./plot/sim_1x3bits_sq_K=%s_kappa=%s_#%s.png' % (K, kappa, num_sim))
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
			sim_1x3bits(n, K, kappa, j)

		# Time option
		end_t = time.perf_counter()
		print('\ntime', end_t - start_t, '\n')
 
if __name__ == '__main__':
	sys.exit(main(sys.argv))



