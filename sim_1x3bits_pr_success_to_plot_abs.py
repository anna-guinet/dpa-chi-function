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

def fct_abs_correct(w, a0, a1, a2):
	"""
	Function for the scalar product of the correct guess.
	
	Parameter:
	w -- float
	a0 -- Decimal
	a1 -- Decimal
	a2 -- Decimal
	
	Return:
	y -- Decimal
	"""
	w = Decimal(w)

	y0 = (Decimal(8) - a0) * w + a0
	y1 = (Decimal(8) - a1) * w + a1
	y2 = (Decimal(8) - a2) * w + a2
	
	y = abs(y0) + abs(y1) + abs(y2)

	return y / Decimal(24)

def fct_abs_common(w, a, b, c):
	"""
	Function for the scalar product of a guess which has two kappa bits in common with the correct one (a).
	
	Parameter:
	w -- float
	a -- Decimal
	b -- Decimal
	c -- Decimal

	Return:
	y -- Decimal
	"""	
	w = Decimal(w)

	y_a = (Decimal(8) - a) * w + a
	y_b = - b * w + b
	y_c = - c * w + c

	y = abs(y_a) + abs(y_b) + abs(y_c)

	return y / Decimal(24)

def fct_abs_incorrect(w, b, c, d):
	"""
	Function for the scalar product of an incorrect guess.
	
	Parameter:
	w -- float
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

	y = abs(y_b) + abs(y_c) + abs(y_d)
		
	return y / Decimal(24)

def reorder_common(correct_init, common_init, common_idx):
	"""
	Reorder common_init and common_idx list according to the sorted correct_init.
	Common kappa's scalar product values changed order because of the sorting on w0 values. 
	
	Parameters:
	correct_init -- list of Decimal
	common_init -- list of lists
	common_idx -- list of strings
	
	Return:
	common_new -- list of tuples
	"""	
	# Merge initial scalar products and kappa strings
	common = [elem for elem in zip(common_init, common_idx)]

	common_new = []

	for i in range(3):
		for tup in common:
			if correct_init[i] in tup[0]:

				tup_new = []

				# Add non-common initial scalar product values in new sublist
				common_init_new = [elem for elem in tup[0] if elem != correct_init[i]]

				# If ever non-common initial scalar product == common initial scalar product
				if len(common_init_new) == 1:
					common_init_new.append(correct_init[i])
				if len(common_init_new) == 0:
					common_init_new.append(correct_init[i])
					common_init_new.append(correct_init[i])

				# Insert common initial scalar product value at the same index than sorted correct_init list
				common_init_new.insert(i, correct_init[i])

				# Kappa value at the same index than sorted correct_init list
				common_str_new = tup[1] 

				tup_new = (common_init_new, common_str_new)

				common_new += [tup_new]

	return common_new

def find_inter1_incorrect(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an incorrect guess.

	Fct_incorrect(w) = fct_correct(w), for which w in [0, w0_0] with 0 < w0_0 < w0_1 < w0_2 < 1 ?
	
	fct_correct(w) = (alpha1 - 24) w - alpha1
	fct_incorrect(w) = -beta w + beta
	
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

def find_inter2_incorrect(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an incorrect guess.

	Fct_incorrect(w) = fct_correct(w), for which w in [w0_0, w0_1] with 0 =< w0_0 < w0_1 < w0_2 < 1 ?
	
	fct_correct(w) = (alpha2 - 8) w - alpha2
	fct_incorrect(w) = -beta w + beta
	
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

def find_inter3_incorrect(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an incorrect guess.

	Fct_incorrect(w) = fct_correct(w), for which w in [w0_1, w0_2] with 0 =< w0_1 < w0_2 < 1 ?
	
	fct_correct(w) = (8 - alpha3) w + alpha3
	fct_incorrect(w) = -beta w + beta
	
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

def find_inter4_incorrect(a0, a1, a2, b, c, d):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an incorrect guess.

	Fct_incorrect(w) = fct_correct(w), for which w in [w0_2, 1] with 0 =< w0_2 < 1?
	
	fct_correct(w) = (24 - alpha1) w + alpha1
	fct_incorrect(w) = -beta w + beta
	
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
	
def find_inter1_common(a0, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an common guess on a0.

	Fct_common(w) = fct_correct(w), for which w in [0, w0_0] with 0 < w0_0 < w0_1 < w0_2 < 1 ?
	
	fct_correct(w) = (alpha1 - 24) w - alpha1
	fct_common(w) = (beta1 - 8) w - beta1

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

def find_inter2_common(a0, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an common guess on a0.

	Fct_common(w) = fct_correct(w), for which w in [w0_0, w0_1] with 0 =< w0_0 < w0_1 < w0_2 < 1 ?
	
	fct_correct(w) = (alpha2 - 8) w - alpha2
	fct_common(w) = (8 - beta2) w + beta2 for correlation on a0

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

def find_inter3_common(a2, a0, a1, b, c):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an common guess on a2.

	Fct_common(w) = fct_correct(w), for which w in [w0_1, w0_2] with 0 =< w0_1 < w0_2 < 1 ?
	
	fct_correct(w) = (8 - alpha3) w + alpha3
	fct_common(w) = (beta1 - 8) w - beta1 for correlation on a2

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

def find_inter4_common(a0, a1, a2, b, c):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an common guess on a0.

	Fct_common(w) = fct_correct(w), for which w in [w0_2, 1] with 0 =< w0_2 < 1 ?
	
	fct_correct(w) = (24 - alpha1) w + alpha1
	fct_common(w) = (8 - beta2) w + beta2

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

def find_rank(inter):
	"""
	Return the list of ranks for the correct kappa.
	
	Parameter:
	inter -- list of Decimal
	
	Return:
	rank -- list of integer
	"""
	# List of ranks
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if inter:

		# Count number of rank increment ('inter4' or 'inter3') and rank decrement ('inter2' or 'inter1')
		counter_1 = sum(1 for tuple in inter if tuple[1] == ('inter4') or tuple[1] == ('inter3'))
		counter_2 = sum(-1 for tuple in inter if tuple[1] == ('inter2') or tuple[1] == ('inter1'))
		num_inter = counter_1 + counter_2
		
		rank_init = 1 + num_inter
		
		rank.append(rank_init)
		
		for tuple in inter:
		
			# If 'inter4'or 'inter3', rank increases
			if tuple[1] == ('inter4') or tuple[1] == ('inter3'):
				rank.append(rank[-1] - 1)
				
			# If 'inter2' or 'inter1', rank decreases
			if tuple[1] == ('inter2') or tuple[1] == ('inter1'):
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, inter

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

	correct_idx = [sublist for sublist in list_kappa_idx if sublist[0] == kappa]
	correct_idx = list(itertools.chain.from_iterable(correct_idx)) # Un-nest the previous list
	
	common0_idx = [sublist for sublist in list_kappa_idx if (sublist[1] == correct_idx[1]) and (sublist[0] != kappa)]
	common0_idx = list(itertools.chain.from_iterable(common0_idx)) # Un-nest the previous list

	common1_idx = [sublist for sublist in list_kappa_idx if (sublist[2] == correct_idx[2]) and (sublist[0] != kappa)]
	common1_idx = list(itertools.chain.from_iterable(common1_idx)) # Un-nest the previous list
	
	common2_idx = [sublist for sublist in list_kappa_idx if (sublist[3] == correct_idx[3]) and (sublist[0] != kappa)]
	common2_idx = list(itertools.chain.from_iterable(common2_idx)) # Un-nest the previous list

	incorrect_idx = [sublist for sublist in list_kappa_idx if (sublist[1] != correct_idx[1]) and (sublist[2] != correct_idx[2]) and (sublist[3] != correct_idx[3])]
	incorrect0_idx = incorrect_idx[0]
	incorrect1_idx = incorrect_idx[1]
	incorrect2_idx = incorrect_idx[2]
	incorrect3_idx = incorrect_idx[3]
	
	""" To find initial scalar product values """
	correct_init = [init_scalar[i][correct_idx[i+1]] for i in range(3)]
	
	common0_init = [init_scalar[i][common0_idx[i+1]] for i in range(3)]
	common1_init = [init_scalar[i][common1_idx[i+1]] for i in range(3)]
	common2_init = [init_scalar[i][common2_idx[i+1]] for i in range(3)]
	
	incorrect0_init = [init_scalar[i][incorrect0_idx[i+1]] for i in range(3)]
	incorrect1_init = [init_scalar[i][incorrect1_idx[i+1]] for i in range(3)]
	incorrect2_init = [init_scalar[i][incorrect2_idx[i+1]] for i in range(3)]
	incorrect3_init = [init_scalar[i][incorrect3_idx[i+1]] for i in range(3)]

	# Determine the sign of initial correct value depending on the activity of the register at bit i
	p_0 = xor_secret_i(K, kappa, 0)
	p_1 = xor_secret_i(K, kappa, 1)
	p_2 = xor_secret_i(K, kappa, 2)
	correct_init[0] = correct_init[0] * p_0
	correct_init[1] = correct_init[1] * p_1
	correct_init[2] = correct_init[2] * p_2
	common0_init[0] = common0_init[0] * p_0
	common1_init[1] = common1_init[1] * p_1
	common2_init[2] = common2_init[2] * p_2

	# Find w0 for each bit of the register
	# tup = (w0, a)
	list_tup_w0 = find_abs_w0(correct_init[0], correct_init[1], correct_init[2])

	# Sort result by increasing order on w0
	list_tup_w0 = sorted(list_tup_w0, key=lambda x: x[0])

	# Sorted list of correct_init by increasing order on w0
	correct_init = [tup[1] for tup in list_tup_w0]

	# Prepare sorting of common vectors according to w0
	common_init = [common0_init, common1_init, common2_init]
	common_idx = [common0_idx[0], common1_idx[0], common2_idx[0]]

	# Sorted list of (common_init, common_idx) according to correct_init
	common = reorder_common(correct_init, common_init, common_idx)
	common0_init = common[0][0]
	common1_init = common[1][0]
	common2_init = common[2][0]
	common0_str = common[0][1]
	common1_str = common[1][1]
	common2_str = common[2][1]

	# Sorted list of w0 by increasing order
	list_w0 = [tup[0] for tup in list_tup_w0]


	""" Find the functions according to the kappas and the intersections with correct function """

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6,5), gridspec_kw={"height_ratios": [2,1]}, sharex=True)
	
	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
	
	# Delimit definition intervals for w
	for w0 in list_w0:
		label_w0 = ["w_0", "w'_0", "w''_0"]
		style=['-.', ':', '--']
		if (w0 > 0) and (w0 < 1):
			ax1.axvline(x=w0, linestyle=style[list_w0.index(w0)], color='tab:blue', markersize=1, label=label_w0[list_w0.index(w0)])

	# List of all intersections with the correct kappa function
	inter = []

	# Incorrect functions
	interval = [0, 1]

	incorrect0_abs = [fct_abs_incorrect(w, incorrect0_init[0], incorrect0_init[1], incorrect0_init[2]) for w in interval]
	incorrect1_abs = [fct_abs_incorrect(w, incorrect1_init[0], incorrect1_init[1], incorrect1_init[2]) for w in interval]
	incorrect2_abs = [fct_abs_incorrect(w, incorrect2_init[0], incorrect2_init[1], incorrect2_init[2]) for w in interval]
	incorrect3_abs = [fct_abs_incorrect(w, incorrect3_init[0], incorrect3_init[1], incorrect3_init[2]) for w in interval]
	ax1.plot(interval, incorrect0_abs, '-', color='tab:red', markersize=1, label='%s' % incorrect0_idx[0])
	ax1.plot(interval, incorrect1_abs, '-', color='tab:orange', markersize=1, label='%s' % incorrect1_idx[0])
	ax1.plot(interval, incorrect2_abs, '-', color='tab:pink', markersize=1, label='%s' % incorrect2_idx[0])
	ax1.plot(interval, incorrect3_abs, '-', color='tab:green', markersize=1, label='%s' % incorrect3_idx[0])

	# Correct function
	interval = [list_w0[2], 1]

	correct_abs = [fct_abs_correct(w, correct_init[0], correct_init[1], correct_init[2]) for w in interval]
	ax1.plot(interval, correct_abs, '-', color='tab:blue', markersize=1, label='%s' % correct_idx[0])

	# common functions
	common0_abs = [fct_abs_common(w, common0_init[0], common0_init[1], common0_init[2]) for w in interval]
	common1_abs = [fct_abs_common(w, common1_init[1], common1_init[2], common1_init[0]) for w in interval]
	common2_abs = [fct_abs_common(w, common2_init[2], common2_init[0], common2_init[1]) for w in interval]
	ax1.plot(interval, common0_abs, '-', color='tab:brown', markersize=1, label='%s' % common0_str)
	ax1.plot(interval, common1_abs, '-', color='tab:olive', markersize=1, label='%s' % common1_str)
	ax1.plot(interval, common2_abs, '-', color='tab:purple', markersize=1, label='%s' % common2_str)

	# Intersections between correct function and incorrect functions	
	inter4_incorrect0 = find_inter4_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect0_init[0], incorrect0_init[1], incorrect0_init[2])
	inter4_incorrect1 = find_inter4_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect1_init[0], incorrect1_init[1], incorrect1_init[2])
	inter4_incorrect2 = find_inter4_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect2_init[0], incorrect2_init[1], incorrect2_init[2])
	inter4_incorrect3 = find_inter4_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect3_init[0], incorrect3_init[1], incorrect3_init[2])

	if (inter4_incorrect0 != None) and (inter4_incorrect0 > interval[0]) and (inter4_incorrect0 < interval[1]):
		inter.append((inter4_incorrect0, 'inter4'))
		# ax1.axvline(x=inter4_incorrect0, linestyle='--', color='tab:red', markersize=1, label='inter4')

	if (inter4_incorrect1 != None) and (inter4_incorrect1 > interval[0]) and (inter4_incorrect1 < interval[1]):
		inter.append((inter4_incorrect1, 'inter4'))
		# ax1.axvline(x=inter4_incorrect1, linestyle='--', color='tab:orange', markersize=1, label='inter4')

	if (inter4_incorrect2 != None) and (inter4_incorrect2 > interval[0]) and (inter4_incorrect2 < interval[1]):
		inter.append((inter4_incorrect2, 'inter4'))
		# ax1.axvline(x=inter4_incorrect2, linestyle='--', color='tab:pink', markersize=1, label='inter4')

	if (inter4_incorrect3 != None) and (inter4_incorrect3 > interval[0]) and (inter4_incorrect3 < interval[1]):
		inter.append((inter4_incorrect3, 'inter4'))
		# ax1.axvline(x=inter4_incorrect3, linestyle='--', color='tab:green', markersize=1, label='inter4')

	# Intersections between correct function and common functions	
	inter4_common0 = find_inter4_common(common0_init[0], correct_init[1], correct_init[2], common0_init[1], common0_init[2])
	inter4_common1 = find_inter4_common(common1_init[1], correct_init[2], correct_init[0], common1_init[2], common1_init[0])
	inter4_common2 = find_inter4_common(common2_init[2], correct_init[0], correct_init[1], common2_init[0], common2_init[1])

	if (inter4_common0 != None) and (inter4_common0 > interval[0]) and (inter4_common0 < interval[1]):
		inter.append((inter4_common0, 'inter4'))
		# ax1.axvline(x=inter4_common0, linestyle='--', color='tab:brown', markersize=1, label='inter4')

	if (inter4_common1 != None) and (inter4_common1 > interval[0]) and (inter4_common1 < interval[1]):
		inter.append((inter4_common1, 'inter4'))
		# ax1.axvline(x=inter4_common1, linestyle='--', color='tab:olive', markersize=1, label='inter4')

	if (inter4_common2 != None) and (inter4_common2 > interval[0]) and (inter4_common2 < interval[1]):
		inter.append((inter4_common2, 'inter4'))
		# ax1.axvline(x=inter4_common2, linestyle='--', color='tab:purple', markersize=1, label='inter4')

	# ------------------------------------------------------------------------------------------- #

	# Correct function and common ones
	if list_w0[2] != 0:

		interval = [list_w0[1], list_w0[2]]

		# Correct function
		correct_abs = [fct_abs_correct(w, correct_init[0], correct_init[1], correct_init[2]) for w in interval]
		ax1.plot(interval, correct_abs, '-', color='tab:blue', markersize=1)

		# common functions
		common0_abs = [fct_abs_common(w, common0_init[0], common0_init[1], common0_init[2]) for w in interval]
		common1_abs = [fct_abs_common(w, common1_init[1], common1_init[2], common1_init[0]) for w in interval]
		common2_abs = [fct_abs_common(w, common2_init[2], common2_init[0], common2_init[1]) for w in interval]
		ax1.plot(interval, common0_abs, '-', color='tab:brown', markersize=1)
		ax1.plot(interval, common1_abs, '-', color='tab:olive', markersize=1)
		ax1.plot(interval, common2_abs, '-', color='tab:purple', markersize=1)

		# Intersections between correct function and incorrect functions	
		inter3_incorrect0 = find_inter3_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect0_init[0], incorrect0_init[1], incorrect0_init[2])
		inter3_incorrect1 = find_inter3_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect1_init[0], incorrect1_init[1], incorrect1_init[2])
		inter3_incorrect2 = find_inter3_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect2_init[0], incorrect2_init[1], incorrect2_init[2])
		inter3_incorrect3 = find_inter3_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect3_init[0], incorrect3_init[1], incorrect3_init[2])

		if (inter3_incorrect0 != None) and (inter3_incorrect0 > interval[0]) and (inter3_incorrect0 < interval[1]):
			inter.append((inter3_incorrect0, 'inter3'))
			# ax1.axvline(x=inter3_incorrect0, linestyle='-.', color='tab:red', markersize=1, label='inter3')

		if (inter3_incorrect1 != None) and (inter3_incorrect1 > interval[0]) and (inter3_incorrect1 < interval[1]):
			inter.append((inter3_incorrect1, 'inter3'))
			# ax1.axvline(x=inter3_incorrect1, linestyle='-.', color='tab:orange', markersize=1, label='inter3')

		if (inter3_incorrect2 != None) and (inter3_incorrect2 > interval[0]) and (inter3_incorrect2 < interval[1]):
			inter.append((inter3_incorrect2, 'inter3'))
			# ax1.axvline(x=inter3_incorrect2, linestyle='-.', color='tab:pink', markersize=1, label='inter3')

		if (inter3_incorrect3 != None) and (inter3_incorrect3 > interval[0]) and (inter3_incorrect3 < interval[1]):
			inter.append((inter3_incorrect3, 'inter3'))
			# ax1.axvline(x=inter3_incorrect3, linestyle='-.', color='tab:green', markersize=1, label='inter3')

		# Intersection between correct function and common function on a2
		inter3_common2 = find_inter3_common(common2_init[2], correct_init[0], correct_init[1], common2_init[0], common2_init[1])

		if (inter3_common2 != None) and (inter3_common2 > interval[0]) and (inter3_common2 < interval[1]):
			inter.append((inter3_common2, 'inter3'))
			# ax1.axvline(x=inter3_common2, linestyle='-.', color='tab:purple', markersize=1, label='inter3')

		# ------------------------------------------------------------------------------------------- #

		if list_w0[1] != 0:
		
			interval = [list_w0[0], list_w0[1]]

			# Correct function
			correct_abs = [fct_abs_correct(w, correct_init[0], correct_init[1], correct_init[2]) for w in interval]
			ax1.plot(interval, correct_abs, '-', color='tab:blue', markersize=1)

			# common functions
			common0_abs = [fct_abs_common(w, common0_init[0], common0_init[1], common0_init[2]) for w in interval]
			common1_abs = [fct_abs_common(w, common1_init[1], common1_init[2], common1_init[0]) for w in interval]
			common2_abs = [fct_abs_common(w, common2_init[2], common2_init[0], common2_init[1]) for w in interval]
			ax1.plot(interval, common0_abs, '-', color='tab:brown', markersize=1)
			ax1.plot(interval, common1_abs, '-', color='tab:olive', markersize=1)
			ax1.plot(interval, common2_abs, '-', color='tab:purple', markersize=1)

			# Intersections between correct function and incorrect functions	
			inter2_incorrect0 = find_inter2_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect0_init[0], incorrect0_init[1], incorrect0_init[2])
			inter2_incorrect1 = find_inter2_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect1_init[0], incorrect1_init[1], incorrect1_init[2])
			inter2_incorrect2 = find_inter2_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect2_init[0], incorrect2_init[1], incorrect2_init[2])
			inter2_incorrect3 = find_inter2_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect3_init[0], incorrect3_init[1], incorrect3_init[2])

			if (inter2_incorrect0 != None) and (inter2_incorrect0 > interval[0]) and (inter2_incorrect0 < interval[1]):
				inter.append((inter2_incorrect0, 'inter2'))
				# ax1.axvline(x=inter2_incorrect0, linestyle=':', color='tab:red', markersize=1, label='inter2')

			if (inter2_incorrect1 != None) and (inter2_incorrect1 > interval[0]) and (inter2_incorrect1 < interval[1]):
				inter.append((inter2_incorrect1, 'inter2'))
				# ax1.axvline(x=inter2_incorrect1, linestyle=':', color='tab:orange', markersize=1, label='inter2')

			if (inter2_incorrect2 != None) and (inter2_incorrect2 > interval[0]) and (inter2_incorrect2 < interval[1]):
				inter.append((inter2_incorrect2, 'inter2'))
				# ax1.axvline(x=inter2_incorrect2, linestyle=':', color='tab:pink', markersize=1, label='inter2')

			if (inter2_incorrect3 != None) and (inter2_incorrect3 > interval[0]) and (inter2_incorrect3 < interval[1]):
				inter.append((inter2_incorrect3, 'inter2'))
				# ax1.axvline(x=inter2_incorrect3, linestyle=':', color='tab:green', markersize=1, label='inter2')

			# Intersection between correct function and common function on a0
			inter2_common0 = find_inter2_common(common0_init[0], correct_init[1], correct_init[2], common0_init[1], common0_init[2])

			if (inter2_common0 != None) and (inter2_common0 > interval[0]) and (inter2_common0 < interval[1]):
				inter.append((inter2_common0, 'inter2'))
				# ax1.axvline(x=inter2_common0, linestyle=':', color='tab:brown', markersize=1, label='inter2')

			# ------------------------------------------------------------------------------------------- #

			if list_w0[0] != 0:
		
				interval = [0, list_w0[0]]

				# Correct function
				correct_abs = [fct_abs_correct(w, correct_init[0], correct_init[1], correct_init[2]) for w in interval]
				ax1.plot(interval, correct_abs, '-', color='tab:blue', markersize=1)

				# common functions
				common0_abs = [fct_abs_common(w, common0_init[0], common0_init[1], common0_init[2]) for w in interval]
				common1_abs = [fct_abs_common(w, common1_init[1], common1_init[2], common1_init[0]) for w in interval]
				common2_abs = [fct_abs_common(w, common2_init[2], common2_init[0], common2_init[1]) for w in interval]
				ax1.plot(interval, common0_abs, '-', color='tab:brown', markersize=1)
				ax1.plot(interval, common1_abs, '-', color='tab:olive', markersize=1)
				ax1.plot(interval, common2_abs, '-', color='tab:purple', markersize=1)

				# Intersections between correct function and incorrect functions
				inter1_incorrect0 = find_inter1_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect0_init[0], incorrect0_init[1], incorrect0_init[2])
				inter1_incorrect1 = find_inter1_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect1_init[0], incorrect1_init[1], incorrect1_init[2])
				inter1_incorrect2 = find_inter1_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect2_init[0], incorrect2_init[1], incorrect2_init[2])
				inter1_incorrect3 = find_inter1_incorrect(correct_init[0], correct_init[1], correct_init[2], incorrect3_init[0], incorrect3_init[1], incorrect3_init[2])

				if (inter1_incorrect0 != None) and (inter1_incorrect0 > interval[0]) and (inter1_incorrect0 < interval[1]):
					inter.append((inter1_incorrect0, 'inter1'))
					# ax1.axvline(x=inter1_incorrect0, linestyle='-', color='tab:red', markersize=1, label='inter1')

				if (inter1_incorrect1 != None) and (inter1_incorrect1 > interval[0]) and (inter1_incorrect1 < interval[1]):
					inter.append((inter1_incorrect1, 'inter1'))
					# ax1.axvline(x=inter1_incorrect1, linestyle='-', color='tab:orange', markersize=1, label='inter1')

				if (inter1_incorrect2 != None) and (inter1_incorrect2 > interval[0]) and (inter1_incorrect2 < interval[1]):
					inter.append((inter1_incorrect2, 'inter1'))
					# ax1.axvline(x=inter1_incorrect2, linestyle='-', color='tab:pink', markersize=1, label='inter1')

				if (inter1_incorrect3 != None) and (inter1_incorrect3 > interval[0]) and (inter1_incorrect3 < interval[1]):
					inter.append((inter1_incorrect3, 'inter1'))
					# ax1.axvline(x=inter1_incorrect3, linestyle='-', color='tab:green', markersize=1, label='inter1')

				# Intersections between correct function and common functions	
				inter1_common0 = find_inter1_common(common0_init[0], correct_init[1], correct_init[2], common0_init[1], common0_init[2])
				inter1_common1 = find_inter1_common(common1_init[1], correct_init[2], correct_init[0], common1_init[2], common1_init[0])
				inter1_common2 = find_inter1_common(common2_init[2], correct_init[0], correct_init[1], common2_init[0], common2_init[1])

				if (inter1_common0 != None) and (inter1_common0 > interval[0]) and (inter1_common0 < interval[1]):
					inter.append((inter1_common0, 'inter1'))
					# ax1.axvline(x=inter1_common0, linestyle='-', color='tab:brown', markersize=1, label='inter1')

				if (inter1_common1 != None) and (inter1_common1 > interval[0]) and (inter1_common1 < interval[1]):
					inter.append((inter1_common1, 'inter1'))
					# ax1.axvline(x=inter1_common1, linestyle='-', color='tab:olive', markersize=1, label='inter1')

				if (inter1_common2 != None) and (inter1_common2 > interval[0]) and (inter1_common2 < interval[1]):
					inter.append((inter1_common2, 'inter1'))
					# ax1.axvline(x=inter1_common2, linestyle='-', color='tab:purple', markersize=1, label='inter1')

	""" Test loop """
	# Results of scalar products
	scalar_000_abs = []
	scalar_001_abs = []
	scalar_010_abs = []
	scalar_011_abs = []
	scalar_100_abs = []
	scalar_101_abs = []
	scalar_110_abs = []
	scalar_111_abs = []
	
	w_list = []
	
	# Signal power consumption for a key
	S = gen_S(mu, n, kappa, K)
	
	for j in range(100):

		# Weight of signal part
		w = j / 100
		
		scalar_abs = []
	
		# i-th bit of register state studied
		for i in range(n):
			
			# Global power consumption P = (1 - w) * R + w S
			#P_i = [(r + w s) for r, s in zip(R, S[i])]	
			P_i = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S[i])]

			# Squared scalar product |<S-ref, P>|
			scalar_i = np.dot(S_ref[i], P_i)
			scalar_abs_i = abs(scalar_i)
			scalar_abs += [scalar_abs_i]
		
		# Scalar product for i = 0
		# s0 : ['_00', '_01', '_10', '_11', '_00', '_01', '_10', '_11']
		s0_abs = scalar_abs[0].tolist()
		s0_abs = s0_abs + s0_abs # Concatenate

		# Scalar product for i = 1
		# s1 : [(('0_0', '0_1'), ('0_0', '0_1')), (('1_0', '1_1'), ('1_0', '1_1'))]		
		s1_abs = [(s,s) for s in zip(scalar_abs[1][::2], scalar_abs[1][1::2])]
		s1_abs = list(itertools.chain.from_iterable(s1_abs)) 
		s1_abs = list(itertools.chain.from_iterable(s1_abs)) # Un-nest the previous list twice

		# Scalar product for i = 2
		# s2 : [('00_', '00_'), ('01_', '01_'), ('10_', '10_'), ('11_', '11_')]
		s2_abs = [s for s in zip(scalar_abs[2], scalar_abs[2])]
		s2_abs = list(itertools.chain.from_iterable(s2_abs)) # Un-nest the previous list

		# Sum of absolute scalar products for all kappa values
		scalar_abs = [s0_abs[mu] + s1_abs[mu] + s2_abs[mu] for mu in range(2**n)]
		
		w_list.append(w)
		
		scalar_000_abs.append(Decimal(scalar_abs[0]) / Decimal(24))
		scalar_001_abs.append(Decimal(scalar_abs[1]) / Decimal(24))
		scalar_010_abs.append(Decimal(scalar_abs[2]) / Decimal(24))
		scalar_011_abs.append(Decimal(scalar_abs[3]) / Decimal(24))
		scalar_100_abs.append(Decimal(scalar_abs[4]) / Decimal(24))
		scalar_101_abs.append(Decimal(scalar_abs[5]) / Decimal(24))
		scalar_110_abs.append(Decimal(scalar_abs[6]) / Decimal(24))
		scalar_111_abs.append(Decimal(scalar_abs[7]) / Decimal(24))
	
	ax1.plot(w_list, scalar_000_abs, '-', color='black', markersize=4, label='000')
	ax1.plot(w_list, scalar_001_abs, '-.', color='black',  markersize=4, label='001')
	ax1.plot(w_list, scalar_010_abs, '-.', color='black', markersize=4, label='010')
	ax1.plot(w_list, scalar_011_abs, ':' , color='black', markersize=4, label='011')
	ax1.plot(w_list, scalar_100_abs, '-.', color='black', markersize=4, label='100')
	ax1.plot(w_list, scalar_101_abs, ':', color='black',  markersize=4, label='101')
	ax1.plot(w_list, scalar_110_abs, ':', color='black', markersize=4, label='110')
	ax1.plot(w_list, scalar_111_abs, ':', color='black', markersize=4, label='111')
	

	""" Find rank of correct kappa	"""

	# Transform Decimal into float
	getcontext().prec = 5
	inter = [(float(w), string) for w, string in inter]

	# Sort by w
	inter.sort(key=lambda x: x[0])

	rank, inter = find_rank(inter)

	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	inter = [i for i in zip(inter, inter)]
		
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	inter = list(itertools.chain.from_iterable(inter))
		
	# Extract abscissas for intersection points
	inter = [tuple[0] for tuple in inter]
	inter.insert(0, 0)
	inter.append(1)

	ax2.plot(inter, rank, color='tab:blue')

	# ax1.legend(loc='upper left', ncol=3, fancybox=True)
	ax1.set_title(r'Combining three scalar products for K=%s and kappa=%s' %(K, kappa))
	ax1.set_ylabel('Scalar product |<S_ref,P>|')
	ax2.set_ylabel('Rank kappa solution')
	ax2.set_xlabel('Weight w')
	# ax2.invert_yaxis()
	ax2.set_ylim([8, 0])
	

	fig.tight_layout()
	# plt.show()
	plt.savefig('./plot/sim_1x3bits_abs_K=%s_kappa=%s_#%s.png' % (K, kappa, num_sim))
	plt.close(fig)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value)')
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
	

