#!/usr/bin/env python

"""DPA ON KECCAK 5-BIT CHI ROW & 1-BIT K

Save the figures for a simulation which display the outcome of a test
according to the SNR, in order to recover kappa bits. 

Within one simulation, we perform 5 separated experiments to guess 2 bits 
of kappa each, with the scalar product, and a fixed noise power consumption vector.  
We combine the individual results to recover the bits of kappa. 
"""

__author__  = "Anna Guinet"
__email__   = "email@annagui.net"
__version__ = "2.3"

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

def noise(mu):
	"""
	Compute a noise power consumption vector R for each message mu. 

	Parameter:
	mu -- list of Bool

	Return:
	R -- list of integers
	"""
	R = []

	# For each message mu
	for j in range(len(mu)):

		# Activity of noise part
		d = random.gauss(0, 2)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def gen_key(i, n):
	"""
	Generate all key values for key_(i+2 mod n) and key_(i+1 mod n), 
	regardless the i-th bit:
	- key_(i mod n) = kappa_i XOR K = 0
	- key_(i+1 mod n) = kappa_(i+1 mod n)
	- key_(i+2 mod n) = kappa_(i+2 mod n)
	- key_(i+3 mod n) = 0 if n = 5 
	- key_(i+4 mod n) = 0 if n = 5

	Eg: for i = 1, n = 5, key = [[ 0, 0, 0, 0, 0 ], [ 0, 0, 0, 1, 0 ], [ 0, 0, 1, 0, 0 ], [ 0, 0, 1, 1, 0 ]]
	
	Parameters:
	i -- integer in [0,n-1]
	n -- integer
	
	Return:
	key -- list of tuples of length 16
	"""

	# All possible values for 2 bits
	key = list(itertools.product([bool(0), bool(1)], repeat=2))

	# Transform nested tuples into nested lists
	key = [list(j) for j in key]

	# Insert 'False' as the i-th, (i+3)-th and (i+4)-th positions
	if n == 5:
		key_extend = [j.insert((i+3) % n, bool(0)) for j in key]
		key_extend = [j.insert((i+4) % n, bool(0)) for j in key]

	# Insert 'False' as the i-th position
	key_extend = [j.insert(i % n, bool(0)) for j in key]
	
	return key

def signal_1D(mu, i, key, n):
	"""
	Generate signal power consumption S values for all messages mu. 

	Parameters:
	mu  -- 2D list of bool, size n x 2^n
	i   -- integer in [0,n-1]
	key -- list of bool, length n
	n   -- integer

	Return:
	S -- list of integers, length 2^n
	"""
	S = []

	# For each message mu
	for j in range(len(mu)):

		# Activity of the first storage cell of the register
		d = key[i] ^ mu[j][i] ^ ((key[(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[(i+2) % n] ^ mu[j][(i+2) % n]))

		# Power consumption model
		S_j = (-1)**(d)

		S.append(S_j)

	return S
	
def signal_2D(n, i, key, mu):
	"""
	Generate signal power consumption S for all mu.
	
	'key' is a value such as key = kappa, except for bit i: 
	key_i = kappa_i XOR K
	
	Parameters:
	n   -- integer
	i   -- integer in [0,n-1]
	key -- list of bool
	mu  -- 2D list of bool, size n x 2^n
	
	Return:
	S -- 2D list of integers, size 2^n x 4
	"""

	S = []

	# For each key value
	for l in range(len(key)):

		S_l = []

		# For each message mu
		for j in range(len(mu)):
			
			# Activity of the first storage cell of the register
			d = key[l][i] ^ mu[j][i] ^ (key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n])

			# Power consumption model
			S_lj = (-1)**(d)

			S_l.append(S_lj)

		S += [S_l]

	return S

def gen_key_i(i, kappa, K):
	""" 
	Create key value where key = kappa, except for bit i: 
	key_(i mod n) = kappa_(i mod n) XOR K_(i mod n)
	
	Parameters:
	i 	  -- integer in [0,n-1]
	kappa -- string
	K     -- string
	
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

def gen_S(mu, n, kappa, K):
	"""
	Generate signal power consumption for every bit i of first chi's row, 
	for a couple (kappa, K). 
	
	Parameters:
	mu 	  -- 2D list of bool, size n x 2^n
	n 	  -- integer
	kappa -- string
	K 	  -- string
	
	Return:
	S -- 2D list of integers, size 2^n x n
	"""
	S = []
	
	for i in range(n):
		key = gen_key_i(i, kappa, K) # Generate the key with XOR at the i-th position
		S_i = signal_1D(mu, i, key, n)
		S += [S_i]
	
	return S

def gen_S_ref(mu, n):
	"""
	Generate all signal power consumption for every bit i of first chi's row. 

	Parameters:
	mu -- list of bool, size n x 2^n
	n  -- integer

	Return:
	S_ref -- list of integers, size 2^n x 2^(n-1) x n
	"""
	S_ref = []
	
	for i in range(n):
		key = gen_key(i, n)
		S_ref_i = signal_2D(n, i, key, mu)
		S_ref += [S_ref_i]

	return S_ref

def gen_scalar_init(n, R, S_ref):
	"""
	Generate initial scalar products for the chi row. 

	Parameters:
	n 	  -- integer
	R 	  -- list
	S_ref -- list of integers

	Return:
	scalar_init -- list
	"""
	scalar_init = [] 

	# i-th bit of register state studied
	for i in range(n):
				
		# Global power consumption P_init = R
		getcontext().prec = 8
		P_init_i = [Decimal(r) for r in R]

		# Scalar product <S-ref, P>
		scalar_init_i = np.dot(S_ref[i], P_init_i)
		scalar_init += [scalar_init_i]

	return scalar_init

def kappa_idx():
	"""
	Provide scalar products indexes for kappa values.

	Return:
	list_kappa_idx -- list of lists
	"""

	# List of indexes
	list_kappa_idx00 = [[0, 0, 0, 0, 0, '00000'],
						[0, 0, 1, 1, 0, '00001'],
						[0, 1, 2, 0, 0, '00010'],
						[0, 1, 3, 1, 0, '00011'],
						[1, 2, 0, 0, 0, '00100'],
						[1, 2, 1, 1, 0, '00101'],
						[1, 3, 2, 0, 0, '00110'],
						[1, 3, 3, 1, 0, '00111']]

	list_kappa_idx01 = [[2, 0, 0, 0, 1, '01000'],
						[2, 0, 1, 1, 1, '01001'],
						[2, 1, 2, 0, 1, '01010'],
						[2, 1, 3, 1, 1, '01011'],
						[3, 2, 0, 0, 1, '01100'],
						[3, 2, 1, 1, 1, '01101'],
						[3, 3, 2, 0, 1, '01110'],
						[3, 3, 3, 1, 1, '01111']]

	list_kappa_idx10 = [[0, 0, 0, 2, 2, '10000'],
						[0, 0, 1, 3, 2, '10001'],
						[0, 1, 2, 2, 2, '10010'],
						[0, 1, 3, 3, 2, '10011'],
						[1, 2, 0, 2, 2, '10100'],
						[1, 2, 1, 3, 2, '10101'],
						[1, 3, 2, 2, 2, '10110'],
						[1, 3, 3, 3, 2, '10111']]
		
	list_kappa_idx11 = [[2, 0, 0, 2, 3, '11000'],
						[2, 0, 1, 3, 3, '11001'],
						[2, 1, 2, 2, 3, '11010'],
						[2, 1, 3, 3, 3, '11011'],
						[3, 2, 0, 2, 3, '11100'],
						[3, 2, 1, 3, 3, '11101'],
						[3, 3, 2, 2, 3, '11110'],
						[3, 3, 3, 3, 3, '11111']]

	list_kappa_idx = list_kappa_idx00 + list_kappa_idx01 + list_kappa_idx10 + list_kappa_idx11

	return list_kappa_idx

def xor_secret_i(K, kappa, i):
	"""
	Compute power consumption at bit i = kappa_i XOR K_i

	Parameters:
	K 	  -- string
	kappa -- string
	i 	  -- integer

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

def xor_solution(n, K, kappa, solution_init):
	"""
	Sign scalar products for solution function.

	Parameters:
	n 			  -- integer
	K 			  -- string
	kappa 		  -- string
	solution_init -- list of Decimal

	Return:
	solution_init -- list of Decimal
	"""	
	for i in range(n):
		p_i = xor_secret_i(K, kappa, i)

		solution_init[i] = solution_init[i] * p_i

def xor_common_i(i, b, n, K, kappa, common_init):
	"""
	Sign scalar products for functions with b common scalar products.

	Parameters:
	i 			-- integer
	b 			-- integer
	n 			-- integer
	K 			-- string
	kappa 		-- string
	common_init -- list of Decimal

	Return:
	common_init -- list of Decimal
	"""	
	for j in range(i, i + b):
		p_j = xor_secret_i(K, kappa, (j % n))

		common_init[(j % n)] = common_init[(j % n)] * p_j

	return common_init

def xor_common(b, n, K, kappa, common_init):
	"""
	Sign iteratively initial scalar products 
	for functions with b common scalar products.

	Parameters:
	b 			-- integer
	n 			-- integer
	K 			-- string
	kappa 		-- string
	common_init -- list of Decimal

	Return:
	None
	"""	
	# Common4b or common3b
	if len(common_init) == 5:

		for j in range(len(common_init)):
				common_init[j] = xor_common_i(j, b, n, K, kappa, common_init[j])

	# Common2b
	if len(common_init) == 10:
		for j in range(5):
			for l in range(2):
				common_init[2*j + l] = xor_common_i(j, b, n, K, kappa, common_init[2*j + l])

def find_idx_common4b(i, kappa, list_kappa_idx, solution_idx):
	"""
	Find the scalar products for common function with 4 solution consecutive kappas. 

	Parameters:
	i 			   -- integer
	kappa 		   -- string
	list_kappa_idx -- list
	solution_idx   -- list of Decimal

	Return:
	common4b_idx -- list
	"""	

	common4b_idx = [sublist for sublist in list_kappa_idx if (sublist[i % 5] == solution_idx[i % 5]) and (sublist[(i + 1) % 5] == solution_idx[(i + 1) % 5]) and (sublist[(i + 2) % 5] == solution_idx[(i + 2) % 5]) and (sublist[5] != kappa)]
	
	# Un-nest the previous list
	common4b_idx = list(itertools.chain.from_iterable(common4b_idx)) 

	return common4b_idx

def find_idx_common3b(i, list_kappa_idx, solution_idx):
	"""
	Find the scalar products for common function with 3 solution consecutive kappas. 

	Parameters:
	i 			   -- integer
	list_kappa_idx -- list
	solution_idx   -- list of Decimal

	Return:
	common3b_idx -- list
	"""	

	common3b_idx = [sublist for sublist in list_kappa_idx if (sublist[(i - 1) % 5] != solution_idx[(i - 1) % 5]) and (sublist[i % 5] == solution_idx[i % 5]) and (sublist[(i + 1) % 5] == solution_idx[(i + 1) % 5]) and (sublist[(i + 2) % 5] != solution_idx[(i + 2) % 5])]

	# Un-nest the previous list
	common3b_idx = list(itertools.chain.from_iterable(common3b_idx)) 

	return common3b_idx

def find_idx_common2b(i, list_kappa_idx, solution_idx):
	"""
	Find the scalar products for common function with 2 solution consecutive kappas. 

	Parameters:
	i 			   -- integer
	list_kappa_idx -- list
	solution_idx   -- list of Decimal

	Return:
	common2b_idx0 -- list
	common2b_idx1 -- list
	"""	

	common2b_idx = [sublist for sublist in list_kappa_idx if (sublist[(i - 1) % 5] != solution_idx[(i - 1) % 5]) and (sublist[i % 5] == solution_idx[i % 5]) and (sublist[(i + 1) % 5] != solution_idx[(i + 1) % 5])]

	common2b_idx0 = common2b_idx[0]
	common2b_idx1 = common2b_idx[1]

	return common2b_idx0, common2b_idx1

def find_idx_nonsol(list_kappa_idx, solution_idx):
	"""
	Find the scalar products for nonsolutions. 

	Parameters:
	list_kappa_idx -- list
	solution_idx   -- list of Decimal

	Return:
	nonsol_idx -- list
	"""	

	nonsol_idx = [sublist for sublist in list_kappa_idx if (sublist[0] != solution_idx[0]) and (sublist[1] != solution_idx[1]) and (sublist[2] != solution_idx[2]) and (sublist[3] != solution_idx[3]) and (sublist[4] != solution_idx[4])  and (sublist[4] != solution_idx[4])]

	return nonsol_idx

def find_init(n, init_scalar, list_idx):
	"""
	Find the list scalar products for functions from init_scalar. 

	Parameters:
	n 			-- integer
	init_scalar -- list of Decimal
	list_idx    -- list of list

	Return:
	list_init -- list of Decimal
	"""	
	list_init = []

	for j in range(len(list_idx)):
		init = [init_scalar[i][list_idx[j][i]] for i in range(n)]
		list_init.append(init)

	return list_init

def find_abs_w0(solution_init, norm):
	"""
	Find the fiber points for each DoM scalar product function.

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal

	Return:
	list_tup_w0 -- list of Decimal
	"""	
	list_tup_w0 = []

	for a in solution_init:
		if (a != norm):
			w0 = a / (a - norm)

			if (w0 > 0) & (w0 < 1):
				list_tup_w0.append((w0, a))
			else:
				list_tup_w0.append((0, a)) # by default 0 for the loop afterwards
		else:
			list_tup_w0.append((0, a)) # by default 0 for the loop afterwards
	
	return list_tup_w0 #(w0, scalar_product)

def fct_abs_solution(w, n, norm, solution_init):
	"""
	Function for the scalar product of the solution.
	
	Parameters:
	w 			  -- float
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	
	Return:
	y -- Decimal
	"""
	w = Decimal(w)

	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	y0 = (norm - a0) * w + a0
	y1 = (norm - a1) * w + a1
	y2 = (norm - a2) * w + a2
	y3 = (norm - a3) * w + a3
	y4 = (norm - a4) * w + a4
	
	y = abs(y0) + abs(y1) + abs(y2) + abs(y3) + abs(y4)

	return y / (Decimal(n) * norm)

def fct_abs_common4(w, i, n, norm, common4b_init):
	"""
	Function for the scalar product of a hypothesis which has 
	four consecutive kappa bits in common with the solution (a0, a1, a2).

	Parameters:
	w 			  -- float
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	common4b_init -- list of Decimal

	Return:
	y -- Decimal
	"""	
	w = Decimal(w)

	a0 = common4b_init[i % n]
	a1 = common4b_init[(i + 1) % n]
	a2 = common4b_init[(i + 2) % n]
	b  = common4b_init[(i + 3) % n]
	c  = common4b_init[(i + 4) % n]

	y_a0 = (norm - a0) * w + a0
	y_a1 = (norm - a1) * w + a1
	y_a2 = (norm - a2) * w + a2
	y_b = - b * w + b
	y_c = - c * w + c

	y = abs(y_a0) + abs(y_a1) + abs(y_a2) + abs(y_b) + abs(y_c)

	return y / (Decimal(n) * norm)

def fct_abs_common3(w, i, n, norm, common3b_init):
	"""
	Function for the scalar product of a hypothesis which has 
	three consecutive kappa bits in common with the solution (a0, a1).

	Parameters:
	w 			  -- float
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	common3b_init -- list of Decimal

	Return:
	y -- Decimal
	"""	
	w = Decimal(w)

	a0 = common3b_init[i % n]
	a1 = common3b_init[(i + 1) % n]
	b  = common3b_init[(i + 2) % n]
	c  = common3b_init[(i + 3) % n]
	d  = common3b_init[(i + 4) % n]

	y_a0 = (norm - a0) * w + a0
	y_a1 = (norm - a1) * w + a1
	y_b = - b * w + b
	y_c = - c * w + c
	y_d = - d * w + d

	y = abs(y_a0) + abs(y_a1) + abs(y_b) + abs(y_c) + abs(y_d)

	return y / (Decimal(n) * norm)

def fct_abs_common2(w, i, n, norm, common2b_init):
	"""
	Function for the scalar product of a hypothesis which has 
	two consecutive kappa bits in common with the solution (a).
	
	Parameters:
	w 			  -- float
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	common2b_init -- list of Decimal

	Return:
	y -- Decimal
	"""	
	w = Decimal(w)

	a = common2b_init[i % n]
	b = common2b_init[(i + 1) % n]
	c = common2b_init[(i + 2) % n]
	d = common2b_init[(i + 3) % n]
	e = common2b_init[(i + 4) % n]

	y_a = (norm - a) * w + a
	y_b = - b * w + b
	y_c = - c * w + c
	y_d = - d * w + d
	y_e = - e * w + e 

	y = abs(y_a) + abs(y_b) + abs(y_c) + abs(y_d) + abs(y_e)

	return y / (Decimal(n) * norm)

def fct_abs_nonsol(w, n, norm, nonsol_init):
	"""
	Function for the scalar product of a nonsolution.

	Parameters:
	w 			-- float
	n 			-- integer
	norm 		-- Decimal
	nonsol_init -- list of Decimal

	Return:
	y -- Decimal
	"""	
	w = Decimal(w)

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	y_b = - b * w + b
	y_c = - c * w + c
	y_d = - d * w + d
	y_e = - e * w + e
	y_f = - f * w + f

	y = abs(y_b) + abs(y_c) + abs(y_d) + abs(y_e) + abs(y_f)
		
	return y / (Decimal(n) * norm)

def xor_solution_p_i(n, K, kappa, solution_init):
	"""
	Sign scalar products.

	Parameters:
	n 			  -- integer
	K 			  -- string
	kappa 		  -- string
	solution_init -- list of Decimal

	Return:
	solution_init -- list of Decimal
	"""	

	for i in range(n):
		p_i = xor_secret_i(K, kappa, i)

		solution_init[i] = solution_init[i] * p_i 

	return solution_init

def xor_common_p_i(i, b, n, K, kappa, common_init):
	"""
	Sign scalar products for functions with b common scalar products.

	Parameters:
	i 			-- integer
	b 			-- integer
	n 			-- integer
	K 			-- string
	kappa 		-- string
	common_init -- list of Decimal

	Return:
	common_init -- list of Decimal
	"""	
	for j in range(i, i + b):
		p_j = xor_secret_i(K, kappa, (j % n))

		common_init[(j % n)] = common_init[(j % n)] * p_j

	return common_init

def reorder_common4b(solution_init, common4b_init):
	"""
	Reorder elements in common4b_init according to the sorted solution_init.

	Parameters:
	solution_init -- list of Decimal
	common4b_init -- list of lists of Decimal

	Return:
	common4b_new -- list of lists of Decimal
	"""	
	aux = []

	n = len(solution_init)

	for i in range(n):

		# Select two non common scalar products
		non_sp =  [common4b_init[i][(i+3) % n]] + [common4b_init[i][(i+4) % n]] 

		# Select three common scalar products
		sol_sp = [common4b_init[i][i]] + [common4b_init[i][(i+1) % n]] + [common4b_init[i][(i+2) % n]]
		
		aux += [[sol_sp] + [non_sp]]

		# Retrieve new idx of common scalar products
		idx0 = solution_init.index(common4b_init[i][i])
		idx1 = solution_init.index(common4b_init[i][(i+1)%n])
		idx2 = solution_init.index(common4b_init[i][(i+2)%n])

		list_idx = [idx0, idx1, idx2]
		
		# Sort scalar products according to sorted correct_init
		to_sort = [(idx, sp) for idx, sp in zip(list_idx, aux[i][0])]
		to_sort = sorted(to_sort, key=lambda x: x[0])

		# Reinsert sorted common scalar products
		aux[i][-1].insert(to_sort[0][0], to_sort[0][1])
		aux[i][-1].insert(to_sort[1][0], to_sort[1][1])
		aux[i][-1].insert(to_sort[2][0], to_sort[2][1])

	common4b_new = [aux[i][-1] for i in range(n)]

	return common4b_new

def reorder_common3b(solution_init, common3b_init):
	"""
	Reorder elements in common3b_init according to the sorted solution_init.

	Parameters:
	solution_init -- list of Decimal
	common3b_init -- list of lists of Decimal

	Return:
	common3b_new -- list of lists of Decimal
	"""	
	aux = []

	n = len(solution_init)

	for i in range(n):

		# Select three non common scalar products
		non_sp =  [common3b_init[i][(i+2)%n]] + [common3b_init[i][(i+3)%n]] + [common3b_init[i][(i+4)%n]] 

		# Select two common scalar products
		sol_sp = [common3b_init[i][i]] + [common3b_init[i][(i+1)%n]]

		aux += [[sol_sp] + [non_sp]]

		# Retrieve new idx of common scalar products
		idx0 = solution_init.index(common3b_init[i][i])
		idx1 = solution_init.index(common3b_init[i][(i+1)%n])

		list_idx = [idx0, idx1]
		
		# Sort scalar products according to sorted correct_init
		to_sort = [(idx, sp) for idx, sp in zip(list_idx, aux[i][0])]
		to_sort = sorted(to_sort, key=lambda x: x[0])

		# Reinsert sorted common scalar products
		aux[i][-1].insert(to_sort[0][0], to_sort[0][1])
		aux[i][-1].insert(to_sort[1][0], to_sort[1][1])

	common3b_new = [aux[i][-1] for i in range(n)]

	return common3b_new

def reorder_common2b(solution_init, common2b_init):
	"""
	Reorder elements in common3b_init according to the sorted solution_init.

	Parameters:
	solution_init -- list of Decimal
	common2b_init -- list of lists of Decimal

	Return:
	common2b_new -- list of lists of Decimal
	"""	

	common2b_new = []

	n = len(solution_init)

	for i in range(n):
		common2b_new.append((common2b_init[i][:i] + common2b_init[i][i+1:])) # rm elem i in list

		# Retrieve idx of common scalar products
		idx = solution_init.index(common2b_init[i][i])

		common2b_new[i].insert(idx, common2b_init[i][i])

	return common2b_new

def find_wr6_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an nonsol.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_4, 1] 
	with 0 =< w0_4 < 1?
	
	fct_solution(w) = (5*norm - A) w + A
	fct_nonsol(w) = -F w + F
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w6 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)
	f = abs(f)
	
	w6 = Decimal()

	A = a0 + a1 + a2 + a3 + a4
	F = b + c + d + e + f
	
	if (Decimal(5) * norm + F - A) != 0:
		w6 = (F - A) / (Decimal(5) * norm + F - A)
	else:
		w6 = None

	return w6

def find_wr5_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an nonsol.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_3, w0_4] 
	with 0 =< w0_3 < w0_4 < 1?
	
	fct_solution(w) = (3*norm - E) w + E
	fct_nonsol(w) = -F w + F
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w5 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)
	f = abs(f)
	
	w5 = Decimal()

	E = a0 + a1 + a2 + a3 - a4
	F = b + c + d + e + f
	
	if (Decimal(3) * norm + F - E) != 0:
		w5 = (F - E) / (Decimal(3) * norm + F - E)
	else:
		w5 = None

	return w5

def find_wr4_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an nonsol.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_2, w0_3] 
	with 0 =< w0_2 < w0_3 < w0_4 < 1?
	
	fct_solution(w) = (norm - D) w + D
	fct_nonsol(w) = -F w + F
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)
	f = abs(f)
	
	w4 = Decimal()

	D = a0 + a1 + a2 - a3 - a4
	F = b + c + d + e + f
	
	if (norm + F - D) != 0:
		w4 = (F - D) / (norm + F - D)
	else:
		w4 = None

	return w4

def find_wr3_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an nonsol.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_1, w0_2] 
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1?
	
	fct_solution(w) = (C - norm) w - C
	fct_nonsol(w) = -F w + F
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w3 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)
	f = abs(f)
	
	w3 = Decimal()

	C = a2 + a3 + a4 - a0 - a1
	F = b + c + d + e + f
	
	if (C + F - norm) != 0:
		w3 = (C + F) / (C + F - norm)
	else:
		w3 = None

	return w3

def find_wr2_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an nonsol.

	Fct_nonsol(w) = fct_solution(w), for which w in [w0_0, w0_1] 
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1?
	
	fct_solution(w) = (B - 3*norm) w - B
	fct_nonsol(w) = -F w + F
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w2 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)
	f = abs(f)
	
	w2 = Decimal()

	B = a1 + a2 + a3 + a4 - a0
	F = b + c + d + e + f
	
	if (B + F - Decimal(3)*norm) != 0:
		w2 = (B + F) / (B + F - Decimal(3)*norm)
	else:
		w2 = None

	return w2

def find_wr1_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an nonsol.

	Fct_nonsol(w) = fct_solution(w), for which w in [0, w0_0] 
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1?
	
	fct_solution(w) = (A - 5*norm) w - A
	fct_nonsol(w) = -F w + F
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w1 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	b = nonsol_init[0]
	c = nonsol_init[1]
	d = nonsol_init[2]
	e = nonsol_init[3]
	f = nonsol_init[4]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)
	f = abs(f)
	
	w1 = Decimal()

	A = a0 + a1 + a2 + a3 + a4
	F = b + c + d + e + f
	
	if (A + F - Decimal(5)*norm) != 0:
		w1 = (A + F) / (A + F - Decimal(5)*norm)
	else:
		w1 = None

	return w1

def find_wr6_common4(i, n, norm, solution_init, common4b_init):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_4, 1]
	with 0 =< w0_4 < 1 ?
	
	fct_solution(w) = (5*norm - A) w + A
	fct_common4b(w) = (3*norm - J) w + J

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_init -- list of Decimal
	
	Return:
	w6 -- Decimal
	"""	
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai  = common4b_init[i % n]
	ai1 = common4b_init[(i + 1) % n]
	ai2 = common4b_init[(i + 2) % n]
	b   = common4b_init[(i + 3) % n]
	c   = common4b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)

	A = a0 + a1 + a2 + a3 + a4
	J = ai + ai1 + ai2 + b + c

	w6 = Decimal()
	
	if (Decimal(2) * norm + J - A) != 0:
		w6 = (J - A) / (Decimal(2) * norm + J - A)
	else:
		w6 = None

	return w6

def find_wr5_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_3, w0_4]
	with 0 =< w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (3*norm - E) w + E
	fct_common4b(w) = (norm - I) w + I

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_inc  -- list of Decimal
	common4b_dec  -- list of Decimal
	common4b_non  -- list of Decimal
	
	Return:
	w5 -- Decimal
	"""	
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common4b_inc) == 2

	ai  = common4b_inc[0]
	ai1 = common4b_inc[1]
	ai2 = common4b_dec
	b   = common4b_non[0]
	c   = common4b_non[1]

	b = abs(b)
	c = abs(c)

	E = a0 + a1 + a2 + a3 - a4
	I = ai + ai1 - ai2 + b + c

	w5 = Decimal()
	
	if (Decimal(2) * norm + I - E) != 0:
		w5 = (I - E) / (Decimal(2) * norm + I - E)
	else:
		w5 = None

	return w5

def find_wr4_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_2, w0_3]
	with 0 =< w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (norm - D) w + D
	fct_common4b(w) = (H - norm) w - H

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_inc  -- list of Decimal
	common4b_dec  -- list of Decimal
	common4b_non  -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common4b_dec) == 2

	ai  = common4b_inc[0]
	ai1 = common4b_dec[0]
	ai2 = common4b_dec[1]
	b   = common4b_non[0]
	c   = common4b_non[1]

	b = abs(b)
	c = abs(c)

	D = a0 + a1 + a2 - a3 - a4
	H = ai1 + ai2 - ai - b - c

	w4 = Decimal()
	
	if (D + H - Decimal(2) * norm) != 0:
		w4 = (D + H) / (D + H - Decimal(2) * norm)
	else:
		w4 = None

	return w4

def find_wr4bis_common4(norm, solution_init, common4b_inc, common4b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_2, w0_3]
	with 0 =< w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (norm - D) w + D
	fct_common4b(w) = (3*norm - J) w + J

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_inc  -- list of Decimal
	common4b_non  -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common4b_inc) == 3

	ai  = common4b_inc[0]
	ai1 = common4b_inc[1]
	ai2 = common4b_inc[2]
	b   = common4b_non[0]
	c   = common4b_non[1]

	b = abs(b)
	c = abs(c)

	D = a0 + a1 + a2 - a3 - a4
	J = ai + ai1 + ai2 + b + c

	w4 = Decimal()
	
	if (J - D - Decimal(2) * norm) != 0:
		w4 = (J - D) / (J - D - Decimal(2) * norm)
	else:
		w4 = None

	return w4

def find_wr3_common4(norm, solution_init, common4b_dec, common4b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_1, w0_2]
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (C - norm) w - C
	fct_common4b(w) = (G - 3*norm) w - G

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_dec  -- list of Decimal
	common4b_non  -- list of Decimal
	
	Return:
	w3 -- Decimal
	"""	
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common4b_dec) == 3

	ai  = common4b_dec[0]
	ai1 = common4b_dec[1]
	ai2 = common4b_dec[2]
	b   = common4b_non[0]
	c   = common4b_non[1]

	b = abs(b)
	c = abs(c)

	C = a2 + a3 + a4 - a0 - a1
	G = ai + ai1 + ai2 - b - c

	w3 = Decimal()
	
	if (C - G + Decimal(2) * norm) != 0:
		w3 = (C - G) / (C - G + Decimal(2) * norm)
	else:
		w3 = None

	return w3

def find_wr3bis_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_1, w0_2]
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (C - norm) w - C
	fct_common4b(w) = (norm - I) w + I

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_inc  -- list of Decimal
	common4b_dec  -- list of Decimal
	common4b_non  -- list of Decimal
	
	Return:
	w5 -- Decimal
	"""	
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common4b_inc) == 2
	assert len(common4b_dec) == 1

	ai  = common4b_inc[0]
	ai1 = common4b_inc[1]
	ai2 = common4b_dec[0]
	b   = common4b_non[0]
	c   = common4b_non[1]

	b = abs(b)
	c = abs(c)

	C = a2 + a3 + a4 - a0 - a1
	I = ai + ai1 - ai2 + b + c

	w3 = Decimal()
	
	if (C + I - Decimal(2) * norm) != 0:
		w3 = (C + I) / (C + I - Decimal(2) * norm)
	else:
		w3 = None

	return w3

def find_wr2_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [w0_0, w0_1]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (B - 3*norm) w - B
	fct_common4b(w) = (H - norm) w - H

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_inc  -- list of Decimal
	common4b_dec  -- list of Decimal
	common4b_non  -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common4b_inc) == 1
	assert len(common4b_dec) == 2

	ai  = common4b_inc[0]
	ai1 = common4b_dec[0]
	ai2 = common4b_dec[1]
	b   = common4b_non[0]
	c   = common4b_non[1]

	b = abs(b)
	c = abs(c)

	B = a1 + a2 + a3 + a4 - a0
	H = ai1 + ai2 - ai - b - c

	w2 = Decimal()
	
	if (B - H - Decimal(2) * norm) != 0:
		w2 = (B - H) / (B - H - Decimal(2) * norm)
	else:
		w2 = None

	return w2

def find_wr1_common4(i, n, norm, solution_init, common4b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 4 consecutive bits in common.

	Fct_common4b(w) = fct_solution(w), for which w in [0, w0_0]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (5*norm - A) w + A
	fct_common4b(w) = (G - 3*norm) w - G

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_init -- list of Decimal
	
	Return:
	w1 -- Decimal
	"""	
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai  = common4b_init[i % n]
	ai1 = common4b_init[(i + 1) % n]
	ai2 = common4b_init[(i + 2) % n]
	b   = common4b_init[(i + 3) % n]
	c   = common4b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)

	A = a0 + a1 + a2 + a3 + a4
	G = ai + ai1 + ai2 - b - c

	w1 = Decimal()
	
	if (A - G - Decimal(2) * norm) != 0:
		w1 = (A - G) / (A - G - Decimal(2) * norm)
	else:
		w1 = None

	return w1

def find_wr6_common3(i, n, norm, solution_init, common3b_init):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_4, 1]
	with 0 =< w0_4 < 1 ?
	
	fct_solution(w) = (5*norm - A) w + A
	fct_common3b(w) = (2*norm - M) w + M

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_init -- list of Decimal

	Return:
	w6 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai  = common3b_init[i % n]
	ai1 = common3b_init[(i + 1) % n]
	b   = common3b_init[(i + 2) % n]
	c   = common3b_init[(i + 3) % n]
	d   = common3b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	A = a0 + a1 + a2 + a3 + a4
	M = ai + ai1 + b + c + d

	w6 = Decimal()
	
	if (M - A + Decimal(3) * norm) != 0:
		w6 = (M - A) / (M - A + Decimal(3) * norm)
	else:
		w6 = None

	return w6

def find_wr5_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_3, w0_4]
	with 0 =< w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (3*norm - E) w + E
	fct_common3b(w) = - L w + L

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w5 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_inc) == 1

	ai  = common3b_inc[0]
	ai1 = common3b_dec
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	E = a0 + a1 + a2 + a3 - a4
	L = ai - ai1 + b + c + d

	w5 = Decimal()
	
	if (L - E + Decimal(3) * norm) != 0:
		w5 = (L - E) / (L - E + Decimal(3) * norm)
	else:
		w5 = None

	return w5

def find_wr5bis_common3(norm, solution_init, common3b_inc, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_3, w0_4]
	with 0 =< w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (3*norm - E) w + E
	fct_common3b(w) = (2*norm - M) w + M

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w5 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_inc) == 2

	ai  = common3b_inc[0]
	ai1 = common3b_inc[1]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	E = a0 + a1 + a2 + a3 - a4
	M = ai + ai1 + b + c + d

	w5 = Decimal()
	
	if (M - E + norm) != 0:
		w5 = (M - E) / (M - E + norm)
	else:
		w5 = None

	return w5

def find_wr4_common3(norm, solution_init, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_2, w0_3]
	with 0 =< w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (norm - D) w + D
	fct_common3b(w) = (K - 2*norm) w - K

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_dec) == 2

	ai  = common3b_dec[0]
	ai1 = common3b_dec[1]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	D = a0 + a1 + a2 - a3 - a4
	K = ai + ai1 - b - c - d

	w4 = Decimal()
	
	if (D + K - Decimal(3)*norm) != 0:
		w4 = (D + K) / (D + K - Decimal(3)*norm)
	else:
		w4 = None

	return w4

def find_wr4bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_2, w0_3]
	with 0 =< w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (norm - D) w + D
	fct_common3b(w) = - L w + L

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_inc) == 1
	assert len(common3b_dec) == 1

	ai  = common3b_inc[0]
	ai1 = common3b_dec[0]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	D = a0 + a1 + a2 - a3 - a4
	L = ai - ai1 + b + c + d

	w4 = Decimal()
	
	if (L - D + norm) != 0:
		w4 = (L - D) / (L - D + norm)
	else:
		w4 = None

	return w4

def find_wr4ter_common3(norm, solution_init, common3b_inc, common3b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_2, w0_3]
	with 0 =< w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (norm - D) w + D
	fct_common3b(w) = (2*norm - M) w + M

	Parameters:
	norm	      -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_inc) == 2

	ai  = common3b_inc[0]
	ai1 = common3b_inc[1]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	D = a0 + a1 + a2 - a3 - a4
	M = ai + ai1 + b + c + d

	w4 = Decimal()
	
	if (M - D - norm) != 0:
		w4 = (M - D) / (M - D - norm)
	else:
		w4 = None

	return w4

def find_wr3_common3(norm, solution_init, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_1, w0_2]
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (C - norm) w + C
	fct_common3b(w) = (K - 2*norm) w - K

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w3 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_dec) == 2

	ai  = common3b_dec[0]
	ai1 = common3b_dec[1]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	C = a2 + a3 + a4 - a0 - a1
	K = ai + ai1 - b - c - d

	w3 = Decimal()
	
	if (C - K + norm) != 0:
		w3 = (C - K) / (C - K + norm)
	else:
		w3 = None

	return w3

def find_wr3bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_1, w0_2]
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1 ?

	fct_solution(w) = (C - norm) w - C
	fct_common3b(w) = - L w + L

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w3 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_inc) == 1
	assert len(common3b_dec) == 1

	ai  = common3b_inc[0]
	ai1 = common3b_dec[0]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	C = a2 + a3 + a4 - a0 - a1
	L = ai - ai1 + b + c + d

	w3 = Decimal()
	
	if (L + C - norm) != 0:
		w3 = (L + C) / (L + C - norm)
	else:
		w3 = None

	return w3

def find_wr3ter_common3(norm, solution_init, common3b_inc, common3b_non):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_1, w0_2]
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (C - norm) w - C
	fct_common3b(w) = (2*norm - M) w + M

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w3 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_inc) == 2

	ai  = common3b_inc[0]
	ai1 = common3b_inc[1]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	C = a2 + a3 + a4 - a0 - a1
	M = ai + ai1 + b + c + d

	w3 = Decimal()
	
	if (M + C - Decimal(3)*norm) != 0:
		w3 = (M + C) / (M + C - Decimal(3)*norm)
	else:
		w3 = None

	return w3

def find_wr2bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_0, w0_1]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (B - 3*norm) w - B
	fct_common3b(w) = - L w + L

	Parameters:
	norm		  -- Decimal
	solution_init -- list of Decimal
	common3b_inc  -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w2 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_dec) == 1

	ai  = common3b_inc
	ai1 = common3b_dec[0]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	B = a1 + a2 + a3 + a4 - a0
	L = ai - ai1 + b + c + d

	w2 = Decimal()
	
	if (B + L - Decimal(3)*norm) != 0:
		w2 = (B + L) / (B + L - Decimal(3)*norm)
	else:
		w2 = None

	return w2

def find_wr2_common3(norm, solution_init, common3b_dec, common3b_non):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [w0_0, w0_1]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (B - 3*norm) w - B
	fct_common3b(w) = (K - 2*norm) w - K

	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_dec  -- list of Decimal
	common3b_non  -- list of Decimal

	Return:
	w2 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	assert len(common3b_dec) == 2

	ai  = common3b_dec[0]
	ai1 = common3b_dec[1]
	b   = common3b_non[0]
	c   = common3b_non[1]
	d   = common3b_non[2]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	B = a1 + a2 + a3 + a4 - a0
	K = ai + ai1 - b - c - d

	w2 = Decimal()
	
	if (B - K - norm) != 0:
		w2 = (B - K) / (B - K - norm)
	else:
		w2 = None

	return w2

def find_wr1_common3(i, n, norm, solution_init, common3b_init):
	"""	
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 3 consecutive bits in common.

	Fct_common3b(w) = fct_solution(w), for which w in [0 , w0_0]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (A - 5*norm) w - A
	fct_common3b(w) = (K - 2*norm) w - K

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_init -- list of Decimal

	Return:
	w1 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai  = common3b_init[i % n]
	ai1 = common3b_init[(i + 1) % n]
	b   = common3b_init[(i + 2) % n]
	c   = common3b_init[(i + 3) % n]
	d   = common3b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)

	A = a0 + a1 + a2 + a3 + a4
	K = ai + ai1 - b - c - d

	w1 = Decimal()
	
	if (A - K - Decimal(3)*norm) != 0:
		w1 = (A - K) / (A - K - Decimal(3)*norm)
	else:
		w1 = None

	return w1

def find_wr6_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_4 , 1]
	with 0 =< w0_4 < 1 ?
	
	fct_solution(w) = (5*norm - A) w + A
	fct_common2b(w) = (norm - P) w + P

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w6 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	A = a0 + a1 + a2 + a3 + a4
	P = ai + b + c + d + e

	w6 = Decimal()
	
	if (P - A + Decimal(4)*norm) != 0:
		w6 = (P - A) / (P - A + Decimal(4)*norm)
	else:
		w6 = None

	return w6

def find_wr5_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_3 , w0_4]
	with 0 =< w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (3*norm - E) w + E
	fct_common2b(w) = (N - norm) w - N

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w5 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	E = a0 + a1 + a2 + a3 - a4
	N = ai - b - c - d - e

	w5 = Decimal()
	
	if (E + N - Decimal(2)*norm) != 0:
		w5 = (E + N) / (E + N - Decimal(2)*norm)
	else:
		w5 = None

	return w5

def find_wr5bis_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_3 , w0_4]
	with 0 =< w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (3*norm - E) w + E
	fct_common2b(w) = (norm - P) w + P

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w5 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	E = a0 + a1 + a2 + a3 - a4
	P = ai + b + c + d + e

	w5 = Decimal()
	
	if (P - E + Decimal(2)*norm) != 0:
		w5 = (P - E) / (P - E + Decimal(2)*norm)
	else:
		w5 = None

	return w5

def find_wr4_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_2, w0_3]
	with 0 =< w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (norm - D) w + D
	fct_common2b(w) = (N - norm) w - N

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w4 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	D = a0 + a1 + a2 - a3 - a4
	N = ai - b - c - d - e

	w4 = Decimal()
	
	if (D + N - Decimal(2)*norm) != 0:
		w4 = (D + N) / (D + N - Decimal(2)*norm)
	else:
		w4 = None

	return w4

def find_wr3_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_1, w0_2]
	with 0 =< w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (C - norm) w - C
	fct_common2b(w) = (norm - P) w + P

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w3 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	C = a2 + a3 + a4 - a0 - a1
	P = ai + b + c + d + e

	w3 = Decimal()
	
	if (C + P - Decimal(2)*norm) != 0:
		w3 = (C + P) / (C + P - Decimal(2)*norm)
	else:
		w3 = None

	return w3

def find_wr2bis_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_0, w0_1]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (B - 3*norm) w - B
	fct_common2b(w) = (norm - P) w + P

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w2 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	B = a1 + a2 + a3 + a4 - a0
	P = ai + b + c + d + e

	w2 = Decimal()
	
	if (B + P - Decimal(4)*norm) != 0:
		w2 = (B + P) / (B + P - Decimal(4)*norm)
	else:
		w2 = None

	return w2

def find_wr2_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [w0_0, w0_1]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (B - 3*norm) w - B
	fct_common2b(w) = (N - norm) w - N

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w2 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	B = a1 + a2 + a3 + a4 - a0
	N = ai - b - c - d - e

	w2 = Decimal()
	
	if (B - N - Decimal(2)*norm) != 0:
		w2 = (B - N) / (B - N - Decimal(2)*norm)
	else:
		w2 = None

	return w2

def find_wr1_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a guess with 2 consecutive bits in common.

	Fct_common2b(w) = fct_solution(w), for which w in [0, w0_0]
	with 0 =< w0_0 < w0_1 < w0_2 < w0_3 < w0_4 < 1 ?
	
	fct_solution(w) = (A - 5*norm) w - A
	fct_common2b(w) = (N - norm) w - N

	Parameters:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal

	Return:
	w1 -- Decimal
	"""
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	ai = common2b_init[i % n]
	b  = common2b_init[(i + 1) % n]
	c  = common2b_init[(i + 2) % n]
	d  = common2b_init[(i + 3) % n]
	e  = common2b_init[(i + 4) % n]

	b = abs(b)
	c = abs(c)
	d = abs(d)
	e = abs(e)

	A = a0 + a1 + a2 + a3 + a4
	N = ai - b - c - d - e

	w1 = Decimal()
	
	if (A - N - Decimal(4)*norm) != 0:
		w1 = (A - N) / (A - N - Decimal(4)*norm)
	else:
		w1 = None

	return w1

def common_elem(start, stop, common, solution):
	"""
	Return the list of common elements between the sublists. 

	Parameters:
	start 	 -- integer
	stop 	 -- integer
	common   -- list of Decimal
	solution -- list of Decimal

	Return:
	sub -- list of Decimal
	"""

	sub = [elem for elem in common[start:stop] if elem in solution[start:stop]]

	return sub

def count_common(stop, common, solution):
	"""
	Count the number of common elements between the sublists. 

	For
		w5: stop = 4
		w4: stop = 3
		w3: stop = 2
		w2: stop = 1

	Parameters:
	stop 	 -- integer
	common   -- list of Decimal
	solution -- list of Decimal

	Return:
	sub -- list of Decimal
	"""
	count = sum(x == y for x, y in zip(common[:stop], solution[:stop]))

	return count

def test_loop(mu, n, norm, kappa, K, R, S_ref, ax1):

	# # Display a figure with two subplots
	# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,8), gridspec_kw={"height_ratios": [4,1]}, sharex=True)
	
	# # Make subplots close to each other and hide x ticks for all but bottom plot.
	# fig.subplots_adjust(hspace=0.2)
	# plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)

	# Results of scalar products
	scalar_00000, scalar_00001, scalar_00010, scalar_00011 = [], [], [], []
	scalar_00100, scalar_00101, scalar_00110, scalar_00111 = [], [], [], []
	scalar_01000, scalar_01001, scalar_01010, scalar_01011 = [], [], [], []
	scalar_01100, scalar_01101, scalar_01110, scalar_01111 = [], [], [], []

	scalar_10000, scalar_10001, scalar_10010, scalar_10011 = [], [], [], []
	scalar_10100, scalar_10101, scalar_10110, scalar_10111 = [], [], [], []
	scalar_11000, scalar_11001, scalar_11010, scalar_11011 = [], [], [], []
	scalar_11100, scalar_11101, scalar_11110, scalar_11111 = [], [], [], []
	
	w_init = []
	
	# Signal power consumption for a key
	S = gen_S(mu, n, kappa, K)
	
	for j in range(1001):

		# Weight of signal part
		w = j / 1000
		
		scalar_abs = []
	
		# i-th bit of register state studied
		for i in range(n):
			
			# Global power consumption P = (1 - w) R  + w S
			P_i = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S[i])]

			# Squared scalar product |<S-ref, P>|
			scalar_i = np.dot(S_ref[i], P_i)
			scalar_abs_i = abs(scalar_i)
			scalar_abs += [scalar_abs_i]

		# Scalar product for i = 0
		# s0 = ['_00__', '_00__', '_00__', '_00__', '_01__', '_01__', '_01__', '_01__', 
		#		'_10__', '_10__', '_10__', '_10__', '_11__', '_11__', '_11__', '_11__',
		#		'_00__', '_00__', '_00__', '_00__', '_01__', '_01__', '_01__', '_01__', 
		#		'_10__', '_10__', '_10__', '_10__', '_11__', '_11__', '_11__', '_11__']
		s0 = [s for s in zip(scalar_abs[0], scalar_abs[0], scalar_abs[0], scalar_abs[0])]
		s0 = list(itertools.chain.from_iterable(s0))
		s0 = s0 * 2

		# Scalar product for i = 1
		# s1 = ['__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_', 
		#		'__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_',
		#		'__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_', 
		#		'__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_']
		s1 = [s for s in zip(scalar_abs[1], scalar_abs[1])]
		s1 = list(itertools.chain.from_iterable(s1))
		s1 = s1 * 4

		# Scalar product for i = 2
		# s2 = ['___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11', 
		#		'___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11', 
		#		'___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11', 
		#		'___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11']
		s2 = scalar_abs[2].tolist()
		s2 = s2 * 8

		# Scalar product for i = 3
		# s3 = ['0___0', '0___1', '0___0', '0___1', '0___0', '0___1', '0___0', '0___1',
		#		'0___0', '0___1', '0___0', '0___1', '0___0', '0___1', '0___0', '0___1', 
		#		'1___0', '1___1', '1___0', '1___1', '1___0', '1___1', '1___0', '1___1', 
		#		'1___0', '1___1', '1___0', '1___1', '1___0', '1___1', '1___0', '1___1']
		s3 = [(s,s) for s in zip(scalar_abs[3][::2], scalar_abs[3][1::2], scalar_abs[3][::2], 
								 scalar_abs[3][1::2], scalar_abs[3][::2], scalar_abs[3][1::2], 
								 scalar_abs[3][::2], scalar_abs[3][1::2])]
		s3 = list(itertools.chain.from_iterable(s3))
		s3 = list(itertools.chain.from_iterable(s3))

		# Scalar product for i = 4
		# s4 = ['00___', '00___', '00___', '00___', '00___', '00___', '00___', '00___', 
		#		'01___', '01___', '01___', '01___', '01___', '01___', '01___', '01___', 
		#		'10___', '10___', '10___', '10___', '10___', '10___', '10___', '10___', 
		#		'11___', '11___', '11___', '11___', '11___', '11___', '11___', '11___']
		s4 = [s for s in zip(scalar_abs[4], scalar_abs[4], scalar_abs[4], scalar_abs[4], 
							 scalar_abs[4], scalar_abs[4], scalar_abs[4], scalar_abs[4])]
		s4 = list(itertools.chain.from_iterable(s4))

		# Sum of absolute scalar products for all kappa values
		scalar5 = [s0[m] + s1[m] + s2[m] + s3[m] + s4[m] for m in range(len(mu))]

		w_init.append(w)

		scalar_00000.append(Decimal(scalar5[0]) / Decimal(5*norm))
		scalar_00001.append(Decimal(scalar5[1]) / Decimal(5*norm))
		scalar_00010.append(Decimal(scalar5[2]) / Decimal(5*norm))
		scalar_00011.append(Decimal(scalar5[3]) / Decimal(5*norm))
		scalar_00100.append(Decimal(scalar5[4]) / Decimal(5*norm))
		scalar_00101.append(Decimal(scalar5[5]) / Decimal(5*norm))
		scalar_00110.append(Decimal(scalar5[6]) / Decimal(5*norm))
		scalar_00111.append(Decimal(scalar5[7]) / Decimal(5*norm))

		scalar_01000.append(Decimal(scalar5[8]) / Decimal(5*norm))
		scalar_01001.append(Decimal(scalar5[9]) / Decimal(5*norm))
		scalar_01010.append(Decimal(scalar5[10]) / Decimal(5*norm))
		scalar_01011.append(Decimal(scalar5[11]) / Decimal(5*norm))
		scalar_01100.append(Decimal(scalar5[12]) / Decimal(5*norm))
		scalar_01101.append(Decimal(scalar5[13]) / Decimal(5*norm))
		scalar_01110.append(Decimal(scalar5[14]) / Decimal(5*norm))
		scalar_01111.append(Decimal(scalar5[15]) / Decimal(5*norm))

		scalar_10000.append(Decimal(scalar5[16]) / Decimal(5*norm))
		scalar_10001.append(Decimal(scalar5[17]) / Decimal(5*norm))
		scalar_10010.append(Decimal(scalar5[18]) / Decimal(5*norm))
		scalar_10011.append(Decimal(scalar5[19]) / Decimal(5*norm))
		scalar_10100.append(Decimal(scalar5[20]) / Decimal(5*norm))
		scalar_10101.append(Decimal(scalar5[21]) / Decimal(5*norm))
		scalar_10110.append(Decimal(scalar5[22]) / Decimal(5*norm))
		scalar_10111.append(Decimal(scalar5[23]) / Decimal(5*norm))

		scalar_11000.append(Decimal(scalar5[24]) / Decimal(5*norm))
		scalar_11001.append(Decimal(scalar5[25]) / Decimal(5*norm))
		scalar_11010.append(Decimal(scalar5[26]) / Decimal(5*norm))
		scalar_11011.append(Decimal(scalar5[27]) / Decimal(5*norm))
		scalar_11100.append(Decimal(scalar5[28]) / Decimal(5*norm))
		scalar_11101.append(Decimal(scalar5[29]) / Decimal(5*norm))
		scalar_11110.append(Decimal(scalar5[30]) / Decimal(5*norm))
		scalar_11111.append(Decimal(scalar5[31]) / Decimal(5*norm))

	ax1.plot(w_init, scalar_00000, ':', color='black', markersize=4, label='00000')
	ax1.plot(w_init, scalar_00001, '-.', color='black',  markersize=4, label='00001')
	ax1.plot(w_init, scalar_00010, '-.', color='black', markersize=4, label='00010')
	ax1.plot(w_init, scalar_00011, '-.' , color='black', markersize=4, label='00011')
	ax1.plot(w_init, scalar_00100, '-.', color='black', markersize=4, label='00100')
	ax1.plot(w_init, scalar_00101, '-.', color='black',  markersize=4, label='00101')
	ax1.plot(w_init, scalar_00110, '-.', color='black', markersize=4, label='00110')
	ax1.plot(w_init, scalar_00111, '-.', color='black', markersize=4, label='00111')

	ax1.plot(w_init, scalar_01000, '-.', color='black', markersize=4, label='01000')
	ax1.plot(w_init, scalar_01001, '-.', color='black',  markersize=4, label='01001')
	ax1.plot(w_init, scalar_01010, '-.', color='black', markersize=4, label='01010')
	ax1.plot(w_init, scalar_01011, '-.', color='black', markersize=4, label='01011')
	ax1.plot(w_init, scalar_01100, '-.', color='black', markersize=4, label='01100')
	ax1.plot(w_init, scalar_01101, '-.', color='black',  markersize=4, label='01101')
	ax1.plot(w_init, scalar_01110, '-.', color='black', markersize=4, label='01110')
	ax1.plot(w_init, scalar_01111, '-.', color='black', markersize=4, label='01111')

	ax1.plot(w_init, scalar_10000, '-.', color='black', markersize=4, label='10000')
	ax1.plot(w_init, scalar_10001, '-.', color='black',  markersize=4, label='10001')
	ax1.plot(w_init, scalar_10010, '-.', color='black', markersize=4, label='10010')
	ax1.plot(w_init, scalar_10011, '-.', color='black', markersize=4, label='10011')
	ax1.plot(w_init, scalar_10100, ':', color='black', markersize=4, label='10100')
	ax1.plot(w_init, scalar_10101, '-.', color='black',  markersize=4, label='10101')
	ax1.plot(w_init, scalar_10110, '-.', color='black', markersize=4, label='10110')
	ax1.plot(w_init, scalar_10111, '-.', color='black', markersize=4, label='10111')

	ax1.plot(w_init, scalar_11000, '-.', color='black', markersize=4, label='11000')
	ax1.plot(w_init, scalar_11001, '-.', color='black',  markersize=4, label='11001')
	ax1.plot(w_init, scalar_11010, '-.', color='black', markersize=4, label='11010')
	ax1.plot(w_init, scalar_11011, '-.' , color='black', markersize=4, label='11011')
	ax1.plot(w_init, scalar_11100, '-.', color='black', markersize=4, label='11100')
	ax1.plot(w_init, scalar_11101, '-.', color='black',  markersize=4, label='11101')
	ax1.plot(w_init, scalar_11110, '-.', color='black', markersize=4, label='11110')
	ax1.plot(w_init, scalar_11111, ':', color='black', markersize=4, label='11111')

def append_wr(j, value, interval, wr):
	"""
	Append value to wr if in interval. 

	Parameters:
	j 		 -- integer
	value    -- Decimal
	interval -- list
	wr 		 -- list of Decimal

	Return: None
	"""
	if (value != None) and (value > interval[0]) and (value < interval[1]):
		wr.append((value, 'wr%s' % j))

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

		# Count number of rank increment ('wr4' or 'wr5' or 'wr6') 
		# and rank decrement ('wr2' or 'wr1' or 'wr3')
		counter_1 = sum(1 for tuple in wr if tuple[1] == ('wr4') or tuple[1] == ('wr5') or tuple[1] == ('wr6'))
		counter_2 = sum(-1 for tuple in wr if tuple[1] == ('wr2') or tuple[1] == ('wr1') or tuple[1] == ('wr3'))
		num_wr = counter_1 + counter_2

		rank_init = 1 + num_wr

		rank.append(rank_init)

		for tuple in wr:

			# If 'wr4'or 'wr5' or 'wr6', rank increases
			if tuple[1] == ('wr4') or tuple[1] == ('wr5') or tuple[1] == ('wr6'):
				rank.append(rank[-1] - 1)
				
			# If 'wr2' or 'wr1' or 'wr3', rank decreases
			if tuple[1] == ('wr2') or tuple[1] == ('wr1') or tuple[1] == ('wr3'):
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, wr

def compute_rank_wr(wr):
	"""
	Compute intervals for each ranks in order to plot them.

	Parameter:
	wr -- list of Decimal

	Return:
	rank, wr -- lists
	"""

	# Transform Decimal into float
	getcontext().prec = 6
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

	return rank, wr

def sim_1x5bits(n, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 
	
	Parameters:
	n 		-- integer
	K 		-- string
	kappa   -- string
	num_sim -- integer
	
	Return:
	NaN
	"""

	# Message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))
	norm = Decimal(2**n)

	# Noise power vector
	R = noise(mu)

	# Signal reference vectors
	S_ref = gen_S_ref(mu, n)

	# Initial values of scalar product
	scalar_init = gen_scalar_init(n, R, S_ref)
	init_scalar = [sublist.tolist() for sublist in scalar_init]

	# List of indexes
	list_kappa_idx = kappa_idx()

	# Retrieve idx of kappa's from solution kappa
	solution_idx = [sublist for sublist in list_kappa_idx if sublist[5] == kappa]
	solution_idx = list(itertools.chain.from_iterable(solution_idx)) # Un-nest the previous list

	common4b_idx = []
	for j in range(5): # Five possibilities with four consecutive kappa bits in common
		common4b_idx += [find_idx_common4b(j, kappa, list_kappa_idx, solution_idx)]

	common3b_idx = []
	for j in range(5): # Five possibilities with three consecutive kappa bits in common
		common3b_idx += [find_idx_common3b(j, list_kappa_idx, solution_idx)]

	common2b_idx = []
	for j in range(5): # Ten possibilities with four consecutive kappa bits in common
		idx0, idx1 = find_idx_common2b(j, list_kappa_idx, solution_idx)
		common2b_idx.append(idx0)
		common2b_idx.append(idx1)

	nonsol_idx = find_idx_nonsol(list_kappa_idx, solution_idx)

	# Retrieve corresponding scalar products
	solution_init = [init_scalar[i][solution_idx[i]] for i in range(n)]
	common4b_init = find_init(n, init_scalar, common4b_idx)
	common3b_init = find_init(n, init_scalar, common3b_idx)
	common2b_init = find_init(n, init_scalar, common2b_idx)
	nonsol_init   = find_init(n, init_scalar, nonsol_idx)

	# Determine the sign of initial solution value depending on the activity of the register at bit i
	xor_solution(n, K, kappa, solution_init)
	xor_common(3, n, K, kappa, common4b_init)
	xor_common(2, n, K, kappa, common3b_init)
	xor_common(1, n, K, kappa, common2b_init)

	# Find w0 for each bit of the register, tup = (w0, a)
	list_tup_w0 = find_abs_w0(solution_init, len(mu))

	# Sort result by increasing order on w0
	list_tup_w0 = sorted(list_tup_w0, key=lambda x: x[0])
	list_w0 = [tup[0] for tup in list_tup_w0]

	# Sorted list of solution_init by increasing order on w0
	solution_init = [tup[1] for tup in list_tup_w0]
	
	# Sorted list of common_init according to solution_init
	common4b_new  = reorder_common4b(solution_init, common4b_init)
	common3b_new  = reorder_common3b(solution_init, common3b_init)
	common2b_new1 = reorder_common2b(solution_init, common2b_init[::2])
	common2b_new2 = reorder_common2b(solution_init, common2b_init[1::2])
	common2b_new  = common2b_new1 + common2b_new2

	""""""

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": [2,1]}, sharex=True)
	
	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
	
	# Delimit definition intervals for w
	for w0 in list_w0:
		label_w0 = ["w_0", "w'_0", "w''_0", "w'''_0", "w''''_0"]
		style=['-.', ':', '--', '-.', ':']
		if (w0 > 0) and (w0 < 1):
			ax1.axvline(x=w0, linestyle=style[list_w0.index(w0)], color='tab:blue', markersize=1, label=label_w0[list_w0.index(w0)])

	# Plot nonsolution scalar product functions in interval
	interval = [0, 1]

	for j in range(len(nonsol_init)):
		nonsol_abs = [fct_abs_nonsol(w, n, norm, nonsol_init[j]) for w in interval]
		ax1.plot(interval, nonsol_abs, '-', color='tab:red', markersize=1, label='%s' % nonsol_idx[j][-1])

	# Plot scalar product functions in interval
	interval = [list_w0[4], 1]

	solution_abs = [fct_abs_solution(w, n, norm, solution_init) for w in interval]
	ax1.plot(interval, solution_abs, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])
	
	for j in range(len(common4b_init)):
		common4b_abs = [fct_abs_common4(w, j, n, norm, common4b_init[j]) for w in interval]
		ax1.plot(interval, common4b_abs, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

	for j in range(len(common3b_init)):
		common3b_abs = [fct_abs_common3(w, j, n, norm, common3b_init[j]) for w in interval]
		ax1.plot(interval, common3b_abs, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

	for j in range(len(common2b_init) // 2):
		for l in range(2):
			common2b_abs = [fct_abs_common2(w, j, n, norm, common2b_init[2*j + l]) for w in interval]
			ax1.plot(interval, common2b_abs, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j +l][-1])

	# List of all intersections with the solution kappa function
	wr = []

	# Intersections wr6 between solution function and other scalar product functions
	for j in range(len(nonsol_init)):
		wr6_nonsol = find_wr6_nonsol(norm, solution_init, nonsol_init[j])
		append_wr(6, wr6_nonsol, interval, wr)
		# ax1.axvline(x=wr6_nonsol, linestyle='--', color='tab:red', markersize=1, label='wr6')

	for j in range(len(common4b_init)):
		wr6_common4b = find_wr6_common4(j, n, norm, solution_init, common4b_init[j])
		append_wr(6, wr6_common4b, interval, wr)
		# ax1.axvline(x=wr6_common4b, linestyle='--', color='tab:brown', markersize=1, label='wr6')

	for j in range(len(common3b_init)):
		wr6_common3b = find_wr6_common3(j, n, norm, solution_init, common3b_init[j])
		append_wr(6, wr6_common3b, interval, wr)
		# ax1.axvline(x=wr6_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr6')

	for j in range(len(common2b_init) // 2):
		for l in range(2):
			wr6_common2b = find_wr6_common2(j, n, norm, solution_init, common2b_init[2*j + l])
			append_wr(6, wr6_common2b, interval, wr)
			# ax1.axvline(x=wr6_common2b, linestyle='--', color='tab:purple', markersize=1, label='wr6')

	# ------------------------------------------------------------------------------------------- #

	# Solution function and common ones
	if list_w0[4] != 0:

		interval = [list_w0[3], list_w0[4]]

		# Plot scalar product functions in interval
		solution_abs = [fct_abs_solution(w, n, norm, solution_init) for w in interval]
		ax1.plot(interval, solution_abs, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])
		
		for j in range(len(common4b_init)):
			common4b_abs = [fct_abs_common4(w, j, n, norm, common4b_init[j]) for w in interval]
			ax1.plot(interval, common4b_abs, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

		for j in range(len(common3b_init)):
			common3b_abs = [fct_abs_common3(w, j, n, norm, common3b_init[j]) for w in interval]
			ax1.plot(interval, common3b_abs, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

		for j in range(len(common2b_init) // 2):
			for l in range(2):
				common2b_abs = [fct_abs_common2(w, j, n, norm, common2b_init[2*j + l]) for w in interval]
				ax1.plot(interval, common2b_abs, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j +l][-1])

		# Intersections between solution function and nonsol functions	
		for j in range(len(nonsol_init)):
			wr5_nonsol = find_wr5_nonsol(norm, solution_init, nonsol_init[j])
			append_wr(5, wr5_nonsol, interval, wr)
			# ax1.axvline(x=wr5_nonsol, linestyle='--', color='tab:red', markersize=1, label='wr5')

		# Intersection between solution function and common4b function
		for sublist in common4b_new:

			if count_common(4, sublist, solution_init) == 2: #I: ai + ai1 - ai2

				i = common4b_new.index(sublist)
				common4b_inc = common_elem(0, 4, sublist, solution_init)      # ai, ai1
				common4b_dec = sublist[-1]									  # ai2
				common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

				wr5_common4b = find_wr5_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
				append_wr(5, wr5_common4b, interval, wr)
				# ax1.axvline(x=wr5_common4b, linestyle='--', color='tab:olive', markersize=1, label='wr5')

		# Intersection between solution function and common3b function
		for sublist in common3b_new:

			if count_common(4, sublist, solution_init) == 2: # M

				i = common3b_new.index(sublist)
				common3b_inc = common_elem(0, 4, sublist, solution_init)      # ai, ai1
				common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

				wr5_common3b = find_wr5bis_common3(norm, solution_init, common3b_inc, common3b_non)
				append_wr(5, wr5_common3b, interval, wr)
				# ax1.axvline(x=wr5_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr5')

			if count_common(4, sublist, solution_init) == 1: # L

				i = common3b_new.index(sublist)
				common3b_inc = common_elem(0, 4, sublist, solution_init)      # ai
				common3b_dec = sublist[-1]									  # ai1
				common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

				wr5_common3b = find_wr5_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
				append_wr(5, wr5_common3b, interval, wr)
				# ax1.axvline(x=wr5_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr5')

		# Intersection between solution function and common2b function
		for sublist in common2b_new:

			if count_common(4, sublist, solution_init) == 1: # P

				i = common2b_new.index(sublist)
				wr5_common2b = find_wr5bis_common2(i, n, norm, solution_init, sublist)
				append_wr(5, wr5_common2b, interval, wr)
				# ax1.axvline(x=wr5_common2b, linestyle='--', color='tab:olive', markersize=1, label='wr5')

			if count_common(4, sublist, solution_init) == 0: # N
				
				i = common2b_new.index(sublist)
				wr5_common2b = find_wr5_common2(i, n, norm, solution_init, sublist)
				append_wr(5, wr5_common2b, interval, wr)
				# ax1.axvline(x=wr5_common2b, linestyle='--', color='tab:olive', markersize=1, label='wr5')

		# ------------------------------------------------------------------------------------------- #

		if list_w0[3] != 0:
		
			interval = [list_w0[2], list_w0[3]]

			# Plot scalar product functions in interval
			solution_abs = [fct_abs_solution(w, n, norm, solution_init) for w in interval]
			ax1.plot(interval, solution_abs, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])
			
			for j in range(len(common4b_init)):
				common4b_abs = [fct_abs_common4(w, j, n, norm, common4b_init[j]) for w in interval]
				ax1.plot(interval, common4b_abs, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

			for j in range(len(common3b_init)):
				common3b_abs = [fct_abs_common3(w, j, n, norm, common3b_init[j]) for w in interval]
				ax1.plot(interval, common3b_abs, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

			for j in range(len(common2b_init) // 2):
				for l in range(2):
					common2b_abs = [fct_abs_common2(w, j, n, norm, common2b_init[2*j + l]) for w in interval]
					ax1.plot(interval, common2b_abs, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j +l][-1])

			# Intersections between solution function and nonsol functions	
			for j in range(len(nonsol_init)):
				wr4_nonsol = find_wr4_nonsol(norm, solution_init, nonsol_init[j])
				append_wr(4, wr4_nonsol, interval, wr)
				# ax1.axvline(x=wr4_nonsol, linestyle='--', color='tab:red', markersize=1, label='wr4')

			# Intersection between solution function and common4b function
			for sublist in common4b_new:
				
				if count_common(3, sublist, solution_init) == 3: # J: ai + ai1 + ai2
					
					i = common4b_new.index(sublist)
					common4b_inc = common_elem(0, 3, sublist, solution_init)      # ai, ai1, ai2
					common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

					wr4_common4b = find_wr4bis_common4(norm, solution_init, common4b_inc, common4b_non)
					append_wr(4, wr4_common4b, interval, wr)
					# ax1.axvline(x=wr4_common4b, linestyle='--', color='tab:olive', markersize=1, label='wr4')

				elif count_common(3, sublist, solution_init) == 1: # H

					i = common4b_new.index(sublist)
					common4b_inc = common_elem(0, 3, sublist, solution_init)  # ai
					common4b_dec = common_elem(3, 5, sublist, solution_init)  # ai1, ai2
					common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

					wr4_common4b = find_wr4_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
					append_wr(4, wr4_common4b, interval, wr)
					# ax1.axvline(x=wr4_common4b, linestyle='--', color='tab:olive', markersize=1, label='wr4')

			# Intersection between solution function and common3b function
			for sublist in common3b_new:
				
				if count_common(3, sublist, solution_init) == 2: # M

					i = common3b_new.index(sublist)
					common3b_inc = common_elem(0, 3, sublist, solution_init)      # ai, ai1
					common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

					wr4_common3b = find_wr4ter_common3(norm, solution_init, common3b_inc, common3b_non)
					append_wr(4, wr4_common3b, interval, wr)
					# ax1.axvline(x=wr4_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr4')

				if count_common(3, sublist, solution_init) == 1: # L

					i = common3b_new.index(sublist)
					common3b_inc = common_elem(0, 3, sublist, solution_init)   # ai
					common3b_dec = common_elem(3, 5, sublist, solution_init)   # ai1
					common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

					wr4_common3b = find_wr4bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
					append_wr(4, wr4_common3b, interval, wr)
					# ax1.axvline(x=wr4_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr4')

				if count_common(3, sublist, solution_init) == 0: # K

					i = common3b_new.index(sublist)
					common3b_dec = sublist[-2:]  # ai, ai1
					common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

					wr4_common3b = find_wr4_common3(norm, solution_init, common3b_dec, common3b_non)
					append_wr(4, wr4_common3b, interval, wr)
					# ax1.axvline(x=wr4_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr4')

			# Intersection between solution function and common2b function
			for sublist in common2b_new:

				if count_common(3, sublist, solution_init) == 0: # N

					wr4_common2b = find_wr4_common2(i, n, norm, solution_init, sublist)
					append_wr(4, wr4_common2b, interval, wr)
					# ax1.axvline(x=wr4_common2b, linestyle='--', color='tab:olive', markersize=1, label='wr4')

			# ------------------------------------------------------------------------------------------- #

			if list_w0[2] != 0:
			
				interval = [list_w0[1], list_w0[2]]

				# Plot scalar product functions in interval
				solution_abs = [fct_abs_solution(w, n, norm, solution_init) for w in interval]
				ax1.plot(interval, solution_abs, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])
				
				for j in range(len(common4b_init)):
					common4b_abs = [fct_abs_common4(w, j, n, norm, common4b_init[j]) for w in interval]
					ax1.plot(interval, common4b_abs, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

				for j in range(len(common3b_init)):
					common3b_abs = [fct_abs_common3(w, j, n, norm, common3b_init[j]) for w in interval]
					ax1.plot(interval, common3b_abs, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

				for j in range(len(common2b_init) // 2):
					for l in range(2):
						common2b_abs = [fct_abs_common2(w, j, n, norm, common2b_init[2*j + l]) for w in interval]
						ax1.plot(interval, common2b_abs, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j +l][-1])

				# Intersections between solution function and nonsol functions	
				for j in range(len(nonsol_init)):
					wr3_nonsol = find_wr3_nonsol(norm, solution_init, nonsol_init[j])
					append_wr(3, wr3_nonsol, interval, wr)
					# ax1.axvline(x=wr3_nonsol, linestyle='--', color='tab:red', markersize=1, label='wr3')

				# Intersection between solution function and common4b function
				for sublist in common4b_new:
					
					if count_common(2, sublist, solution_init) == 2: # I
						
						i = common4b_new.index(sublist)
						common4b_inc = common_elem(0, 2, sublist, solution_init) # ai, ai1
						common4b_dec = common_elem(2, 5, sublist, solution_init) # ai2
						common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

						wr3_common4b = find_wr3bis_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
						append_wr(3, wr3_common4b, interval, wr)
						# ax1.axvline(x=wr3_common4b, linestyle='--', color='tab:olive', markersize=1, label='wr3')

					elif count_common(2, sublist, solution_init) == 0: # G

						i = common4b_new.index(sublist)
						common4b_dec = common_elem(2, 5, sublist, solution_init)  # ai, ai1, ai2
						common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

						wr3_common4b = find_wr3_common4(norm, solution_init, common4b_dec, common4b_non)
						append_wr(3, wr3_common4b, interval, wr)
						# ax1.axvline(x=wr3_common4b, linestyle='--', color='tab:olive', markersize=1, label='wr3')

				# Intersection between solution function and common3b function
				for sublist in common3b_new:
					
					if count_common(2, sublist, solution_init) == 2: # M

						i = common3b_new.index(sublist)
						common3b_inc = sublist[:2]      # ai, ai1
						common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

						wr3_common3b = find_wr3ter_common3(norm, solution_init, common3b_inc, common3b_non)
						append_wr(3, wr3_common3b, interval, wr)
						# ax1.axvline(x=wr3_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr3')

					if count_common(2, sublist, solution_init) == 1: # L

						i = common3b_new.index(sublist)
						common3b_inc = common_elem(0, 2, sublist, solution_init)   # ai
						common3b_dec = common_elem(2, 5, sublist, solution_init)   # ai1
						common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

						wr3_common3b = find_wr3bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
						append_wr(3, wr3_common3b, interval, wr)
						# ax1.axvline(x=wr3_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr3')

					if count_common(2, sublist, solution_init) == 0: # K

						i = common3b_new.index(sublist)
						common3b_dec = common_elem(2, 5, sublist, solution_init) # ai, ai1
						common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

						wr3_common3b = find_wr3_common3(norm, solution_init, common3b_dec, common3b_non)
						append_wr(3, wr3_common3b, interval, wr)
						# ax1.axvline(x=wr3_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr3')

				# Intersection between solution function and common2b function
				for sublist in common2b_new:

					if count_common(2, sublist, solution_init) == 1: # P

						wr3_common2b = find_wr3_common2(i, n, norm, solution_init, sublist)
						append_wr(3, wr3_common2b, interval, wr)
						# ax1.axvline(x=wr3_common2b, linestyle='--', color='tab:olive', markersize=1, label='wr3')

				# ------------------------------------------------------------------------------------------- #

				if list_w0[1] != 0:
				
					interval = [list_w0[0], list_w0[1]]

					# Plot scalar product functions in interval
					solution_abs = [fct_abs_solution(w, n, norm, solution_init) for w in interval]
					ax1.plot(interval, solution_abs, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])
					
					for j in range(len(common4b_init)):
						common4b_abs = [fct_abs_common4(w, j, n, norm, common4b_init[j]) for w in interval]
						ax1.plot(interval, common4b_abs, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

					for j in range(len(common3b_init)):
						common3b_abs = [fct_abs_common3(w, j, n, norm, common3b_init[j]) for w in interval]
						ax1.plot(interval, common3b_abs, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

					for j in range(len(common2b_init) // 2):
						for l in range(2):
							common2b_abs = [fct_abs_common2(w, j, n, norm, common2b_init[2*j + l]) for w in interval]
							ax1.plot(interval, common2b_abs, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j +l][-1])


					# Intersections between solution function and nonsol functions	
					for j in range(len(nonsol_init)):
						wr2_nonsol = find_wr2_nonsol(norm, solution_init, nonsol_init[j])
						append_wr(2, wr2_nonsol, interval, wr)
						# ax1.axvline(x=wr2_nonsol, linestyle='--', color='tab:red', markersize=1, label='wr2')

					# Intersection between solution function and common4b function
					for sublist in common4b_new:
						
						if count_common(1, sublist, solution_init) == 1: # H
							
							i = common4b_new.index(sublist)
							common4b_inc = common_elem(0, 1, sublist, solution_init) # ai
							common4b_dec = common_elem(1, 5, sublist, solution_init) # ai1, ai2
							common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

							wr2_common4b = find_wr2_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
							append_wr(2, wr2_common4b, interval, wr)
							# ax1.axvline(x=wr2_common4b, linestyle='--', color='tab:olive', markersize=1, label='wr2')

					# Intersection between solution function and common3b function
					for sublist in common3b_new:

						if  count_common(1, sublist, solution_init) == 1: # L

							i = common3b_new.index(sublist)
							common3b_inc = sublist[1]   # ai
							common3b_dec = common_elem(1, 5, sublist, solution_init)   # ai1
							common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

							wr2_common3b = find_wr2bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
							append_wr(2, wr2_common3b, interval, wr)
							# ax1.axvline(x=wr2_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr2')

						if count_common(1, sublist, solution_init) == 0: # K

							i = common3b_new.index(sublist)
							common3b_dec = common_elem(1, 5, sublist, solution_init) # ai, ai1
							common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

							wr2_common3b = find_wr2_common3(norm, solution_init, common3b_dec, common3b_non)
							append_wr(2, wr2_common3b, interval, wr)
							# ax1.axvline(x=wr2_common3b, linestyle='--', color='tab:olive', markersize=1, label='wr2')

					# Intersection between solution function and common2b function
					for sublist in common2b_new:

						if count_common(1, sublist, solution_init) == 1: # P

							i = common2b_new.index(sublist)
							wr2_common2b = find_wr2bis_common2(i, n, norm, solution_init, sublist)
							append_wr(2, wr2_common2b, interval, wr)
							# ax1.axvline(x=wr2_common2b, linestyle='--', color='tab:olive', markersize=1, label='wr2')

						if count_common(1, sublist, solution_init) == 0: # N
							
							i = common2b_new.index(sublist)
							wr2_common2b = find_wr2_common2(i, n, norm, solution_init, sublist)
							append_wr(2, wr2_common2b, interval, wr)
							# ax1.axvline(x=wr2_common2b, linestyle='--', color='tab:olive', markersize=1, label='wr2')

					# ------------------------------------------------------------------------------------------- #

					if list_w0[0] != 0:
					
						interval = [0, list_w0[0]]

						# Plot scalar product functions in interval
						solution_abs = [fct_abs_solution(w, n, norm, solution_init) for w in interval]
						ax1.plot(interval, solution_abs, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])
						
						for j in range(len(common4b_init)):
							common4b_abs = [fct_abs_common4(w, j, n, norm, common4b_init[j]) for w in interval]
							ax1.plot(interval, common4b_abs, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

						for j in range(len(common3b_init)):
							common3b_abs = [fct_abs_common3(w, j, n, norm, common3b_init[j]) for w in interval]
							ax1.plot(interval, common3b_abs, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

						for j in range(len(common2b_init) // 2):
							for l in range(2):
								common2b_abs = [fct_abs_common2(w, j, n, norm, common2b_init[2*j + l]) for w in interval]
								ax1.plot(interval, common2b_abs, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j +l][-1])


						# Intersections between solution function and nonsol functions	
						for j in range(len(nonsol_init)):
							wr1_nonsol = find_wr1_nonsol(norm, solution_init, nonsol_init[j])
							append_wr(1, wr1_nonsol, interval, wr)
							# ax1.axvline(x=wr1_nonsol, linestyle='--', color='tab:red', markersize=1, label='wr1')

						# Intersections between solution function and common4b functions
						for j in range(len(common4b_init)):
							wr1_common4b = find_wr1_common4(j, n, norm, solution_init, common4b_init[j])
							append_wr(1, wr1_common4b, interval, wr)
							# ax1.axvline(x=wr1_common4b, linestyle='--', color='tab:brown', markersize=1, label='wr1')

						# Intersections between solution function and common3b functions
						for j in range(len(common3b_init)):
							wr1_common3b = find_wr1_common3(j, n, norm, solution_init, common3b_init[j])
							append_wr(1, wr1_common3b, interval, wr)
							# ax1.axvline(x=wr1_common3b, linestyle='--', color='tab:brown', markersize=1, label='wr1')

						# Intersections between solution function and common2b functions	
						for j in range(len(common2b_init) // 2):
							for l in range(2):
								wr1_common2b = find_wr1_common2(j, n, norm, solution_init, common2b_init[2*j + l])
								append_wr(1, wr1_common2b, interval, wr)
								# ax1.axvline(x=wr1_common2b, linestyle='--', color='tab:purple', markersize=1, label='wr1')

	# # Test with scalar product values
	# test_loop(mu, n, norm, kappa, K, R, S_ref, ax1)

	# Find rank of solution kappa
	rank, wr = compute_rank_wr(wr)
	
	# ax1.legend(loc='upper left', ncol=4, fancybox=True)
	ax1.set_title('Combined DOM for K=%s and kappa=%s' %(K, kappa))
	ax1.set_ylabel('Absolute scalar product |<S_ref,P>|')

	ax2.plot(wr, rank, color='tab:blue')
	ax2.set_ylabel('Rank kappa solution')
	ax2.set_xlabel('Weight w')
	ax2.set_ylim([32, 0])

	fig.tight_layout()
	plt.show()
	# plt.savefig('./plot/sim_1x3bits_abs_K=%s_kappa=%s_#%s.pdf' % (K, kappa, num_sim), dpi=300)
	# plt.savefig('./plot/sim_1x3bits_abs_K=%s_kappa=%s_#%s.png' % (K, kappa, num_sim))
	# plt.close(fig)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa, and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations to perform')

	args = parser.parse_args()

	if len(args.K) != 5 or len(args.kappa) != 5:
		print('\n**ERROR**')
		print('Required length of K and kappa: 5')

	else:
		# Length of signal part
		n = 5

		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		# Time option
		start_t = time.perf_counter()
		
		for j in range(num_sim):
			sim_1x5bits(n, K, kappa, j)

		# Time option
		end_t = time.perf_counter()
		print('\ntime', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))
	


