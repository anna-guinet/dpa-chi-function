#!/usr/bin/env python

""" DPA ON KECCAK 5-BIT CHI ROW & 1-BIT K

Save the figures for a simulation which display the outcome of a test
according to the SNR, in order to recover kappa bits. 

Within one simulation, we guess 2 bits of kappa, with the scalar product,
and a fixed noise power consumption vector.  
"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__     = "2.1"

import numpy as np
import itertools
import pandas as pd
from decimal import *
import math
import random
import bisect

import matplotlib.pylab as plt

import argparse
import sys
import time

def noise(mu):
	"""
	Compute a noise power consumption vector R for each message mu. 

	Parameter:
	mu -- list of Bool

	Return:
	R -- list of integers, length 2^n
	"""
	R = []

	# For each message mu
	for j in range(len(mu)):

		# Activity of noise part
		d = random.gauss(0, 2)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def gen_key_i(i, kappa, K):
	""" 
	Create key value where key = kappa, except for bit i: 
	key_(i mod n) = kappa_(i mod n) XOR K_(i mod n)
	
	Parameters:
	i 	  -- integer in [0,n-1]
	kappa -- string
	K 	  -- string

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

def gen_key(i, n):
	"""
	Generate all key values for key_(i+2 mod n) and key_(i+1 mod n), 
	regardless the i-th bit:
	- key_(i mod n) = kappa_i XOR K = 0
	- key_(i+1 mod n) = kappa_(i+1 mod n)
	- key_(i+2 mod n) = kappa_(i+2 mod n)
	- key_(i+3 mod n) = 0 if n = 5 
	- key_(i+4 mod n) = 0 if n = 5
	
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
	mu  -- list of bool, size n x 2^n
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

def gen_S(mu, n, kappa, K):
	"""
	Generate signal power consumption for every bit i of first chi's row, 
	for a couple (kappa, K). 
	
	Parameters:
	mu    -- 2D list of bool, size n x 2^n
	n     -- integer
	kappa -- string
	K     -- string
	
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
	mu -- 2D list of bool, size n x 2^n
	n  -- integer (default 3)

	Return:
	S_ref -- 3D list of integers, size 2^n x 2^(n-1) x n
	"""
	S_ref = []
	
	for i in range(n):
		key = gen_key(i, n)
		S_ref_i = signal_2D(n, i, key, mu)
		S_ref += [S_ref_i]

	return S_ref

def gen_scalar(n, w, R, S, S_ref):
	"""
	Generate initial scalar products for the chi row. 

	Parameters:
	n 	  -- integer
	w 	  -- Decimal
	R 	  -- list of Decimals
	S 	  -- list of integers
	S_ref -- list of integers

	Return:
	scalar -- list of Decimals
	"""
	scalar = []

	# i-th bit of register state studied
	for i in range(n):
					
		# Global power consumption P = (1 - w) R + w S
		P_i = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S[i])]

		# Scalar product <S-ref, P>
		scalar_i = np.dot(S_ref[i], P_i)
		scalar += [scalar_i]

	return scalar

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

	# Global power consumption P_init = R
	getcontext().prec = 8
	P_init = [Decimal(r) for r in R]

	# i-th bit of register state studied
	for i in range(n):	

		# Scalar product <S-ref, P>
		scalar_init_i = np.dot(S_ref[i], P_init)
		scalar_init += [scalar_init_i]

	return scalar_init

def kappa_idx():
	"""
	Provide scalar products indexes for kappa values.

	Return:
	list_kappa_idx -- list of lists
	"""
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
	n 	  		  -- integer
	K 	  		  -- string
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
	i 	  		-- integer
	b 	  		-- integer
	n 	  		-- integer
	K 	  		-- string
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
	b 	 		-- integer
	n 	  		-- integer
	K 	  		-- string
	kappa 		-- string
	common_init -- list of Decimal

	Return: None
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

def find_abs_w0(solution_init, norm):
	"""
	Find the fiber points for each DoM scalar product function.

	Parameters:
	norm 		  -- integer
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

def find_idx_common4b(i, kappa, list_kappa_idx, solution_idx):
	"""
	Find the scalar products for common function with 4 solution consecutive kappas. 

	Parameters:
	i 	  		   -- integer
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

def find_scalar_sol(n, scalar, solution_idx):
	"""
	Find the list scalar products for solution function from init_scalar. 

	Parameters:
	n 	         -- integer
	scalar 		 -- list of Decimal
	solution_idx -- list of Decimal

	Return:
	solution -- list of Decimal
	"""
	solution = [scalar[i][solution_idx[i]] for i in range(n)]

	return solution

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

def reorder_common2b(solution_init, common2b_list):
	"""
	Reorder elements in common3b_init according to the sorted solution_init.

	Parameters:
	solution_init -- list of Decimal
	common2b_list -- list of lists of Decimal

	Return:
	common2b_new -- list of lists of Decimal
	"""	
	common2b_new = []

	n = len(solution_init)

	for i in range(n):
		common2b_new.append((common2b_list[i][:i] + common2b_list[i][i+1:])) # rm elem i in list

		# Retrieve idx of common scalar products
		idx = solution_init.index(common2b_list[i][i])

		common2b_new[i].insert(idx, common2b_list[i][i])

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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	i   		  -- integer
	n    		  -- integer
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm	      -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	norm 		  -- integer
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
	start    -- integer
	stop     -- integer
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
	stop     -- integer
	common   -- list of Decimal
	solution -- list of Decimal

	Return:
	sub -- list of Decimal
	"""
	count = sum(x == y for x, y in zip(common[:stop], solution[:stop]))

	return count

def append_wr(j, value, interval, wr):
	"""
	Append value to wr if in interval. 

	Parameters:
	j 		 -- integer
	value    -- Decimal
	interval -- list
	wr       -- list of Decimal

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
	rank_wr -- list of tuples
	"""
	# Transform Decimal into float
	# getcontext().prec = 6
	wr = [(float(w), string) for w, string in wr]
		
	# Sort by w
	wr.sort(key=lambda x: x[0])
			
	rank, wr = find_rank(wr)
	
	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr   = [i for i in zip(wr, wr)]
			
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr   = list(itertools.chain.from_iterable(wr))
			
	# Extract abscissas for intersection points
	wr = [tuple[0] for tuple in wr]
	wr.insert(0, 0)
	wr.append(1)
		
	# Group to save and manipulate later on
	rank_wr = [i for i in zip(rank, wr)]

	return rank_wr

def insert_w0(list_tup, list_w0):
	"""
	Insert w0 values in list_tup to have a sorted list of tuples.

	Parameters:
	list_tup -- list of tuples
	list_w0  -- list of Decimal

	Return:
	new_list_tup -- list of tuples
	"""	
	list_w0 = [float(w0) for w0 in list_w0]
	list_aux = []

	for tup in list_tup:
		tup = list(tup)

		for w0 in list_w0:
			if (w0 > tup[0]) and (w0 < tup[-1]):
				bisect.insort(tup, w0)

		list_aux.append(tup)

	new_list_tup = []

	for sublist in list_aux:
		aux = [elem for elem in zip(sublist, sublist)]
		aux = list(itertools.chain.from_iterable(aux)) # Un-nest list
		aux = aux[1:-1]  # Cut the edges
		aux = [elem for elem in zip(aux[::2], aux[1::2])]

		new_list_tup += aux

	return new_list_tup

def pr_success(df, num_sim):
	"""
	Computes probability of success for a w value.
	Count from the min w value to the max one the probability of success
	according to the number of simulations (pr_sim).

	Parameters:
	df      -- DataFrame
	num_sim -- integer

	Return:
	df -- DataFrame
	"""
	# Probability of a simulation
	pr_sim = 1 / num_sim

	# Drop succ column
	df.drop(['succ'], axis=1, inplace=True)

	count0 = df['w'].loc[df['w'] == 0].count()
	pr_init = count0 * pr_sim

	# print('\n+----- nb zeros:', count0)
	# print('+----- pr_init:', pr_init, '\n')

	# Drop rows with 0 and -1 for w
	df = df[(df != 0).all(1)]
	df = df[(df != -1).all(1)]

	df.loc['w'] = [0]

	# Sort by absolute value, ascending order
	df = df.iloc[(df['w'].abs()).argsort()].reset_index(drop=True)

	# Add a new pr_sucess column
	df.loc[0, 'pr_success'] = df.loc[0, 'w']
	df.loc[0, 'pr_success'] = pr_init

	# Compute probabilities of success
	for i in range(1, len(df)):
		if df.at[i, 'w'] < 0:
			df.loc[i, 'pr_success'] = df.loc[i-1, 'pr_success'] - pr_sim
		else:
			df.loc[i, 'pr_success'] = df.loc[i-1, 'pr_success'] + pr_sim

	# Round float in pr_success column
	decimals = 8 
	df['pr_success'] = df['pr_success'].apply(lambda x: round(x, decimals))
	df['w'] = df['w'].abs()
	
	return df

def list_to_frames(num_rank, rank_wr_list):
	"""
	Transform list in Dataframes for plotting.

	Parameter:
	rank_wr_list -- list of Decimal

	Return:
	frames -- list of Dataframes
	"""
	# Prepare dataset by sorting by rank first
	rank_wr_list = sorted(rank_wr_list)
	rank_wr_df = pd.DataFrame(rank_wr_list, columns=['rank', 'wr'])

	# Drop rows with zero values for w
	rank_wr_df = rank_wr_df[(rank_wr_df != 0).all(1)]

	# Sort by absolute value, descending order, for each rank
	frames = []
	for rank in range(1, num_rank + 1):
		df = rank_wr_df.loc[rank_wr_df['rank'] == rank]
		df = df.iloc[(-df['wr'].abs()).argsort()].reset_index(drop=True)
		frames.append(df)

	return frames

def succ_bitsK(num, w, count_diff, rank1_df, mean_interval_rank1):
	"""
	Select intervals where successful recovery of num K bits.

	Parameters:
	num 				-- integer
	w 					-- Decimal
	count_diff 			-- integer
	rank1_df 			-- Dataframe
	mean_interval_rank1 -- list

	Return: None
	"""
	if (count_diff == num):
		rank1_df.at[2 * mean_interval_rank1.index(w), 'succ'] = True
		rank1_df.at[2 * mean_interval_rank1.index(w) + 1, 'succ'] = True

def concat_df(succ_df, rank1_df):
	"""
	Concatenation

	Parameters:
	succ_df  -- Dataframe
	rank1_df -- Dataframe

	Return:
	succ_df -- Dataframe
	"""
	if not rank1_df.empty:
		succ_df = pd.concat([succ_df, rank1_df], ignore_index=True, sort=False)

	return succ_df

def drop_fail(rank1_df):
	"""
	Save only successes.

	Parameter:
	rank1_df -- Dataframe

	Return: None
	"""
	rank1_df = rank1_df.loc[rank1_df['succ'] == True]
	rank1_df.reset_index(drop=True, inplace=True)

	return rank1_df

def sign_interval(succ_df, rank1_df):
	"""
	Put positive float at beginning interval and negative sign at end interval

	Parameters:
	succ_df  -- Dataframe
	rank1_df -- Dataframe

	Return:
	succ_df -- Dataframe
	"""
	if not rank1_df.empty:
		succ_df['w'] = succ_df['w'].apply(lambda x: ((-1)**(succ_df.loc[succ_df['w'] == x].index[0])*x))

	return succ_df

def create_fig(K, kappa, frames):
	"""
	Create figure to plot or to save

	Parameters:
	K 	   -- string
	kappa  -- string
	frames -- list of DataFrames

	Return: None
	"""
	fig = plt.figure()

	colors = ['tab:brown', 'tab:purple', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue']

	num_bitK = 0 
	for frame in frames:
		if not frame.empty:
			plt.plot(frame['w'], frame['pr_success'], '-', color=colors[num_bitK], markersize=2, label='%s-bit K' % (num_bitK))
		num_bitK += 1

	plt.legend(loc='upper left')
	plt.title(r'cDoM with absolute scalar product | kappa=%s & K=%s' %(kappa, K))
	plt.xlabel('w')
	plt.ylabel('Probability of success')
	plt.show()

def write_csv(K, kappa, num_sim, frames):
	"""
	Write Dataframes in CSV files.

	Parameters:
	K       -- string
	kappa   -- string
	num_sim -- integer
	frames  -- list of DataFrames

	Return: None
	"""
	num_bitK = 0
	for frame in frames:
		file_name = r'./csv/1x5bits_6bits_abs_num-sim=%s_K=%s_kappa=%s_%sK.csv' % (num_sim, K, kappa, num_bitK)
		frame.to_csv(file_name, encoding='utf-8', index=False)
		num_bitK += 1

def sim_1x5bits(K, kappa, num_sim):
	"""
	Compute simulation scalar products and 
	return a list of ranks and intersection values for several simulations. 

	Parameters:
	K 		-- string
	kappa   -- string
	num_sim -- integer

	Return:
	rank_wr_list -- list of tuples
	"""
	# Length of signal part
	n = 5

	# Message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))
	norm = Decimal(2**n)

	# Signal reference vectors
	S_ref = gen_S_ref(mu, n)
	
	# Save the results of simulations
	rank_wr_list = []

	# Save the results of the simulations
	succ_df_5K, succ_df_4K, succ_df_3K = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	succ_df_2K, succ_df_1K, succ_df_0K = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	for j in range(num_sim):
	
		# Noise power consumption for m bits
		R = noise(mu)

		# Initial values of scalar product
		scalar_init = gen_scalar_init(n, R, S_ref)

		# Transform numpy.ndarray into list
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

		# ------------------------------------------------------------------------------------------- #
		
		# Retrieve corresponding scalar products
		solution_init = [init_scalar[i][solution_idx[i]] for i in range(n)]
		nonsol_init   = find_init(n, init_scalar, nonsol_idx)
		common4b_init = find_init(n, init_scalar, common4b_idx)
		common3b_init = find_init(n, init_scalar, common3b_idx)
		common2b_init = find_init(n, init_scalar, common2b_idx)
		
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
		common2b_new  = reorder_common2b(solution_init, common2b_init[::2]) + reorder_common2b(solution_init, common2b_init[1::2])

		# ------------------------------------------------------------------------------------------- #

		# List of all intersection values
		wr = []

		interval = [list_w0[4], 1]

		# Intersections between solution function and nonsol functions
		for j in range(len(nonsol_init)):
			wr6_nonsol = find_wr6_nonsol(norm, solution_init, nonsol_init[j])
			append_wr(6, wr6_nonsol, interval, wr)

		# Intersections between solution function and common4b functions
		for j in range(len(common4b_init)):
			wr6_common4b = find_wr6_common4(j, n, norm, solution_init, common4b_init[j])
			append_wr(6, wr6_common4b, interval, wr)

		# Intersections between solution function and common3b functions
		for j in range(len(common3b_init)):
			wr6_common3b = find_wr6_common3(j, n, norm, solution_init, common3b_init[j])
			append_wr(6, wr6_common3b, interval, wr)

		# Intersections between solution function and common2b functions
		for j in range(len(common2b_init) // 2):
			for l in range(2):
				wr6_common2b = find_wr6_common2(j, n, norm, solution_init, common2b_init[2*j + l])
				append_wr(6, wr6_common2b, interval, wr)

		# ------------------------------------------------------------------------------------------- #

		if list_w0[4] != 0:

			interval = [list_w0[3], list_w0[4]]

			# Intersections between solution function and nonsol functions
			for j in range(len(nonsol_init)):
				wr5_nonsol = find_wr5_nonsol(norm, solution_init, nonsol_init[j])
				append_wr(5, wr5_nonsol, interval, wr)

			# Intersection between solution function and common4b function
			for sublist in common4b_new:

				if count_common(4, sublist, solution_init) == 2: #I: ai + ai1 - ai2

					i = common4b_new.index(sublist)
					common4b_inc = common_elem(0, 4, sublist, solution_init)      # ai, ai1
					common4b_dec = sublist[-1]									  # ai2
					common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

					wr5_common4b = find_wr5_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
					append_wr(5, wr5_common4b, interval, wr)

			# Intersection between solution function and common3b function
			for sublist in common3b_new:

				if count_common(4, sublist, solution_init) == 2: # M

					i = common3b_new.index(sublist)
					common3b_inc = common_elem(0, 4, sublist, solution_init)      # ai, ai1
					common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

					wr5_common3b = find_wr5bis_common3(norm, solution_init, common3b_inc, common3b_non)
					append_wr(5, wr5_common3b, interval, wr)

				if count_common(4, sublist, solution_init) == 1: # L

					i = common3b_new.index(sublist)
					common3b_inc = common_elem(0, 4, sublist, solution_init)      # ai
					common3b_dec = sublist[-1]									  # ai1
					common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

					wr5_common3b = find_wr5_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
					append_wr(5, wr5_common3b, interval, wr)

			# Intersection between solution function and common2b function
			for sublist in common2b_new:

				if count_common(4, sublist, solution_init) == 1: # P
					i = common2b_new.index(sublist)
					wr5_common2b = find_wr5bis_common2(i, n, norm, solution_init, sublist)
					append_wr(5, wr5_common2b, interval, wr)

				if count_common(4, sublist, solution_init) == 0: # N
					i = common2b_new.index(sublist)
					wr5_common2b = find_wr5_common2(i, n, norm, solution_init, sublist)
					append_wr(5, wr5_common2b, interval, wr)

			# ------------------------------------------------------------------------------------------- #

			if list_w0[3] != 0:
			
				interval = [list_w0[2], list_w0[3]]

				# Intersections between solution function and nonsol functions
				for j in range(len(nonsol_init)):
					wr4_nonsol = find_wr4_nonsol(norm, solution_init, nonsol_init[j])
					append_wr(4, wr4_nonsol, interval, wr)

				# Intersection between solution function and common4b function
				for sublist in common4b_new:
					
					if count_common(3, sublist, solution_init) == 3: # J: ai + ai1 + ai2
						
						i = common4b_new.index(sublist)
						common4b_inc = common_elem(0, 3, sublist, solution_init)      # ai, ai1, ai2
						common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

						wr4_common4b = find_wr4bis_common4(norm, solution_init, common4b_inc, common4b_non)
						append_wr(4, wr4_common4b, interval, wr)

					elif count_common(3, sublist, solution_init) == 1: # H

						i = common4b_new.index(sublist)
						common4b_inc = common_elem(0, 3, sublist, solution_init)  # ai
						common4b_dec = common_elem(3, 5, sublist, solution_init)  # ai1, ai2
						common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

						wr4_common4b = find_wr4_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
						append_wr(4, wr4_common4b, interval, wr)

				# Intersection between solution function and common3b function
				for sublist in common3b_new:
					
					if count_common(3, sublist, solution_init) == 2: # M

						i = common3b_new.index(sublist)
						common3b_inc = common_elem(0, 3, sublist, solution_init)      # ai, ai1
						common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

						wr4_common3b = find_wr4ter_common3(norm, solution_init, common3b_inc, common3b_non)
						append_wr(4, wr4_common3b, interval, wr)

					if count_common(3, sublist, solution_init) == 1: # L

						i = common3b_new.index(sublist)
						common3b_inc = common_elem(0, 3, sublist, solution_init)   # ai
						common3b_dec = common_elem(3, 5, sublist, solution_init)   # ai1
						common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

						wr4_common3b = find_wr4bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
						append_wr(4, wr4_common3b, interval, wr)
							
					if count_common(3, sublist, solution_init) == 0: # K

						i = common3b_new.index(sublist)
						common3b_dec = sublist[-2:]  # ai, ai1
						common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

						wr4_common3b = find_wr4_common3(norm, solution_init, common3b_dec, common3b_non)
						append_wr(4, wr4_common3b, interval, wr)
							
				# Intersection between solution function and common2b function
				for sublist in common2b_new:

					if count_common(3, sublist, solution_init) == 0: # N
						wr4_common2b = find_wr4_common2(i, n, norm, solution_init, sublist)
						append_wr(4, wr4_common2b, interval, wr)

				# ------------------------------------------------------------------------------------------- #

				if list_w0[2] != 0:
			
					interval = [list_w0[1], list_w0[2]]

					# Intersections between solution function and nonsol functions
					for j in range(len(nonsol_init)):
						wr3_nonsol = find_wr3_nonsol(norm, solution_init, nonsol_init[j])
						append_wr(3, wr3_nonsol, interval, wr)

					# Intersection between solution function and common4b function
					for sublist in common4b_new:
						
						if count_common(2, sublist, solution_init) == 2: # I
							
							i = common4b_new.index(sublist)
							common4b_inc = common_elem(0, 2, sublist, solution_init) # ai, ai1
							common4b_dec = common_elem(2, 5, sublist, solution_init) # ai2
							common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

							wr3_common4b = find_wr3bis_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
							append_wr(3, wr3_common4b, interval, wr)

						elif count_common(2, sublist, solution_init) == 0: # G

							i = common4b_new.index(sublist)
							common4b_dec = common_elem(2, 5, sublist, solution_init)  # ai, ai1, ai2
							common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

							wr3_common4b = find_wr3_common4(norm, solution_init, common4b_dec, common4b_non)
							append_wr(3, wr3_common4b, interval, wr)

					# Intersection between solution function and common3b function
					for sublist in common3b_new:
						
						if count_common(2, sublist, solution_init) == 2: # M

							i = common3b_new.index(sublist)
							common3b_inc = sublist[:2]      # ai, ai1
							common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

							wr3_common3b = find_wr3ter_common3(norm, solution_init, common3b_inc, common3b_non)
							append_wr(3, wr3_common3b, interval, wr)

						if count_common(2, sublist, solution_init) == 1: # L

							i = common3b_new.index(sublist)
							common3b_inc = common_elem(0, 2, sublist, solution_init)   # ai
							common3b_dec = common_elem(2, 5, sublist, solution_init)   # ai1
							common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

							wr3_common3b = find_wr3bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
							append_wr(3, wr3_common3b, interval, wr)

						if count_common(2, sublist, solution_init) == 0: # K

							i = common3b_new.index(sublist)
							common3b_dec = common_elem(2, 5, sublist, solution_init) # ai, ai1
							common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

							wr3_common3b = find_wr3_common3(norm, solution_init, common3b_dec, common3b_non)
							append_wr(3, wr3_common3b, interval, wr)

					# Intersection between solution function and common2b function
					for sublist in common2b_new:

						if count_common(2, sublist, solution_init) == 1: # P
							wr3_common2b = find_wr3_common2(i, n, norm, solution_init, sublist)
							append_wr(3, wr3_common2b, interval, wr)

					# ------------------------------------------------------------------------------------------- #

					if list_w0[1] != 0:
					
						interval = [list_w0[0], list_w0[1]]

						# Intersections between solution function and nonsol functions
						for j in range(len(nonsol_init)):
							wr2_nonsol = find_wr2_nonsol(norm, solution_init, nonsol_init[j])
							append_wr(2, wr2_nonsol, interval, wr)

						# Intersection between solution function and common4b function
						for sublist in common4b_new:
							
							if count_common(1, sublist, solution_init) == 1: # H
								
								i = common4b_new.index(sublist)
								common4b_inc = common_elem(0, 1, sublist, solution_init) # ai
								common4b_dec = common_elem(1, 5, sublist, solution_init) # ai1, ai2
								common4b_non = [common4b_init[i][(i + 3) % n]] + [common4b_init[i][(i + 4) %n]] # b, c

								wr2_common4b = find_wr2_common4(norm, solution_init, common4b_inc, common4b_dec, common4b_non)
								append_wr(2, wr2_common4b, interval, wr)

						# Intersection between solution function and common3b function
						for sublist in common3b_new:

							if  count_common(1, sublist, solution_init) == 1: # L

								i = common3b_new.index(sublist)
								common3b_inc = sublist[1]   # ai
								common3b_dec = common_elem(1, 5, sublist, solution_init)   # ai1
								common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

								wr2_common3b = find_wr2bis_common3(norm, solution_init, common3b_inc, common3b_dec, common3b_non)
								append_wr(2, wr2_common3b, interval, wr)

							if count_common(1, sublist, solution_init) == 0: # K

								i = common3b_new.index(sublist)
								common3b_dec = common_elem(1, 5, sublist, solution_init) # ai, ai1
								common3b_non = [common3b_init[i][(i + 2) % n]] +  [common3b_init[i][(i + 3) % n]] + [common3b_init[i][(i + 4) %n]] # b, c, d

								wr2_common3b = find_wr2_common3(norm, solution_init, common3b_dec, common3b_non)
								append_wr(2, wr2_common3b, interval, wr)

						# Intersection between solution function and common2b function
						for sublist in common2b_new:

							if count_common(1, sublist, solution_init) == 1: # P
								i = common2b_new.index(sublist)
								wr2_common2b = find_wr2bis_common2(i, n, norm, solution_init, sublist)
								append_wr(2, wr2_common2b, interval, wr)

							if count_common(1, sublist, solution_init) == 0: # N
								i = common2b_new.index(sublist)
								wr2_common2b = find_wr2_common2(i, n, norm, solution_init, sublist)
								append_wr(2, wr2_common2b, interval, wr)

						# ------------------------------------------------------------------------------------------- #

						if list_w0[0] != 0:
						
							interval = [0, list_w0[0]]

							# Intersections between solution function and nonsol functions
							for j in range(len(nonsol_init)):
								wr1_nonsol = find_wr1_nonsol(norm, solution_init, nonsol_init[j])
								append_wr(1, wr1_nonsol, interval, wr)

							# Intersections between solution function and common4b functions
							for j in range(len(common4b_init)):
								wr1_common4b = find_wr1_common4(j, n, norm, solution_init, common4b_init[j])
								append_wr(1, wr1_common4b, interval, wr)

							# Intersections between solution function and common3b functions
							for j in range(len(common3b_init)):
								wr1_common3b = find_wr1_common3(j, n, norm, solution_init, common3b_init[j])
								append_wr(1, wr1_common3b, interval, wr)

							# Intersections between solution function and common2b functions
							for j in range(len(common2b_init) // 2):
								for l in range(2):
									wr1_common2b = find_wr1_common2(j, n, norm, solution_init, common2b_init[2*j + l])
									append_wr(1, wr1_common2b, interval, wr)

		# Determine the intervals for the ranks per noise vector R
		rank_wr    = compute_rank_wr(wr)
		rank_wr_df = pd.DataFrame(rank_wr, columns=['rank', 'w'])

		# ------------------------------------------------------------------------------------------- #

		# Select 1st rank only and creat succ column in new Dataframe
		rank_wr_df1 = rank_wr_df.loc[rank_wr_df['rank'] == 1]

		rank1_df = rank_wr_df1.assign(succ=bool(0))
		rank1_df.drop(['rank'], axis=1, inplace=True)
		rank1_df.reset_index(drop=True, inplace=True)

		# Create w intervals for 1st rank
		interval_rank1 = [w for w in zip(rank1_df['w'][::2], rank1_df['w'][1::2])]
		interval_rank1_w0 = insert_w0(interval_rank1, list_w0) # insert w0 if w0 in interval_rank1

		# Compute mean of intervals
		mean_interval_rank1 = [np.mean(tup) for tup in interval_rank1_w0]

		# if insertion of w0 in interval_rank1
		if interval_rank1 != interval_rank1_w0:
			aux = list(itertools.chain.from_iterable(interval_rank1_w0)) 
			rank1_df = pd.DataFrame(aux, columns=['w'])
			rank1_df = rank1_df.assign(succ=bool(0))
			rank1_df.reset_index(drop=True, inplace=True)

		# Dataframes to save results of one simulatiom
		rank1_df_5K, rank1_df_4K, rank1_df_3K = rank1_df.copy(), rank1_df.copy(), rank1_df.copy()
		rank1_df_2K, rank1_df_1K, rank1_df_0K = rank1_df.copy(), rank1_df.copy(), rank1_df.copy()

		# ------------------------------------------------------------------------------------------- #

		# Signal power consumption for a key
		S = gen_S(mu, n, kappa, K)

		for w in mean_interval_rank1:
			scalar_mean = gen_scalar(n, w, R, S, S_ref)
			mean_scalar = [sublist.tolist() for sublist in scalar_mean]

			# Combine scalar products for hypothetical kappa solutions
			mean_scalar_hyp = [mean_scalar[i][solution_idx[i]] for i in range(n)]

			# Transform sign into bit
			mean_hyp_sign = [int(s < 0) for s in mean_scalar_hyp]

			# K hypothesis = sign XOR kappa
			hyp_K = [str(mean_hyp_sign[i] ^ int(kappa[i])) for i in range(n)]
			hyp_K_str = ''.join(hyp_K)
			list_K = list(K)

			# Set success for interval where (5 - x) bits retrieved
			count_diff = sum(1 for i, j in zip(hyp_K, list_K) if i != j)

			succ_bitsK(0, w, count_diff, rank1_df_5K, mean_interval_rank1)
			succ_bitsK(1, w, count_diff, rank1_df_4K, mean_interval_rank1)
			succ_bitsK(2, w, count_diff, rank1_df_3K, mean_interval_rank1)
			succ_bitsK(3, w, count_diff, rank1_df_2K, mean_interval_rank1)
			succ_bitsK(4, w, count_diff, rank1_df_1K, mean_interval_rank1)
			succ_bitsK(5, w, count_diff, rank1_df_0K, mean_interval_rank1)

		# Save only successes
		rank1_df_5K = drop_fail(rank1_df_5K)
		rank1_df_4K = drop_fail(rank1_df_4K)
		rank1_df_3K = drop_fail(rank1_df_3K)
		rank1_df_2K = drop_fail(rank1_df_2K)
		rank1_df_1K = drop_fail(rank1_df_1K)
		rank1_df_0K = drop_fail(rank1_df_0K)

		# Save results of one simulation
		succ_df_5K = concat_df(succ_df_5K, rank1_df_5K)
		succ_df_4K = concat_df(succ_df_4K, rank1_df_4K)
		succ_df_3K = concat_df(succ_df_3K, rank1_df_3K)
		succ_df_2K = concat_df(succ_df_2K, rank1_df_2K)
		succ_df_1K = concat_df(succ_df_1K, rank1_df_1K)
		succ_df_0K = concat_df(succ_df_0K, rank1_df_0K)

	# Put positive float at beginning interval and negative sign at end interval
	succ_df_5K = sign_interval(succ_df_5K, rank1_df_5K)
	succ_df_4K = sign_interval(succ_df_4K, rank1_df_4K)
	succ_df_3K = sign_interval(succ_df_3K, rank1_df_3K)
	succ_df_2K = sign_interval(succ_df_2K, rank1_df_2K)
	succ_df_1K = sign_interval(succ_df_1K, rank1_df_1K)
	succ_df_0K = sign_interval(succ_df_0K, rank1_df_0K)

	return succ_df_5K, succ_df_4K, succ_df_3K, succ_df_2K, succ_df_1K, succ_df_0K

def sim_1x5bits_to_csv(K, kappa, num_sim):
	"""
	Plot probabilities of success for all possibilities for several simulations. 
	
	Parameters:
	K 		-- string
	kappa   -- string
	num_sim -- integer
	
	Return: None
	"""
	succ_df_5K, succ_df_4K, succ_df_3K, succ_df_2K, succ_df_1K, succ_df_0K = sim_1x5bits(K, kappa, num_sim)

	if not succ_df_5K.empty:
		succ_df_5K = pr_success(succ_df_5K, num_sim)

	if not succ_df_4K.empty:
		succ_df_4K = pr_success(succ_df_4K, num_sim)

	if not succ_df_3K.empty:
		succ_df_3K = pr_success(succ_df_3K, num_sim)

	if not succ_df_2K.empty:
		succ_df_2K = pr_success(succ_df_2K, num_sim)

	if not succ_df_1K.empty:
		succ_df_1K  = pr_success(succ_df_1K, num_sim)

	if not succ_df_0K.empty:
		succ_df_0K  = pr_success(succ_df_0K, num_sim)

	frames = [succ_df_0K, succ_df_1K, succ_df_2K, succ_df_3K, succ_df_4K, succ_df_5K]

	# create_fig(K, kappa, frames)
	write_csv(K, kappa, num_sim, frames)
	
def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa, and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='5-bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='5-bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations to perform')

	args = parser.parse_args()

	if len(args.K) != 5 or len(args.kappa) != 5:
		print('\n**ERROR**')
		print('Required length of K and kappa: 5\n')

	else:
		# Time option
		start_t = time.perf_counter()
		
		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		sim_1x5bits_to_csv(K, kappa, num_sim)
		
		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))
