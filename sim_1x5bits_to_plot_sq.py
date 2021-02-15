#!/usr/bin/env python

"""DPA ON KECCAK 5-BIT CHI ROW & 1-BIT K

Save the figures for a simulation which display the outcome of a test
according to the SNR, in order to recover kappa bits. 

Within one simulation, we perform 5 separated experiments to guess 2 bits 
of kappa each, with the scalar product, and a fixed noise power consumption vector.  
We combine the individual results to recover the bits of kappa. 
"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__     = "2.3"

import numpy as np
import itertools
import xarray as xr
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
	key_i = kappa_i XOR K

	Parameters:
	i     -- integer in [0,n-1]
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
	kappa -- string
	K 	  -- string
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
	list_idx 	-- list of list

	Return:
	list_init -- list of Decimal
	"""	
	list_init = []

	for j in range(len(list_idx)):
		init = [init_scalar[i][list_idx[j][i]] for i in range(n)]
		list_init.append(init)

	return list_init

def find_sq_w0(solution_init, norm):
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

def fct_sq_solution(w, n, norm, solution_init):
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
	
	y = y0**2 + y1**2 + y2**2 + y3**2 + y4**2

	return y / (Decimal(n)*(norm**2))

def fct_sq_common4(w, i, n, norm, common4b_init):
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

	y = y_a0**2 + y_a1**2 + y_a2**2 + y_b**2 + y_c**2

	return y / (Decimal(n)*(norm**2))

def fct_sq_common3(w, i, n, norm, common3b_init):
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

	y = y_a0**2 + y_a1**2 + y_b**2 + y_c**2 + y_d**2

	return y / (Decimal(n)*(norm**2))

def fct_sq_common2(w, i, n, norm, common2b_init):
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

	y = y_a**2 + y_b**2 + y_c**2 + y_d**2 + y_e**2

	return y / (Decimal(n)*(norm**2))

def fct_sq_nonsol(w, n, norm, nonsol_init):
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

	y = y_b**2 + y_c**2 + y_d**2 + y_e**2 + y_f**2
		
	return y / (Decimal(n)*(norm**2))

def find_sq_min(n, norm, solution_init):
	"""
	Find the minimum point for the scalar product of the correct guess.
	
	Parameters:
	n 			-- integer
	norm 		-- Decimal
	nonsol_init -- list of Decimal
	
	Return:
	w_min -- Decimal
	"""	
	a0 = solution_init[0]
	a1 = solution_init[1]
	a2 = solution_init[2]
	a3 = solution_init[3]
	a4 = solution_init[4]

	# A = SUM (norm - ai)^2
	A = (norm - a0)**2 + (norm - a1)**2 + (norm - a2)**2 + (norm - a3)**2 + (norm - a4)**2

	# B = SUM (aj (norm - ai)
	B = a0 * (norm - a0) +  a1 * (norm - a1) + a2 * (norm - a2) + a3 * (norm - a3) + a4 * (norm - a4)

	# C = SUM aj^2
	C = a0**2 + a1**2 + a2**2 + a3**2 + a4**2

	w_min = Decimal()
	
	if (A != 0):
		# w_min = - 2 B / 2 A
		w_min = (- B) / (A)	
	else:
		w_min = None

	return w_min

def find_wr_nonsol(norm, solution_init, nonsol_init):
	"""
	Find the point when for the scalar product of the solution 
	equals the scalar product of a nonsolutions.
	
	F(w) = (A - D) w^2 + 2 (B + D) w + (C - D) = 0 for which w? 
	
	Parameters:
	norm 		  -- Decimal
	solution_init -- list of Decimal
	nonsol_init   -- list of Decimal
	
	Return:
	w1 -- Decimal
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

	# A = SUM (norm - ai)^2
	A = (norm - a0)**2 + (norm - a1)**2 + (norm - a2)**2 + (norm - a3)**2 + (norm - a4)**2

	# B = SUM (aj (norm - ai)
	B = a0 * (norm - a0) +  a1 * (norm - a1) + a2 * (norm - a2) + a3 * (norm - a3) + a4 * (norm - a4)

	# C = SUM aj^2
	C = a0**2 + a1**2 + a2**2 + a3**2 + a4**2
	
	# D = b^2 + c^2 + d^2 + e^2 + f^2
	D = b**2 + c**2 + d**2 + e**2 + f**2
	
	# discriminant = 4 (B + D)^2 - 4 (A - D) (C - D)
	discriminant = Decimal(4) * ((B + D)**2) - (Decimal(4) * (A - D) * (C - D))
	
	w1 = Decimal()
	w2 = Decimal()
	
	if (A != D) and (discriminant >= 0):
		
		sqrt_discriminant = Decimal(discriminant).sqrt()
	
		# w1 = - 2(B + D) + sqrt_discriminant / 2 (A - D)
		w1 = (- Decimal(2) * (B + D) + sqrt_discriminant) / (Decimal(2) * (A - D))

		# w2 = - 2(B + D) - sqrt_discriminant / 2 (A - D)
		w2 = (- Decimal(2) * (B + D) - sqrt_discriminant) / (Decimal(2) * (A - D))
		
	else:
		w1 = None
		w2 = None

	return w1, w2

def find_wr_common4(i, n, norm, solution_init, common4b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an common guess with 4 consecutive kappa bits.
	
	F(w) = (A - E) w^2 + 2 (B - F) w + (C - G) = 0 for which w? 
	
	Parameter:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common4b_init -- list of Decimal
	
	Return:
	w1 -- Decimal
	w2 -- Decimal
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

	# A = SUM (norm - ai)^2
	A = (norm - a0)**2 + (norm - a1)**2 + (norm - a2)**2 + (norm - a3)**2 + (norm - a4)**2

	# B = SUM (aj (norm - ai)
	B = a0 * (norm - a0) +  a1 * (norm - a1) + a2 * (norm - a2) + a3 * (norm - a3) + a4 * (norm - a4)

	# C = SUM aj^2
	C = a0**2 + a1**2 + a2**2 + a3**2 + a4**2
	
	# E = SUM (norm - ai)^2
	E = (norm - ai)**2 + (norm - ai1)**2 + (norm - ai2)**2 + b**2 + c**2

	# F = SUM (aj (norm - ai)
	F = ai * (norm - ai) +  ai1 * (norm - ai1)  +  ai2 * (norm - ai2) - b**2 - c**2

	# G = ai^2 + ai1^2 + ai2^2 + b^2 + c^2
	G = ai**2 + ai1**2 + ai2**2 + b**2 + c**2

	# discriminant = 4 (B - F)^2 - 4 (A - E) (C - G)
	discriminant = Decimal(4) * ((B - F)**2) - (Decimal(4) * (A - E) * (C - G))
	
	w1 = Decimal()
	w2 = Decimal()
	
	if (A != E) and (discriminant >= 0):
		
		sqrt_discriminant = Decimal(discriminant).sqrt()
	
		# w1 = - 2(B - F) + sqrt_discriminant / 2 (A - E)
		w1 = (- Decimal(2) * (B - F) + sqrt_discriminant) / (Decimal(2) * (A - E))

		# w2 = - 2(B - F) - sqrt_discriminant / 2 (A - E)
		w2 = (- Decimal(2) * (B - F) - sqrt_discriminant) / (Decimal(2) * (A - E))
		
	else:
		w1 = None
		w2 = None

	return w1, w2

def find_wr_common3(i, n, norm, solution_init, common3b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an common guess with 3 consecutive kappa bits.
	
	F(w) = (A - H) w^2 + 2 (B - I) w + (C - J) = 0 for which w? 
	
	Parameter:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common3b_init -- list of Decimal
	
	Return:
	w1 -- Decimal
	w2 -- Decimal
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

	# A = SUM (norm - ai)^2
	A = (norm - a0)**2 + (norm - a1)**2 + (norm - a2)**2 + (norm - a3)**2 + (norm - a4)**2

	# B = SUM (aj (norm - ai)
	B = a0 * (norm - a0) +  a1 * (norm - a1) + a2 * (norm - a2) + a3 * (norm - a3) + a4 * (norm - a4)

	# C = SUM aj^2
	C = a0**2 + a1**2 + a2**2 + a3**2 + a4**2
	
	# H = (norm - ai)^2 + (norm - ai1)^2 + b^2 + c^2 + d^2
	H = (norm - ai)**2 + (norm - ai1)**2 + b**2 + c**2 + d**2

	# I = ai(norm - ai) + ai1(norm - ai1) - b^2 - c^2 - d^2
	I = (ai * (norm - ai) +  ai1 * (norm - ai1)  - b**2 - c**2 - d**2)

	# J = ai^2 + ai1^2 + b^2 + c^2 + d^2
	J = ai**2 + ai1**2 + b**2 + c**2 + d**2

	# discriminant = 4 (B - F)^2 - 4 (A - E) (C - G)
	discriminant = Decimal(4) * ((B - I)**2) - (Decimal(4) * (A - H) * (C - J))
	
	w1 = Decimal()
	w2 = Decimal()
	
	if (A != H) and (discriminant >= 0):
		
		sqrt_discriminant = Decimal(discriminant).sqrt()
	
		# w1 = - 2(B - I) + sqrt_discriminant / 2 (A - H)
		w1 = (- Decimal(2) * (B - I) + sqrt_discriminant) / (Decimal(2) * (A - H))

		# w2 = - 2(B - I) - sqrt_discriminant / 2 (A - H)
		w2 = (- Decimal(2) * (B - I) - sqrt_discriminant) / (Decimal(2) * (A - H))
		
	else:
		w1 = None
		w2 = None

	return w1, w2

def find_wr_common2(i, n, norm, solution_init, common2b_init):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of an common guess with 2 consecutive kappa bits.
	
	F(w) = (A - K) w^2 + 2 (B - L) w + (C - M) = 0 for which w? 
	
	Parameter:
	i 			  -- integer
	n 			  -- integer
	norm 		  -- Decimal
	solution_init -- list of Decimal
	common2b_init -- list of Decimal
	
	Return:
	w1 -- Decimal
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

	# A = SUM (norm - ai)^2
	A = (norm - a0)**2 + (norm - a1)**2 + (norm - a2)**2 + (norm - a3)**2 + (norm - a4)**2

	# B = SUM (aj (norm - ai)
	B = a0 * (norm - a0) +  a1 * (norm - a1) + a2 * (norm - a2) + a3 * (norm - a3) + a4 * (norm - a4)

	# C = SUM aj^2
	C = a0**2 + a1**2 + a2**2 + a3**2 + a4**2
	
	# K = (norm - ai)^2 + b^2 + c^2 + d^2 + e^2
	K = (norm - ai)**2 + b**2 + c**2 + d**2 + e**2

	# L = ai(norm - ai) - b^2 - c^2 - d^2 - e^2
	L = (ai * (norm - ai)  - b**2 - c**2 - d**2 - e**2)

	# M = ai^2 + b^2 + c^2 + d^2 + e^2
	M = ai**2 + b**2 + c**2 + d**2 + e**2

	# discriminant = 4 (B - L)^2 - 4 (A - K) (C - M)
	discriminant = Decimal(4) * ((B - L)**2) - (Decimal(4) * (A - K) * (C - M))
	
	w1 = Decimal()
	w2 = Decimal()
	
	if (A != K) and (discriminant >= 0):

		sqrt_discriminant = Decimal(discriminant).sqrt()

		# w1 = - 2(B - L) + sqrt_discriminant / 2 (A - K)
		w1 = (- Decimal(2) * (B - L) + sqrt_discriminant) / (Decimal(2) * (A - K))

		# w2 = - 2(B - L) - sqrt_discriminant / 2 (A - K)
		w2 = (- Decimal(2) * (B - L) - sqrt_discriminant) / (Decimal(2) * (A - K))
		
	else:
		w1 = None
		w2 = None

	return w1, w2

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

		# Count number of rank increment ('wr1') 
		# and rank decrement ('wr2')
		counter_1 = sum(1 for tuple in wr if tuple[1] == ('wr1'))
		counter_2 = sum(-1 for tuple in wr if tuple[1] == ('wr2'))
		num_wr = counter_1 + counter_2

		rank_init = 1 + num_wr

		rank.append(rank_init)

		for tuple in wr:

			# If 'wr1', rank increases
			if tuple[1] == ('wr1'):
				rank.append(rank[-1] - 1)
				
			# If 'wr2', rank decreases
			if tuple[1] == ('wr2'):
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, wr

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
		
		scalar_sq = []
	
		# i-th bit of register state studied
		for i in range(n):
			
			# Global power consumption P = (1 - w) R  + w S
			P_i = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S[i])]

			# Squared scalar product <S-ref, P>^2
			scalar_i = np.dot(S_ref[i], P_i)
			scalar_sq_i = scalar_i**2
			scalar_sq += [scalar_sq_i]

		# Scalar product for i = 0
		# s0 = ['_00__', '_00__', '_00__', '_00__', '_01__', '_01__', '_01__', '_01__', 
		#		'_10__', '_10__', '_10__', '_10__', '_11__', '_11__', '_11__', '_11__',
		#		'_00__', '_00__', '_00__', '_00__', '_01__', '_01__', '_01__', '_01__', 
		#		'_10__', '_10__', '_10__', '_10__', '_11__', '_11__', '_11__', '_11__']
		s0 = [s for s in zip(scalar_sq[0], scalar_sq[0], scalar_sq[0], scalar_sq[0])]
		s0 = list(itertools.chain.from_iterable(s0))
		s0 = s0 * 2

		# Scalar product for i = 1
		# s1 = ['__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_', 
		#		'__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_',
		#		'__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_', 
		#		'__00_', '__00_', '__01_', '__01_', '__10_', '__10_', '__11_', '__11_']
		s1 = [s for s in zip(scalar_sq[1], scalar_sq[1])]
		s1 = list(itertools.chain.from_iterable(s1))
		s1 = s1 * 4

		# Scalar product for i = 2
		# s2 = ['___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11', 
		#		'___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11', 
		#		'___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11', 
		#		'___00', '___01', '___10', '___11', '___00', '___01', '___10', '___11']
		s2 = scalar_sq[2].tolist()
		s2 = s2 * 8

		# Scalar product for i = 3
		# s3 = ['0___0', '0___1', '0___0', '0___1', '0___0', '0___1', '0___0', '0___1',
		#		'0___0', '0___1', '0___0', '0___1', '0___0', '0___1', '0___0', '0___1', 
		#		'1___0', '1___1', '1___0', '1___1', '1___0', '1___1', '1___0', '1___1', 
		#		'1___0', '1___1', '1___0', '1___1', '1___0', '1___1', '1___0', '1___1']
		s3 = [(s,s) for s in zip(scalar_sq[3][::2], scalar_sq[3][1::2], scalar_sq[3][::2], 
								 scalar_sq[3][1::2], scalar_sq[3][::2], scalar_sq[3][1::2], 
								 scalar_sq[3][::2], scalar_sq[3][1::2])]
		s3 = list(itertools.chain.from_iterable(s3))
		s3 = list(itertools.chain.from_iterable(s3))

		# Scalar product for i = 4
		# s4 = ['00___', '00___', '00___', '00___', '00___', '00___', '00___', '00___', 
		#		'01___', '01___', '01___', '01___', '01___', '01___', '01___', '01___', 
		#		'10___', '10___', '10___', '10___', '10___', '10___', '10___', '10___', 
		#		'11___', '11___', '11___', '11___', '11___', '11___', '11___', '11___']
		s4 = [s for s in zip(scalar_sq[4], scalar_sq[4], scalar_sq[4], scalar_sq[4], 
							 scalar_sq[4], scalar_sq[4], scalar_sq[4], scalar_sq[4])]
		s4 = list(itertools.chain.from_iterable(s4))

		# Sum of absolute scalar products for all kappa values
		scalar5 = [s0[m] + s1[m] + s2[m] + s3[m] + s4[m] for m in range(len(mu))]

		w_init.append(w)

		scalar_00000.append(Decimal(scalar5[0]) / (Decimal(5)*(norm**2)))
		scalar_00001.append(Decimal(scalar5[1]) / (Decimal(5)*(norm**2)))
		scalar_00010.append(Decimal(scalar5[2]) / (Decimal(5)*(norm**2)))
		scalar_00011.append(Decimal(scalar5[3]) / (Decimal(5)*(norm**2)))
		scalar_00100.append(Decimal(scalar5[4]) / (Decimal(5)*(norm**2)))
		scalar_00101.append(Decimal(scalar5[5]) / (Decimal(5)*(norm**2)))
		scalar_00110.append(Decimal(scalar5[6]) / (Decimal(5)*(norm**2)))
		scalar_00111.append(Decimal(scalar5[7]) / (Decimal(5)*(norm**2)))

		scalar_01000.append(Decimal(scalar5[8]) / (Decimal(5)*(norm**2)))
		scalar_01001.append(Decimal(scalar5[9]) / (Decimal(5)*(norm**2)))
		scalar_01010.append(Decimal(scalar5[10]) / (Decimal(5)*(norm**2)))
		scalar_01011.append(Decimal(scalar5[11]) / (Decimal(5)*(norm**2)))
		scalar_01100.append(Decimal(scalar5[12]) / (Decimal(5)*(norm**2)))
		scalar_01101.append(Decimal(scalar5[13]) / (Decimal(5)*(norm**2)))
		scalar_01110.append(Decimal(scalar5[14]) / (Decimal(5)*(norm**2)))
		scalar_01111.append(Decimal(scalar5[15]) / (Decimal(5)*(norm**2)))

		scalar_10000.append(Decimal(scalar5[16]) / (Decimal(5)*(norm**2)))
		scalar_10001.append(Decimal(scalar5[17]) / (Decimal(5)*(norm**2)))
		scalar_10010.append(Decimal(scalar5[18]) / (Decimal(5)*(norm**2)))
		scalar_10011.append(Decimal(scalar5[19]) / (Decimal(5)*(norm**2)))
		scalar_10100.append(Decimal(scalar5[20]) / (Decimal(5)*(norm**2)))
		scalar_10101.append(Decimal(scalar5[21]) / (Decimal(5)*(norm**2)))
		scalar_10110.append(Decimal(scalar5[22]) / (Decimal(5)*(norm**2)))
		scalar_10111.append(Decimal(scalar5[23]) / (Decimal(5)*(norm**2)))

		scalar_11000.append(Decimal(scalar5[24]) / (Decimal(5)*(norm**2)))
		scalar_11001.append(Decimal(scalar5[25]) / (Decimal(5)*(norm**2)))
		scalar_11010.append(Decimal(scalar5[26]) / (Decimal(5)*(norm**2)))
		scalar_11011.append(Decimal(scalar5[27]) / (Decimal(5)*(norm**2)))
		scalar_11100.append(Decimal(scalar5[28]) / (Decimal(5)*(norm**2)))
		scalar_11101.append(Decimal(scalar5[29]) / (Decimal(5)*(norm**2)))
		scalar_11110.append(Decimal(scalar5[30]) / (Decimal(5)*(norm**2)))
		scalar_11111.append(Decimal(scalar5[31]) / (Decimal(5)*(norm**2)))

	# ax1.plot(w_init, scalar_00000, ':', color='black', markersize=4, label='00000')
	# ax1.plot(w_init, scalar_00001, '-.', color='black',  markersize=4, label='00001')
	# ax1.plot(w_init, scalar_00010, '-.', color='black', markersize=4, label='00010')
	# ax1.plot(w_init, scalar_00011, '-.' , color='black', markersize=4, label='00011')
	# ax1.plot(w_init, scalar_00100, '-.', color='black', markersize=4, label='00100')
	# ax1.plot(w_init, scalar_00101, '-.', color='black',  markersize=4, label='00101')
	# ax1.plot(w_init, scalar_00110, '-.', color='black', markersize=4, label='00110')
	# ax1.plot(w_init, scalar_00111, '-.', color='black', markersize=4, label='00111')

	ax1.plot(w_init, scalar_01000, '-.', color='black', markersize=4, label='01000')
	# ax1.plot(w_init, scalar_01001, '-.', color='black',  markersize=4, label='01001')
	# ax1.plot(w_init, scalar_01010, '-.', color='black', markersize=4, label='01010')
	# ax1.plot(w_init, scalar_01011, '-.', color='black', markersize=4, label='01011')
	# ax1.plot(w_init, scalar_01100, '-.', color='black', markersize=4, label='01100')
	# ax1.plot(w_init, scalar_01101, '-.', color='black',  markersize=4, label='01101')
	# ax1.plot(w_init, scalar_01110, '-.', color='black', markersize=4, label='01110')
	# ax1.plot(w_init, scalar_01111, '-.', color='black', markersize=4, label='01111')

	# ax1.plot(w_init, scalar_10000, '-.', color='black', markersize=4, label='10000')
	# ax1.plot(w_init, scalar_10001, '-.', color='black',  markersize=4, label='10001')
	# ax1.plot(w_init, scalar_10010, '-.', color='black', markersize=4, label='10010')
	# ax1.plot(w_init, scalar_10011, '-.', color='black', markersize=4, label='10011')
	# ax1.plot(w_init, scalar_10100, ':', color='black', markersize=4, label='10100')
	# ax1.plot(w_init, scalar_10101, '-.', color='black',  markersize=4, label='10101')
	# ax1.plot(w_init, scalar_10110, '-.', color='black', markersize=4, label='10110')
	# ax1.plot(w_init, scalar_10111, '-.', color='black', markersize=4, label='10111')

	# ax1.plot(w_init, scalar_11000, '-.', color='black', markersize=4, label='11000')
	# ax1.plot(w_init, scalar_11001, '-.', color='black',  markersize=4, label='11001')
	# ax1.plot(w_init, scalar_11010, '-.', color='black', markersize=4, label='11010')
	# ax1.plot(w_init, scalar_11011, '-.' , color='black', markersize=4, label='11011')
	# ax1.plot(w_init, scalar_11100, '-.', color='black', markersize=4, label='11100')
	# ax1.plot(w_init, scalar_11101, '-.', color='black',  markersize=4, label='11101')
	# ax1.plot(w_init, scalar_11110, '-.', color='black', markersize=4, label='11110')
	# ax1.plot(w_init, scalar_11111, ':', color='black', markersize=4, label='11111')

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
	Compute simulation' scalar products and plot outcomes. 
	
	Parameters:
	n  		-- integer
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

	# Transform numpy.ndarray into list
	init_scalar = [sublist.tolist() for sublist in scalar_init]

	# ------------------------------------------------------------------------------------------- #

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
	common4b_init = find_init(n, init_scalar, common4b_idx)
	common3b_init = find_init(n, init_scalar, common3b_idx)
	common2b_init = find_init(n, init_scalar, common2b_idx)
	nonsol_init   = find_init(n, init_scalar, nonsol_idx)

	# Determine the sign of initial solution value depending on the activity of the register at bit i
	xor_solution(n, K, kappa, solution_init)

	xor_common(3, n, K, kappa, common4b_init)
	xor_common(2, n, K, kappa, common3b_init)
	xor_common(1, n, K, kappa, common2b_init)

	# Find w0 for each bit of the register with tup = (w0, a)
	# Sort result by increasing order on w0
	list_tup_w0 = find_sq_w0(solution_init, len(mu))
	list_tup_w0 = sorted(list_tup_w0, key=lambda x: x[0])
	list_w0 = [tup[0] for tup in list_tup_w0]

	# ------------------------------------------------------------------------------------------- #

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": [2,1]}, sharex=True)

	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)

	# Abscissas
	interval = [0, 1]
	precision = 1000
	w_list = [(w / precision) for w in range(precision)]

	# Delimit definition intervals for w
	for w0 in list_w0:
		label_w0 = ["w_0", "w'_0", "w''_0", "w'''_0", "w''''_0"]
		style=['--', '--', '--', '--', '--']
		if (w0 > 0) and (w0 < 1):
			ax1.axvline(x=w0, linestyle=style[list_w0.index(w0)], color='tab:blue', markersize=1, label=label_w0[list_w0.index(w0)])

	# Plot scalar product functions in interval
	solution_sq = [fct_sq_solution(w, n, norm, solution_init) for w in w_list]
	ax1.plot(w_list, solution_sq, '-', color='tab:blue', markersize=1, label='%s' % solution_idx[-1])

	for j in range(len(common4b_init)):
		common4b_sq = [fct_sq_common4(w, j, n, norm, common4b_init[j]) for w in w_list]
		ax1.plot(w_list, common4b_sq, '-', color='tab:purple', markersize=1, label='%s' % common4b_idx[j][-1])

	for j in range(len(common3b_init)):
		common3b_sq = [fct_sq_common3(w, j, n, norm, common3b_init[j]) for w in w_list]
		ax1.plot(w_list, common3b_sq, '-', color='tab:orange', markersize=1, label='%s' % common3b_idx[j][-1])

	for j in range(len(common2b_init) // 2):
		for l in range(2):
			common2b_sq = [fct_sq_common2(w, j, n, norm, common2b_init[2*j + l]) for w in w_list]
			ax1.plot(w_list, common2b_sq, '-', color='tab:green', markersize=1, label='%s' % common2b_idx[2*j + l][-1])

	for j in range(len(nonsol_init)):
		nonsol_sq = [fct_sq_nonsol(w, n, norm, nonsol_init[j]) for w in w_list]
		ax1.plot(w_list, nonsol_sq, '-', color='tab:red', markersize=1, label='%s' % nonsol_idx[j][-1])
	
	# Minimum of correct function
	w_min = find_sq_min(n, norm, solution_init)
	if (w_min != None) and (w_min > 0) and (w_min < 1):
		ax1.axvline(x=w_min, linestyle=':', color='tab:blue', markersize=1, label='w_min')

	# # Test for verification purposes
	# test_loop(mu, n, norm, kappa, K, R, S_ref, ax1)

	# ------------------------------------------------------------------------------------------- #

	# List of all intersections
	wr = []

	# Intersections between solution function and other scalar product functions
	for j in range(len(common4b_init)):
		wr1_common4b, wr2_common4b = find_wr_common4(j, n, norm, solution_init, common4b_init[j])
		
		append_wr(1, wr1_common4b, interval, wr)
		append_wr(2, wr2_common4b, interval, wr)
		# ax1.axvline(x=wr1_common4b, linestyle='--', color='tab:brown', markersize=1, label='wr1')
		# ax1.axvline(x=wr2_common4b, linestyle='--', color='tab:brown', markersize=1, label='wr2')

	for j in range(len(common3b_init)):
		wr1_common3b, wr2_common3b = find_wr_common3(j, n, norm, solution_init, common3b_init[j])
		
		append_wr(1, wr1_common3b, interval, wr)
		append_wr(2, wr2_common3b, interval, wr)
		# ax1.axvline(x=wr1_common3b, linestyle='--', color='tab:brown', markersize=1, label='wr1')
		# ax1.axvline(x=wr2_common3b, linestyle='--', color='tab:brown', markersize=1, label='wr2')

	for j in range(len(common2b_init) // 2):
		for l in range(2):
			wr1_common2b, wr2_common2b = find_wr_common2(j, n, norm, solution_init, common2b_init[2*j + l])

			append_wr(1, wr1_common2b, interval, wr)
			append_wr(2, wr2_common2b, interval, wr)
			# ax1.axvline(x=wr1_common2b, linestyle='--', color='tab:brown', markersize=1, label='wr1')
			# ax1.axvline(x=wr2_common2b, linestyle='--', color='tab:brown', markersize=1, label='wr2')

	for j in range(len(nonsol_init)):
		wr1_nonsol, wr2_nonsol = find_wr_nonsol(norm, solution_init, nonsol_init[j])
		
		append_wr(1, wr1_nonsol, interval, wr)
		append_wr(2, wr2_nonsol, interval, wr)
		# ax1.axvline(x=wr1_nonsol, linestyle='--', color='tab:brown', markersize=1, label='wr1')
		# ax1.axvline(x=wr2_nonsol, linestyle='--', color='tab:brown', markersize=1, label='wr2')

	# ------------------------------------------------------------------------------------------- #

	# Find rank of solution kappa
	rank, wr = compute_rank_wr(wr)

	# ax1.legend(loc='upper left', ncol=8, fancybox=True)
	ax1.set_title('Combined DOM for K=%s and kappa=%s' %(K, kappa))
	ax1.set_ylabel('Squared scalar product <S_ref,P>^2')
	
	ax2.plot(wr, rank, color='tab:blue')
	ax2.set_ylabel('Rank kappa solution')
	ax2.set_xlabel('Weight w')
	ax2.set_ylim([32, 0])

	fig.tight_layout()
	plt.show()
	# plt.savefig('./plot/sim_1x5bits_sq_K=%s_kappa=%s_#%s.pdf' % (K, kappa, num_sim), dpi=300)
	# plt.savefig('./plot/sim_1x5bits_sq_K=%s_kappa=%s_#%s.png' % (K, kappa, num_sim))
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

