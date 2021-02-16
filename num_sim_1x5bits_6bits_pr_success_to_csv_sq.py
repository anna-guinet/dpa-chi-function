#!/usr/bin/env python

""" DPA ON KECCAK 5-BIT CHI ROW & 1-BIT K """

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__     = "1.1"

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
	kappa -- string
	K     -- string
	i 	  -- integer in [0,n-1]
	
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
	- key_(i mod n)   = kappa_i XOR K = 0
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

def gen_S(mu, n, kappa, K):
	"""
	Generate signal power consumption for every bit i of first chi's row, 
	for a couple (kappa, K). 
	
	Parameters:
	mu    -- 2D list of bool, size n x 2^n
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
	b 	  		-- integer
	n 	  		-- integer
	K 	  		-- string
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

def find_w0(solution_init, norm):
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
			w0 = a / (a - Decimal(norm))

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
	i     		   -- integer
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
	
	Parameters:
	i    		  -- integer
	n    		  -- integer
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
	i    		  -- integer
	n    		  -- integer
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
	i    -- integer
	n    -- integer
	norm -- Decimal
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
	j        -- integer
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

def compute_rank_wr(wr):
	"""
	Compute intervals for each ranks in order to plot them.

	Parameter:
	wr -- list of Decimal

	Return:
	rank_wr -- list of tuples
	"""
	# Transform Decimal into float
	getcontext().prec = 6
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
	df 		-- DataFrame
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
	w 		   			-- Decimal
	count_diff 			-- integer
	rank1_df   			-- Dataframe
	mean_interval_rank1 -- list

	Return:
	None
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

	Return:
	None
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
	Create figure to plot

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
	plt.title(r'cDoM with squared scalar product | kappa=%s & K=%s' %(kappa, K))
	plt.xlabel('w')
	plt.ylabel('Probability of success')
	plt.show()

def write_csv(K, kappa, num_sim, frames):
	"""
	Write Dataframes in CSV files.

	Parameters:
	K 		-- string
	kappa   -- string
	num_sim -- integer
	frames  -- list of DataFrames

	Return:
	None
	"""
	num_bitK = 0
	for frame in frames:
		file_name = r'./csv/1x5bits_6bits_sq_num-sim=%s_K=%s_kappa=%s_%sK.csv' % (num_sim, K, kappa, num_bitK)
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
	mu   = list(itertools.product([bool(0), bool(1)], repeat=n))
	norm = Decimal(2**n)

	# Signal reference vectors
	S_ref = gen_S_ref(mu, n)
	
	# Save the results of simulations
	rank_wr_list = []

	# Save the results of the simulations
	succ_df_5K, succ_df_4K, succ_df_3K = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
	succ_df_2K, succ_df_1K, succ_df_0K = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

	# ------------------------------------------------------------------------------------------- #
	
	for j in range(num_sim):
	
		# Noise power consumption for m bits
		R = noise(mu)

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

		# ------------------------------------------------------------------------------------------- #
		
		# List of all intersections
		wr = []

		interval = [0, 1]

		# Intersections between solution function and other scalar product functions
		for j in range(len(common4b_init)):
			wr1_common4b, wr2_common4b = find_wr_common4(j, n, norm, solution_init, common4b_init[j])
			append_wr(1, wr1_common4b, interval, wr)
			append_wr(2, wr2_common4b, interval, wr)

		for j in range(len(common3b_init)):
			wr1_common3b, wr2_common3b = find_wr_common3(j, n, norm, solution_init, common3b_init[j])
			append_wr(1, wr1_common3b, interval, wr)
			append_wr(2, wr2_common3b, interval, wr)

		for j in range(len(common2b_init) // 2):
			for l in range(2):
				wr1_common2b, wr2_common2b = find_wr_common2(j, n, norm, solution_init, common2b_init[2*j + l])
				append_wr(1, wr1_common2b, interval, wr)
				append_wr(2, wr2_common2b, interval, wr)

		for j in range(len(nonsol_init)):
			wr1_nonsol, wr2_nonsol = find_wr_nonsol(norm, solution_init, nonsol_init[j])		
			append_wr(1, wr1_nonsol, interval, wr)
			append_wr(2, wr2_nonsol, interval, wr)

		# ------------------------------------------------------------------------------------------- #

		# Determine the intervals for the ranks per noise vector R
		rank_wr    = compute_rank_wr(wr)
		rank_wr_df = pd.DataFrame(rank_wr, columns=['rank', 'w'])

		# Select 1st rank only and creat succ column in new Dataframe
		rank_wr_df1 = rank_wr_df.loc[rank_wr_df['rank'] == 1]

		rank1_df = rank_wr_df1.assign(succ=bool(0))
		rank1_df.drop(['rank'], axis=1, inplace=True)
		rank1_df.reset_index(drop=True, inplace=True)

		# Create w intervals for 1st rank
		interval_rank1 = [w for w in zip(rank1_df['w'][::2], rank1_df['w'][1::2])]

		# Find w0 for each bit of the register, tup = (w0, a)
		list_tup_w0 = find_w0(solution_init, len(mu))
		list_tup_w0 = sorted(list_tup_w0, key=lambda x: x[0])
		list_w0 = [tup[0] for tup in list_tup_w0]
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
