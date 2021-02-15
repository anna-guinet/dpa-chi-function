#!/usr/bin/env python

"""DPA SIMULATION 3-BIT CHI ROW & 3-BIT K"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__     = "3.1"

import numpy as np
import itertools
from decimal import *
import math
import matplotlib.pylab as plt
import random
import argparse
import time
import sys

def noise(mu):
	"""
	Compute a noise power consumption R vector for each possible message mu. 

	Parameter:
	mu -- list of Bool

	Return:
	R -- list of integers, length 2^n
	"""
	R = []

	# For each message mu
	for j in range(len(mu)):
		
		# Activity of noise part
		d = random.gauss(0, 1)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def signal_1D(mu, K, kappa, n):
	"""
	Generate signal power consumption S values for a secret state key, for all messages mu. 
	
	Parameters:
	mu 	  -- 2D list of bool, size n x 2^n
	K 	  -- string
	kappa -- string
	n 	  -- integer
	
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

def signal_2D(mu, K, n):
	"""
	Generate signal power consumption S reference values for all messages mu.

	Parameters:
	mu -- 2D list of bool, size n x 2^n
	K  -- string
	n  -- integer (default 3)
	
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

def signal_3D(mu, n):
	"""
	Generate signal power consumption S reference values for all messages mu.

	Parameters:
	mu -- 2D list of bool, size n x 2^n
	n  -- integer
	
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
	value  -- float

	Return
	indexes -- list of integer
	"""
	# Give sublist index and element index in scalar if element == value
	indexes = [[sublist_idx, elem_idx] for sublist_idx,sublist in enumerate(scalar) for elem_idx, elem in enumerate(sublist) if elem==value]

	return indexes

def kappa_K_idx():
	"""
	Provide scalar products indexes for K and kappa values.

	Return:
	list_K_kappa_idx -- list of lists
	"""
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

	list_K_kappa = [list_K_kappa_idx0, list_K_kappa_idx1, list_K_kappa_idx2, list_K_kappa_idx3, 
					list_K_kappa_idx4, list_K_kappa_idx5, list_K_kappa_idx6, list_K_kappa_idx7]

	return list_K_kappa

def find_idx(list_K_kappa_idx, list_scalar):
	"""
	Link scalar products with indexes for K and kappa values.

	Parameters:
	list_K_kappa_idx -- list of lists
	list_scalar 	 -- list of integer

	Return:
	list_scalar -- list of lists
	"""
	list_idx = [sublist_K_kappa for sublist_K in list_K_kappa_idx for sublist_K_kappa in sublist_K for sublist_sc in list_scalar if (sublist_K_kappa[2] == sublist_sc[0] and (sublist_K_kappa[3] == sublist_sc[1]))]
	
	return list_idx

def find_init(num_deg, num_ndeg, scalar_init, list_idx):
	"""
	Find the scalar products for function. 

	Parameters:
	num_deg     -- integer
	num_ndeg    -- integer
	scalar_init -- list of Decimal
	list_idx 	-- list of list

	Return:
	list_init -- list of Decimal
	"""	
	list_init = []

	if len(list_idx) == num_deg: # Degenerated case
		for j in range(num_deg):
			list_init += [scalar_init[list_idx[j][2]][list_idx[j][3]]]

	else: # Non-degenerated case
		for j in range(num_ndeg):
			list_init += [scalar_init[list_idx[j][2]][list_idx[j][3]]]

	return list_init

def fct_scalar(w, a, value):
	"""
	Function for the scalar product of any possibility for a given value of correlation.
	
	Parameter:
	w 	  -- float
	a 	  -- Decimal
	value -- Decimal
	
	Return:
	y -- Decimal
	"""
	# y = (value - a) * w + a
	y = (Decimal(value) - a) * Decimal(w) + a
		
	return y / Decimal(24)

def plot_fct(value, num_deg, num_ndeg, interval, ax1, color, list_idx, list_init):
	"""
	Plot the scalar product function for common. 

	Parameters:
	value 	  -- integer
	num_deg   -- integer
	num_ndeg  -- integer
	interval  -- list of integer
	ax1 	  -- subplot
	color 	  -- string
	list_idx  -- list of list
	list_init -- list of Decimal

	Return: None
	"""
	if len(list_idx) == num_deg: # Degenerated case
		for j in range(num_deg):
			fct = [fct_scalar(w, list_init[j], value) for w in interval]
			ax1.plot(interval, fct, '-', color='tab:%s' % (color), markersize=1, label='%s %s' % (list_idx[j][0], list_idx[j][1]))
	else:
		for j in range(num_ndeg):
			fct = [fct_scalar(w, list_init[j], value) for w in interval]
			ax1.plot(interval, fct, '-', color='tab:%s' % (color), markersize=1, label='%s %s' % (list_idx[j][0], list_idx[j][1]))

def find_wr(a, b, value_b):
	"""
	Find the point when for the scalar product of the solution guess equals the scalar product of an nonsol guess.
	
	Parameter:
	a 		-- Decimal
	b 		-- Decimal
	value_b -- float
	
	Return:
	y -- Decimal
	"""	
	value = Decimal(24) - Decimal(value_b)

	# w = (b - a)/ (value + b - a)
	w = (b - a) / (value + b - a)

	return w

def find_wr_common(value, num_deg, num_ndeg, solution_init, list_idx, list_init):
	"""
	Find intersection wr with solution function. 

	Parameters:
	value 		  -- integer
	num_deg 	  -- integer
	num_ndeg 	  -- integer
	solution_init -- list of Decimal
	list_idx 	  -- list of list
	list_init 	  -- list of Decimal

	Return:
	list_wr -- list of Decimal
	"""
	list_wr = []

	if len(list_idx) == num_deg: # Degenerated case
		for j in range(num_deg):
			wr = find_wr(solution_init[0], list_init[j], value)
			list_wr.append(wr)
	else:
		for j in range(num_ndeg):
			wr = find_wr(solution_init[0], list_init[j], value)
			list_wr.append(wr)
		
	# Drop values < 0 and > 1
	list_wr = [w for w in list_wr if (w > 0) and (w < 1)]

	return list_wr

def test_loop(n, mu, K, kappa, ax1, R, S_ref):
	"""
	Control test.
	"""

	w_list = []

	# Signal power consumption for a key
	S = signal_1D(mu, K, kappa, n)

	# Parameters for loop
	scalar_K_000_kappa_000, scalar_K_000_kappa_001 = [], []
	scalar_K_000_kappa_010, scalar_K_000_kappa_011 = [], []
	scalar_K_000_kappa_100, scalar_K_000_kappa_101 = [], []
	scalar_K_000_kappa_110, scalar_K_000_kappa_111 = [], []

	scalar_K_001_kappa_000, scalar_K_001_kappa_001 = [], []
	scalar_K_001_kappa_010, scalar_K_001_kappa_011 = [], []
	scalar_K_001_kappa_100, scalar_K_001_kappa_101 = [], []
	scalar_K_001_kappa_110, scalar_K_001_kappa_111 = [], []

	scalar_K_010_kappa_000, scalar_K_010_kappa_001 = [], []
	scalar_K_010_kappa_010, scalar_K_010_kappa_011 = [], []
	scalar_K_010_kappa_100, scalar_K_010_kappa_101 = [], []
	scalar_K_010_kappa_110, scalar_K_010_kappa_111 = [], []

	scalar_K_011_kappa_000, scalar_K_011_kappa_001 = [], []
	scalar_K_011_kappa_010, scalar_K_011_kappa_011 = [], []
	scalar_K_011_kappa_100, scalar_K_011_kappa_101 = [], []
	scalar_K_011_kappa_110, scalar_K_011_kappa_111 = [], []

	scalar_K_100_kappa_000, scalar_K_100_kappa_001 = [], []
	scalar_K_100_kappa_010, scalar_K_100_kappa_011 = [], []
	scalar_K_100_kappa_100, scalar_K_100_kappa_101 = [], []
	scalar_K_100_kappa_110, scalar_K_100_kappa_111 = [], []

	scalar_K_101_kappa_000, scalar_K_101_kappa_001 = [], []
	scalar_K_101_kappa_010, scalar_K_101_kappa_011 = [], []
	scalar_K_101_kappa_100, scalar_K_101_kappa_101 = [], []
	scalar_K_101_kappa_110, scalar_K_101_kappa_111 = [], []

	scalar_K_110_kappa_000, scalar_K_110_kappa_001 = [], []
	scalar_K_110_kappa_010, scalar_K_110_kappa_011 = [], []
	scalar_K_110_kappa_100, scalar_K_110_kappa_101 = [], []
	scalar_K_110_kappa_110, scalar_K_110_kappa_111 = [], []

	scalar_K_111_kappa_000, scalar_K_111_kappa_001 = [], []
	scalar_K_111_kappa_010, scalar_K_111_kappa_011 = [], []
	scalar_K_111_kappa_100, scalar_K_111_kappa_101 = [], []
	scalar_K_111_kappa_110, scalar_K_111_kappa_111 = [], []

	for j in range(1001):

		# Weight of signal part
		w = j / 1000
		
		scalar = []
			
		# Global power consumption P = (1 - w) * R + w * S
		getcontext().prec = 8
		P = [(Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s)) for r, s in zip(R, S)]

		# Scalar product <S-ref, P>
		scalar = np.dot(S_ref, P)

		w_list.append(w)

		scalar_K_000_kappa_000.append(scalar[0][0] / Decimal(24))
		scalar_K_000_kappa_001.append(scalar[0][1] / Decimal(24))
		scalar_K_000_kappa_010.append(scalar[0][2] / Decimal(24))
		scalar_K_000_kappa_011.append(scalar[0][3] / Decimal(24))
		scalar_K_000_kappa_100.append(scalar[0][4] / Decimal(24))
		scalar_K_000_kappa_101.append(scalar[0][5] / Decimal(24))
		scalar_K_000_kappa_110.append(scalar[0][6] / Decimal(24))
		scalar_K_000_kappa_111.append(scalar[0][7] / Decimal(24))

		scalar_K_001_kappa_000.append(scalar[1][0] / Decimal(24))
		scalar_K_001_kappa_001.append(scalar[1][1] / Decimal(24))
		scalar_K_001_kappa_010.append(scalar[1][2] / Decimal(24))
		scalar_K_001_kappa_011.append(scalar[1][3] / Decimal(24))
		scalar_K_001_kappa_100.append(scalar[1][4] / Decimal(24))
		scalar_K_001_kappa_101.append(scalar[1][5] / Decimal(24))
		scalar_K_001_kappa_110.append(scalar[1][6] / Decimal(24))
		scalar_K_001_kappa_111.append(scalar[1][7] / Decimal(24))

		scalar_K_010_kappa_000.append(scalar[2][0] / Decimal(24))
		scalar_K_010_kappa_001.append(scalar[2][1] / Decimal(24))
		scalar_K_010_kappa_010.append(scalar[2][2] / Decimal(24))
		scalar_K_010_kappa_011.append(scalar[2][3] / Decimal(24))
		scalar_K_010_kappa_100.append(scalar[2][4] / Decimal(24)) 
		scalar_K_010_kappa_101.append(scalar[2][5] / Decimal(24))
		scalar_K_010_kappa_110.append(scalar[2][6] / Decimal(24))
		scalar_K_010_kappa_111.append(scalar[2][7] / Decimal(24))

		scalar_K_011_kappa_000.append(scalar[3][0] / Decimal(24))
		scalar_K_011_kappa_001.append(scalar[3][1] / Decimal(24))
		scalar_K_011_kappa_010.append(scalar[3][2] / Decimal(24))
		scalar_K_011_kappa_011.append(scalar[3][3] / Decimal(24))
		scalar_K_011_kappa_100.append(scalar[3][4] / Decimal(24))
		scalar_K_011_kappa_101.append(scalar[3][5] / Decimal(24))
		scalar_K_011_kappa_110.append(scalar[3][6] / Decimal(24))
		scalar_K_011_kappa_111.append(scalar[3][7] / Decimal(24))

		scalar_K_100_kappa_000.append(scalar[4][0] / Decimal(24))
		scalar_K_100_kappa_001.append(scalar[4][1] / Decimal(24))
		scalar_K_100_kappa_010.append(scalar[4][2] / Decimal(24))
		scalar_K_100_kappa_011.append(scalar[4][3] / Decimal(24))
		scalar_K_100_kappa_100.append(scalar[4][4] / Decimal(24))
		scalar_K_100_kappa_101.append(scalar[4][5] / Decimal(24))
		scalar_K_100_kappa_110.append(scalar[0][6] / Decimal(24))
		scalar_K_100_kappa_111.append(scalar[4][7] / Decimal(24))

		scalar_K_101_kappa_000.append(scalar[5][0] / Decimal(24))
		scalar_K_101_kappa_001.append(scalar[5][1] / Decimal(24))
		scalar_K_101_kappa_010.append(scalar[5][2] / Decimal(24))
		scalar_K_101_kappa_011.append(scalar[5][3] / Decimal(24))
		scalar_K_101_kappa_100.append(scalar[5][4] / Decimal(24))
		scalar_K_101_kappa_101.append(scalar[5][5] / Decimal(24))
		scalar_K_101_kappa_110.append(scalar[5][6] / Decimal(24))
		scalar_K_101_kappa_111.append(scalar[5][7] / Decimal(24))

		scalar_K_110_kappa_000.append(scalar[6][0] / Decimal(24))
		scalar_K_110_kappa_001.append(scalar[6][1] / Decimal(24))
		scalar_K_110_kappa_010.append(scalar[6][2] / Decimal(24))
		scalar_K_110_kappa_011.append(scalar[6][3] / Decimal(24))
		scalar_K_110_kappa_100.append(scalar[6][4] / Decimal(24))
		scalar_K_110_kappa_101.append(scalar[6][5] / Decimal(24))
		scalar_K_110_kappa_110.append(scalar[6][6] / Decimal(24))
		scalar_K_110_kappa_111.append(scalar[6][7] / Decimal(24))

		scalar_K_111_kappa_000.append(scalar[7][0] / Decimal(24))
		scalar_K_111_kappa_001.append(scalar[7][1] / Decimal(24))
		scalar_K_111_kappa_010.append(scalar[7][2] / Decimal(24))
		scalar_K_111_kappa_011.append(scalar[7][3] / Decimal(24))
		scalar_K_111_kappa_100.append(scalar[7][4] / Decimal(24))
		scalar_K_111_kappa_101.append(scalar[7][5] / Decimal(24))
		scalar_K_111_kappa_110.append(scalar[7][6] / Decimal(24))
		scalar_K_111_kappa_111.append(scalar[7][7] / Decimal(24))
	
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

def find_rank(wr):
	"""
	Return the list of ranks for the solution kappa.
	
	Parameter:
	wr -- list of float
	
	Return:
	rank -- list of integer
	wr   -- list of float
	"""
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if wr:
		# Count number of rank increment
		rank = [1 + count for count in range(len(wr), 0, -1)]

	rank += [1]

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
	getcontext().prec = 8
	wr = [float(w) for w in wr]
	wr = sorted(wr)

	rank, wr = find_rank(wr)

	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr   = [i for i in zip(wr, wr)]

	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr   = list(itertools.chain.from_iterable(wr))

	# Insert edges for plotting
	wr.insert(0, 0)
	wr.append(1)

	return rank, wr

def sim_3bits(n, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer

	Return:
	NaN
	"""
	# Message
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	getcontext().prec = 10

	# Noise power consumption
	R = noise(mu)

	# All possible signal power consumption values
	S_ref = signal_3D(mu, n)

	# ------------------------------------------------------------------------------------------- #

	# Scalar product for full signal <S-ref, P = S>
	S = signal_1D(mu, K, kappa, n)
	P_fin = S
	scalar_fin = np.dot(S_ref, P_fin)

	# Find the indexes of the elements equals to a specific scalar product value
	scalar_fin_pos_24 = find_idx_scalar(scalar_fin, 24)
	scalar_fin_pos_16 = find_idx_scalar(scalar_fin, 16)
	scalar_fin_pos_8  = find_idx_scalar(scalar_fin, 8)
	scalar_fin_0 	  = find_idx_scalar(scalar_fin, 0)
	scalar_fin_neg_8  = find_idx_scalar(scalar_fin, -8)
	scalar_fin_neg_16 = find_idx_scalar(scalar_fin, -16)
	scalar_fin_neg_24 = find_idx_scalar(scalar_fin, -24)

	# Match the vectors with the secret values
	list_K_kappa_idx = kappa_K_idx()

	solution_idx 	  = find_idx(list_K_kappa_idx, scalar_fin_pos_24)
	common_pos_16_idx = find_idx(list_K_kappa_idx, scalar_fin_pos_16)
	common_pos_8_idx  = find_idx(list_K_kappa_idx, scalar_fin_pos_8)
	uncommon_0_idx 	  = find_idx(list_K_kappa_idx, scalar_fin_0)
	common_neg_8_idx  = find_idx(list_K_kappa_idx, scalar_fin_neg_8)
	common_neg_16_idx = find_idx(list_K_kappa_idx, scalar_fin_neg_16)
	common_neg_24_idx = find_idx(list_K_kappa_idx, scalar_fin_neg_24)

	# ------------------------------------------------------------------------------------------- #

	# Initial scalar product <S-ref, P = R>
	P_init = [Decimal(r) for r in R]
	scalar_init = np.dot(S_ref, P_init)

	# Find initial scalar products according to idx
	solution_init 	   = find_init(2, 1, scalar_init, solution_idx)
	common_pos_16_init = find_init(6, 4, scalar_init, common_pos_16_idx)
	common_pos_8_init  = find_init(18, 19, scalar_init, common_pos_8_idx)
	uncommon_0_init    = find_init(12, 16, scalar_init, uncommon_0_idx)
	common_neg_8_init  = find_init(18, 19, scalar_init, common_neg_8_idx)
	common_neg_16_init = find_init(6, 4, scalar_init, common_neg_16_idx)
	common_neg_24_init = find_init(2, 1, scalar_init, common_neg_24_idx)

	# ------------------------------------------------------------------------------------------- #

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,5),  gridspec_kw={"width_ratios": [2,1]})

	# Plot scalar product functions
	interval = [0, 1]

	plot_fct(24, 2, 1, interval, ax1, 'red', solution_idx, solution_init)
	plot_fct(16, 6, 4, interval, ax1, 'purple', common_pos_16_idx, common_pos_16_init)
	plot_fct(8, 18, 19, interval, ax1, 'orange', common_pos_8_idx, common_pos_8_init)
	plot_fct(0, 12, 16, interval, ax1, 'blue', uncommon_0_idx, uncommon_0_init)
	plot_fct(-8, 18, 19, interval, ax1, 'green', common_neg_8_idx, common_neg_8_init)
	plot_fct(-16, 6, 4, interval, ax1, 'brown', common_neg_16_idx, common_neg_16_init)
	plot_fct(-24, 2, 1, interval, ax1, 'pink', common_neg_24_idx, common_neg_24_init)

	# Find intersection values
	wr_common_pos_16 = find_wr_common(16, 6, 4, solution_init, common_pos_16_idx, common_pos_16_init)
	wr_common_pos_8  = find_wr_common(8, 18, 19, solution_init, common_pos_8_idx, common_pos_8_init)
	wr_uncommon_0    = find_wr_common(0, 12, 16, solution_init, uncommon_0_idx, uncommon_0_init)
	wr_common_neg_8  = find_wr_common(-8, 18, 19, solution_init, common_neg_8_idx, common_neg_8_init)
	wr_common_neg_16 = find_wr_common(-16, 6, 4, solution_init, common_neg_16_idx, common_neg_16_init)
	wr_common_neg_24 = find_wr_common(-24, 1, 2, solution_init, common_neg_24_idx, common_neg_24_init)
	
	"""
	for w in wr_common_pos_16:
		ax1.axvline(x=w, linestyle='--', color='tab:purple', markersize=1, label='wr_common_pos_16')
	for w in wr_common_pos_8:
		ax1.axvline(x=w, linestyle='--', color='tab:orange', markersize=1, label='wr_common_pos_8')
	for w in wr_uncommon_0:
		ax1.axvline(x=w, linestyle='--', color='tab:blue', markersize=1, label='wr_uncommon_0')
	for w in wr_common_neg_8:
		ax1.axvline(x=w, linestyle='--', color='tab:green', markersize=1, label='wr_common_neg_8')
	for w in wr_common_neg_16:
		ax1.axvline(x=w, linestyle='--', color='tab:brown', markersize=1, label='wr_common_neg_16')
	for w in wr_common_neg_24:
		ax1.axvline(x=w, linestyle='--', color='tab:pink', markersize=1, label='wr_common_neg_24')
	"""

	# ------------------------------------------------------------------------------------------- #

	wr = wr_common_pos_16 + wr_common_pos_8 + wr_uncommon_0 +  wr_common_neg_8 +  wr_common_neg_16 +  wr_common_neg_24

	rank, wr = compute_rank_wr(wr)

	test_loop(n, mu, K, kappa, ax1, R, S_ref)

	# ax1.legend(loc='upper right')
	ax1.set_title(r'CPA strategy for K=%s & kappa=%s' %(K, kappa))
	ax1.set_ylabel('Scalar product <S_ref,P>')
	ax1.set_xlabel('Weight w')

	ax2.plot(wr, rank, color='tab:red')
	ax2.set_ylabel('Rank (K, kappa) solution', rotation=-90, labelpad=12)
	ax2.set_xlabel('Weight w')
	ax2.set_ylim([65, -5])
	ax2.yaxis.set_label_position("right")
	ax2.yaxis.tick_right()
	
	fig.tight_layout()
	plt.show()
	# plt.savefig('./plot/sim_3bits_K=%s_kappa=%s_#%s.png' % (K, kappa, num_sim))
	# plt.close(fig)

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


