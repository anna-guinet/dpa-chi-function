#!/usr/bin/env python

"""SUCCESS PROBA DPA 1-BIT CHI ROW & 1-BIT K """

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

def gen_key_i(i, kappa='000', K='0'):
	""" 
	Create key value where key = kappa, except for bit i: key_i = kappa_i XOR K
	
	Parameters:
	kappa -- string (default '000')
	K -- string (default '0')
	i -- integer in [0,n-1]
	
	Return:
	key_i -- list of bool
	"""

	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = bool(int(K))
	
	# Initialize new key value
	key_i = kappa.copy()
	
	# XOR at indice i
	key_i[i] = K ^ kappa[i]
	
	return key_i

def gen_key(i):
	""" 
	Create key value where key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	Eg: for i = 1, key = [[ 0, 0, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ], [ 1, 0, 1 ]]
	
	Note: We do not take into account key_i as it adds only symmetry the signal power consumption.
	
	Parameters:
	i -- integer in [0,n-1]
	
	Return:
	key -- 2D list of bool, size n x 2^(n-1)
	"""
	key_0 = [[bool(0), bool(0), bool(0)], 
			 [bool(0), bool(0), bool(1)], 
			 [bool(0), bool(1), bool(0)], 
			 [bool(0), bool(1), bool(1)]]

	key_1 = [[bool(0), bool(0), bool(0)], 
			 [bool(0), bool(0), bool(1)], 
			 [bool(1), bool(0), bool(0)], 
			 [bool(1), bool(0), bool(1)]]

	key_2 = [[bool(0), bool(0), bool(0)], 
			 [bool(0), bool(1), bool(0)], 
			 [bool(1), bool(0), bool(0)], 
			 [bool(1), bool(1), bool(0)]]

	key = [key_0, key_1, key_2]
	
	return key[i]
	
def signal_2D(mu, i, key, n=3):
	"""
	Generate signal power consumption S values for a 3-bit kappa and a 1-bit K, for all messages mu.
	
	'key' is a value such as key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	
	Parameters:
	mu -- 2D list of bool, size n x 2^n
	i -- integer in [0,n-1]
	key -- list of bool
	n -- integer (default 3)
	
	Return:
	S -- 2D list of integers, size 2^n x 2^(n-1)
	"""	
	S = []
	
	# For each key value
	#for l in range(2**n):
	for l in range(2**(n-1)):
	
		S_l = []

		# For each message mu value
		for j in range(2**n):
			
			# Activity of the first storage cell of the register
			d = key[l][i] ^ mu[j][i] ^ ((key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n]))
				
			S_lj = (-1)**(d)
				
			S_l.append(S_lj)
		
		S += [S_l]
	
	return S

def noise(n=3):	
	"""
	Compute a noise power consumption R vectors for m bits, for each possible message mu. 

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
	Computes probability of success to have a given rank.
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

def find_abs_w0(a):
	"""
	Find the point when for the scalar product of the solution equals 0.
	
	Parameter:
	a -- Decimal
	
	Return:
	w0 -- Decimal
	"""	
	w0 = 0

	if (a - Decimal(8)) != 0:
		# w0 = a / a - 8
		w0 = a / (a - Decimal(8))
		
	return w0

def find_wr(a, b):
	"""
	Find the point when for the scalar product of the solutionequals the one of a nonsolution.
	
	Parameter:
	a -- Decimal
	b -- Decimal
	
	Return:
	y -- Decimal
	"""	
	b = abs(b)
	w1 = 0
	w2 = 0

	if (Decimal(8) + (b - a)) != 0:
		# w1 = (|b| - a)/ (8 - (|b| - a))
		w1 = (b - a) / (Decimal(8) + (b - a))
	
	if (b + a - Decimal(8)) != 0:

		# w2 = (|b| + a)/ (|b| + a - 8)
		w2 = (b + a) / (b + a - Decimal(8))

	return w1, w2

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

def find_rank(wr, w0):
	"""
	Return the list of ranks for the solution kappa.
	
	Parameter:
	wr -- list of Decimal
	w0 -- Decimal
	
	Return:
	rank -- list of integer
	wr -- list of Decimal
	"""

	# List of ranks
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if wr:

		# Count number of rank increment ('wr2') and rank decrement (wr1')
		counter_1 = sum(1 for w in wr if w > w0)
		counter_2 = sum(-1 for w in wr if w < w0)
		num_wr = counter_1 + counter_2
		
		rank_init = 1 + num_wr
		
		rank.append(rank_init)
		
		for w in wr:
		
			# Rank increases
			if w > w0:
				rank.append(rank[-1] - 1)
				
			# Rank decreases
			if w < w0:
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, wr

def sim_1bit(n, i, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n -- integer
	i -- integer
	K -- string
	kappa -- string
	num_sim -- integer

	Return:
	NaN
	"""
	
	# Signal message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	getcontext().prec = 10

	# Secret values for i-th bit
	key_i = gen_key_i(i, kappa, K)
	key = gen_key(i)

	# All possible signal power consumption values
	S_ref = signal_2D(mu, i, key, n)

	# Save the results of the experiments
	rank_wr_list = []

	""" Generate many simulations """
	for j in range(num_sim):
	
		# Noise power consumption
		R = noise(n)

		""" Initial values of scalar product """

		scalar_init = []

		# Global power consumption P_init = R
		P_init = [Decimal(r) for r in R]

		# Scalar product <S-ref, P>
		scalar_init = np.dot(S_ref, P_init)

		# Parameters for functions
		list_kappa_idx0 = [['.00', 0, False, False],
						   ['.01', 1, False, True],
						   ['.10', 2, True, False],
						   ['.11', 3, True, True]]

		list_kappa_idx1 = [['0.0', 0, False, False],
						   ['0.1', 1, True, False],
						   ['1.0', 2, False, True],
						   ['1.1', 3, True, True]]

		list_kappa_idx2 = [['00.', 0, False, False],
						  ['01.', 1, False, True],
						  ['10.', 2, True, False],
						  ['11.', 3, True, True]]

		list_kappa_idx = [list_kappa_idx0, list_kappa_idx1, list_kappa_idx2]

		solution_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[2] == key_i[(i+1) % n]) and (sublist[3] == key_i[(i+2) % n])]
		solution_idx = list(itertools.chain.from_iterable(solution_idx))

		nonsol_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[0] != solution_idx[0])]
		nonsol0_idx = nonsol_idx[0]
		nonsol1_idx = nonsol_idx[1]
		nonsol2_idx = nonsol_idx[2]

		# Find initial scalar product values
		solution_init = scalar_init[solution_idx[1]]

		nonsol0_init = scalar_init[nonsol0_idx[1]]
		nonsol1_init = scalar_init[nonsol1_idx[1]]
		nonsol2_init = scalar_init[nonsol2_idx[1]]

		# Determine the sign of initial solution value depending on the activity of the register at bit i
		p_i = xor_secret_i(K, kappa, i)
		solution_init = solution_init * p_i

		""" Find the intersections with solution function """
		
		# List of all intersections with the solution kappa function
		wr = []

		# Find the fiber such as f(w0) = 0 
		w0 = find_abs_w0(solution_init)

		# Intersections between solution function and nonsol functions
		wr1_nonsol0, wr2_nonsol0 = find_wr(solution_init, nonsol0_init)
		wr1_nonsol1, wr2_nonsol1 = find_wr(solution_init, nonsol1_init)
		wr1_nonsol2, wr2_nonsol2 = find_wr(solution_init, nonsol2_init)
			
		if (wr1_nonsol0 > 0) and (wr1_nonsol0 < 1):
			wr.append(wr1_nonsol0)
			
		if (wr2_nonsol0 > 0) and (wr2_nonsol0 < 1):
			wr.append(wr2_nonsol0)
			
		if (wr1_nonsol1 > 0) and (wr1_nonsol1 < 1):
			wr.append(wr1_nonsol1)
				
		if (wr2_nonsol1 > 0) and (wr2_nonsol1 < 1):
			wr.append(wr2_nonsol1)
			
		if (wr1_nonsol2 > 0) and (wr1_nonsol2 < 1):
			wr.append(wr1_nonsol2)
				
		if (wr2_nonsol2 > 0) and (wr2_nonsol2 < 1):
			wr.append(wr2_nonsol2)

		""" Find rank of solution kappa	"""

		# Transform Decimal into float
		getcontext().prec = 5
		wr = [float(w) for w in wr]

		# Sort by w
		wr = sorted(wr)

		rank, wr = find_rank(wr, w0)

		# Expand the lists for plotting
		rank = [r for r in zip(rank, rank)]
		wr = [i for i in zip(wr, wr)]
			
		# Un-nest the previous lists
		rank = list(itertools.chain.from_iterable(rank))
		wr = list(itertools.chain.from_iterable(wr))
			
		# Insert edges for plotting
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

def sim_1bit_to_csv(n, i, K, kappa, num_sim):
	"""
	Plot probabilities of success for all possibilities for several simulations. 
	
	Parameters:
	n -- integer
	i -- integer
	K -- string
	kappa - string
	num_sim -- integer
	
	Return:
	NaN
	"""
	
	rank_wr_list = sim_1bit(n, i, K, kappa, num_sim)
	
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

	# Sort by absolute value, descending order
	rank_wr_df1 = rank_wr_df1.iloc[(-rank_wr_df1['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df2 = rank_wr_df2.iloc[(-rank_wr_df2['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df3 = rank_wr_df3.iloc[(-rank_wr_df3['wr'].abs()).argsort()].reset_index(drop=True)
	rank_wr_df4 = rank_wr_df4.iloc[(-rank_wr_df4['wr'].abs()).argsort()].reset_index(drop=True)

	# Probability of a simulation
	pr_sim = 1 / num_sim
	
	# Compute probabilities of success
	rank_wr_df1 = pr_success(rank_wr_df1, 1, pr_sim)
	rank_wr_df2 = pr_success(rank_wr_df2, 0, pr_sim)
	rank_wr_df3 = pr_success(rank_wr_df3, 0, pr_sim)
	rank_wr_df4 = pr_success(rank_wr_df4, 0, pr_sim)

	""" Write in CSV files """

	num_rank = 1
	for df in [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4]:
		file_name = r'./csv/1bit_abs_num-sim=%s_K=%s_kappa=%s_i=%s_rank%s.csv' % (num_sim, K, kappa, i, num_rank)
		df.to_csv(file_name, encoding='utf-8', index=False)
		num_rank += 1

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='i, K, kappa and num_sim')

	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, 2]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3 or (args.i < 0) or (args.i > 2):
		print('\n**ERROR**')
		print('Required length of K and kappa: 3')
		print('i is in [0, 2]\n')

	else:
		# Length of signal part
		n = 3

		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		# i-th bit of register state studied
		i = args.i

		# Time option
		start_t = time.perf_counter()

		sim_1bit_to_csv(n, i, K, kappa, num_sim)
		
		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))
