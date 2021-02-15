#!/usr/bin/env python

"""DPA ON KECCAK n-BIT CHI ROW & 1-BIT K

Save the figures for a simulation which display the outcome of a test
according to the SNR, in order to recover kappa bits. 

Within one simulation, we guess 2 bits of kappa, with the scalar product, 
and a fixed noise power consumption vector.  
"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__     = "3.1"

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
		if (len(mu) == 32):
			d = random.gauss(0, 2)
		elif (len(mu) == 8):
			d = random.gauss(0, 1)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def gen_key_i(i, kappa, K):
	""" 
	Create key value where key equals kappa, except:
	key_(i mod n) = kappa_(i mod n) XOR K_(i mod n)
	
	Parameters:
	i 	  -- integer in [0,n-1]
	kappa -- string
	K 	  -- string
	
	Return:
	key_i -- list of bool
	"""
	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = list(K)
	K = [bool(int(j)) for j in K]
	
	# Initialize new key value
	key_i = kappa.copy()
	
	# XOR at indice i
	key_i[i] = K[i] ^ kappa[i]
	
	return key_i

def gen_key(i, n):
	"""
	Generate all key values for key_(i+2 mod n) and key_(i+1 mod n), regardless the i-th bit:
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
	key -- 2D list, size n x 4
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
	
def signal_2D(mu, i, key, n):
	"""
	Generate signal power consumption S values for all messages mu.
	
	'key' is a value such as key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	
	Parameters:
	mu  -- 2D list of bool, size n x 2^n
	i 	-- integer in [0,n-1]
	key -- list of bool
	n 	-- integer
	
	Return:
	S -- 2D list of integers, size 2^n x 2^(n-1)
	"""	
	S = []
	
	# For each key value
	for l in range(len(key)):
	
		S_l = []

		# For each message mu value
		for j in range(len(mu)):
			
			# Activity of the first storage cell of the register
			d = key[l][i] ^ mu[j][i] ^ ((key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n]))
				
			S_lj = (-1)**(d)
				
			S_l.append(S_lj)
		
		S += [S_l]
	
	return S

def kappa_idx(n):
	"""
	Provide scalar products indexes for kappa values.

	Parameter:
	n -- integer

	Return:
	list_kappa_idx -- list of lists
	"""
	list_kappa_idx = []

	if n == 5:
		list_kappa_idx0 = [['*00**', 0, False, False],
						   ['*01**', 1, False, True],
						   ['*10**', 2, True, False],
						   ['*11**', 3, True, True]]

		list_kappa_idx1 = [['**00*', 0, False, False],
						   ['**01*', 1, False, True],
						   ['**10*', 2, True, False],
						   ['**11*', 3, True, True]]

		list_kappa_idx2 = [['***00', 0, False, False],
						   ['***01', 1, False, True],
						   ['***10', 2, True, False],
						   ['***11', 3, True, True]]

		list_kappa_idx3 = [['0***0', 0, False, False],
						   ['0***1', 1, True, False],
						   ['1***0', 2, False, True],
						   ['1***1', 3, True, True]]

		list_kappa_idx4 = [['00***', 0, False, False],
						   ['01***', 1, False, True],
						   ['10***', 2, True, False],
						   ['11***', 3, True, True]]

		list_kappa_idx = [list_kappa_idx0, list_kappa_idx1, 
						  list_kappa_idx2, list_kappa_idx3, 
						  list_kappa_idx4]

	if n == 3:
		list_kappa_idx0 = [['*00', 0, False, False],
						   ['*01', 1, False, True],
						   ['*10', 2, True, False],
						   ['*11', 3, True, True]]

		list_kappa_idx1 = [['0*0', 0, False, False],
						   ['0*1', 1, True, False],
						   ['1*0', 2, False, True],
						   ['1*1', 3, True, True]]

		list_kappa_idx2 = [['00*', 0, False, False],
						  ['01*', 1, False, True],
						  ['10*', 2, True, False],
						  ['11*', 3, True, True]]

		list_kappa_idx = [list_kappa_idx0, list_kappa_idx1, list_kappa_idx2]

	return list_kappa_idx

def find_abs_w0(a, norm):
	"""
	Find the point when for the scalar product of the solution guess equals 0.
	
	Parameters:
	a 	 -- Decimal
	norm -- integer
	
	Return:
	w0 -- Decimal
	"""	
	w0 = 0

	if (a - Decimal(norm)) != 0:
		# w0 = a / a - norm
		w0 = a / (a - Decimal(norm))

	return w0

def find_wr(a, b, norm):
	"""
	Find the point when for the scalar product of the solution
	equals the scalar product of a nonsol.

	Parameters:
	a 	 -- Decimal
	b 	 -- Decimal
	norm -- integer

	Return:
	y -- Decimal
	"""	
	b = abs(b)

	w1, w2 = 0, 0

	if (Decimal(norm) + (b - a)) != 0:
		# w1 = (|b| - a)/ (norm - (|b| - a))
		w1 = (b - a) / (Decimal(norm) + (b - a))
	
	if (b + a - Decimal(norm)) != 0:
		# w2 = (|b| + a)/ (|b| + a - norm)
		w2 = (b + a) / (b + a - Decimal(norm))

	return w1, w2

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

def append_wr(value, wr):
	"""
	Append value to wr if in [0,1]. 

	Parameters:
	value 	 -- Decimal
	wr 		 -- list of Decimal

	Return: None
	"""
	if (value != None) and (value > 0) and (value < 1):
		wr.append(value)

def find_rank(wr, w0):
	"""
	Return the list of ranks for the solution kappa.

	Parameters:
	wr -- list of Decimal
	w0 -- Decimal

	Return:
	rank -- list of integer
	wr   -- list of Decimal
	"""
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

def compute_rank_wr(wr, w0):
	"""
	Compute intervals for each ranks in order to plot them.

	Parameters:
	wr -- list of Decimal
	w0 -- Decimal

	Return:
	rank_wr -- list of tuples
	"""
	# Transform Decimal into float
	getcontext().prec = 8
	wr = [float(w) for w in wr]

	# Sort by w
	wr = sorted(wr)

	rank, wr = find_rank(wr, w0)

	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr   = [i for i in zip(wr, wr)]
	
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr   = list(itertools.chain.from_iterable(wr))
	
	# Insert edges for plotting
	wr.insert(0, 0)
	wr.append(1)

	# Cut the edges
	rank = rank[1:-1]
	wr   = wr[1:-1]

	# w_i = (-1)^i * w_i
	# To determine the intervals of the rank values:
	# - negative value is the beginning of the interval from the origin
	# - positive value is the end of the interval from the origin
	wr = [((-1)**i)*wr[i] for i in range(len(wr))]

	# Group to save and manipulate later on
	rank_wr = [i for i in zip(rank, wr)]

	return rank_wr

def pr_success(df, pr_end, num_sim):
	"""
	Computes probability of success for a w value regarding the rank in order to plot it.
	Count down from the max w value to the min one the probability of success
	according to the number of simulations .
	Need to specify the end probability of success for a rank (pr_end).

	Parameters:
	df 	   	-- DataFrame
	pr_end 	-- integer
	num_sim -- float

	Return:
	df -- DataFrame
	"""
	# Probability of a simulation
	pr_sim = 1 / num_sim

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

def list_to_frames(num_rank, rank_wr_list):
	"""
	Transform list in Dataframes for plotting.

	Parameters:
	num_rank 	 -- integer
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

def create_fig(K, kappa, frames):
	"""
	Create figure to plot or to save

	Parameters:
	K 	   -- string
	kappa  -- string
	frames -- list of DataFrames

	Return:
	None
	"""
	fig = plt.figure()

	rank = 1
	for frame in frames:
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='rank %s' % (rank))
		rank += 1

	plt.legend(loc='upper left')
	plt.title(r'DoM with absolute scalar product | K=%s & kappa=%s' %(K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')

def write_csv(n, i, K, kappa, num_sim, frames):
	"""
	Write Dataframes in CSV files.

	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	frames  -- list of DataFrames

	Return:
	None
	"""
	num_rank = 1
	for df in frames:
		file_name = r'./csv/1bit_%srow_abs_num-sim=%s_K=%s_kappa=%s_i=%s_rank%s.csv' % (n, num_sim, K, kappa, i, num_rank)
		df.to_csv(file_name, encoding='utf-8', index=False)
		num_rank += 1

def sim_1bit(n, i, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer

	Return: None
	"""
	# Message
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	# Secret values for i-th bit
	key_i = gen_key_i(i, kappa, K)
	key   = gen_key(i, n)

	# Signal reference vectors
	S_ref = signal_2D(mu, i, key, n)

	# Save the results of the experiments
	rank_wr_list = []

	""" Generate many simulations """
	for j in range(num_sim):
	
		# Noise power vector
		R = noise(mu)

		# Initial scalar product <S-ref, P = R>s
		P_init 		= [Decimal(r) for r in R]
		scalar_init = np.dot(S_ref, P_init)

		# List of indexes for scalar product
		list_kappa_idx = kappa_idx(n)

		solution_idx = [sublist for sublist in list_kappa_idx[i] 
					   if (sublist[2] == key_i[(i+1) % n]) and (sublist[3] == key_i[(i+2) % n])]
		solution_idx = list(itertools.chain.from_iterable(solution_idx))

		nonsol_idx   = [sublist for sublist in list_kappa_idx[i] if (sublist[0] != solution_idx[0])]

		# Find initial scalar product values
		solution_init = scalar_init[solution_idx[1]]

		nonsol_init = []
		for j in range(3):
			nonsol_init.append(scalar_init[nonsol_idx[j][1]])

		# Determine the sign of initial solution value depending on the activity of the register at bit i
		p_i = xor_secret_i(K, kappa, i)
		solution_init = solution_init * p_i

		# ------------------------------------------------------------------------------------------- #
		
		# List of all intersections with the solution kappa function
		wr = []

		# Interval of w for abscissas
		interval = [0, 1]

		# Find the fiber such as f(w0) = 0 
		w0 = find_abs_w0(solution_init, len(mu))

		# Intersections between solution function and nonsol functions
		for j in range(3):
			wr1_nonsol, wr2_nonsol = find_wr(solution_init, nonsol_init[j], len(mu))
			append_wr(wr1_nonsol, wr)
			append_wr(wr2_nonsol, wr)

		# ------------------------------------------------------------------------------------------- #

		# Compute rank from intersections wr
		rank_wr = compute_rank_wr(wr, w0)

		# Save result for the specific noise vector R
		rank_wr_list += rank_wr
		
	return rank_wr_list

def sim_1bit_to_csv(n, i, K, kappa, num_sim):
	"""
	Plot probabilities of success for all possibilities for several simulations. 
	
	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer

	Return: None
	"""
	rank_wr_list = sim_1bit(n, i, K, kappa, num_sim)
	
	num_rank = 4
	frames   = list_to_frames(num_rank, rank_wr_list)

	# Compute probabilities of success
	for rank in range(num_rank):
		if rank == 0: # 1st rank
			frames[rank] = pr_success(frames[rank], 1, num_sim)
		else:
			frames[rank] = pr_success(frames[rank], 0, num_sim)

	# create_fig(K, kappa, frames)
	# plt.savefig('./plot/1bit_%srow_abs_num-sim=%s_K=%s_kappa=%s_i=%s.png' % (n, num_sim, K, kappa, i))
	# plt.close(fig)
	# plt.show()

	write_csv(n, i, K, kappa, num_sim, frames)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='n, i, K, kappa, and num_sim')

	parser.add_argument('n', metavar='n', type=int, default=1, help='length of chi row in {3, 5}')
	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, n-1]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if (args.n != 3) and (args.n != 5):
		print('\n**ERROR**')
		print('Required length of chi row: 3 or 5 bits')

	else:
		if (args.n == 3) and (len(args.K) != 3 or len(args.kappa) != 3 or (args.i < 0) or (args.i > 2)):
				print('\n**ERROR**')
				print('Required length of K and kappa: 3')
				print('Index studied bit i: [0, 2]\n')
		elif (args.n == 5) and (len(args.K) != 5 or len(args.kappa) != 5 or (args.i < 0) or (args.i > 4)):
				print('\n**ERROR**')
				print('Required length of K and kappa: 5')
				print('Index studied bit i: [0, 4]\n')

		else:
			# Length of signal part
			n = args.n

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
