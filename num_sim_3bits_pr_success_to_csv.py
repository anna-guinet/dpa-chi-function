#!/usr/bin/env python

"""DPA ON KECCAK 3-BIT CHI ROW & 3-BIT K

Save the figures for a simulation which display the outcome of a test
according to the SNR, in order to recover kappa bits. 
"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
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
	n  -- integer
	
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
	
	return 

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

def find_wr(a, b, value_b):
	"""
	Find the point when for the scalar product of the solution 
	equals the scalar product of a nonsolution.
	
	Parameters:
	a 		-- Decimal
	b 		-- Decimal
	value_b -- float
	
	Return:
	y -- Decimal
	"""	
	value = Decimal(24) - Decimal(value_b)

	if (value + b - a) != 0: 
		# w = (b - a)/ (value + b - a)
		w = (b - a) / (value + b - a)
	else:
		w = 0

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
	rank_wr -- list of tuples
	"""
	# Transform Decimal into float
	getcontext().prec = 8
	wr = [float(w) for w in wr]

	# Sort by w
	wr = sorted(wr)
				
	rank, wr = find_rank(wr)
		
	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr   = [i for i in zip(wr, wr)]
				
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr   = list(itertools.chain.from_iterable(wr))
				
	# Extract abscissas for wrsection points
	wr.insert(0, 0)
	wr.append(1)
			
	# Cut the edges
	rank = rank[1:-1]
	wr   = wr[1:-1]

	# w_i = (-1)^i * w_i
	# To determine the wrvals of the rank values:
	# - negative value is the beginning of the interval from the origin
	# - positive value is the end of the interrval from the origin
	wr = [((-1)**i)*wr[i] for i in range(len(wr))]
			
	# Group to save and manipulate later on
	rank_wr = [i for i in zip(rank, wr)]

	return rank_wr

def pr_success(df, pr_end, num_sim):
	"""
	Computes probability of success for a w value regarding the rank in order to plot it.
	COunt down from the max w value to the min one the probability of success
	according to the number of simulations (num_sim).
	Therefore, we need to specify the end probability of success for a rank (pr_end).

	Parameters:
	df 		-- DataFrame
	pr_end 	-- integer
	num_sim -- integer

	Return:
	df -- DataFrame
	"""
	pr_sim = 1 / num_sim

	if not df.empty:
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
		df['wr'] = df['wr'].abs()

	else:
		df['wr'] = 0
		df['pr_success'] = pr_end

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
	plt.title(r'CPA | K=%s & kappa=%s' %(K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')

def write_csv(K, kappa, num_sim, frames):
	"""
	Write Dataframes in CSV files.

	Parameters:
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	frames  -- list of DataFrames

	Return:
	None
	"""
	num_rank = 1
	for df in frames:
		file_name = r'./csv/3bits_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (num_sim, K, kappa, num_rank)
		df.to_csv(file_name, encoding='utf-8', index=False)
		num_rank += 1

def sim_3bits(K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n 		-- integer
	K 		-- string
	kappa   -- string
	num_sim -- integer

	Return: None
	"""
	# Length chi row
	n = 3

	# Message
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	# Signal power consumption values
	S_ref = signal_3D(mu, n)

	# Save the results of the experiments
	rank_wr_list = []

	# ------------------------------------------------------------------------------------------- #

	# Scalar product for full signal <S-ref, P = S>
	S 	  = signal_1D(mu, K, kappa, n)
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

	for j in range(num_sim):

		# Noise power consumption for m extra bits
		R = noise(mu)

		# Scalar product <S-ref, P = R>
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

		#------------------------------------------------------------------------------------------- #

		# Find intersection values
		wr_common_pos_16 = find_wr_common(16, 6, 4, solution_init, common_pos_16_idx, common_pos_16_init)
		wr_common_pos_8  = find_wr_common(8, 18, 19, solution_init, common_pos_8_idx, common_pos_8_init)
		wr_uncommon_0    = find_wr_common(0, 12, 16, solution_init, uncommon_0_idx, uncommon_0_init)
		wr_common_neg_8  = find_wr_common(-8, 18, 19, solution_init, common_neg_8_idx, common_neg_8_init)
		wr_common_neg_16 = find_wr_common(-16, 6, 4, solution_init, common_neg_16_idx, common_neg_16_init)
		wr_common_neg_24 = find_wr_common(-24, 1, 2, solution_init, common_neg_24_idx, common_neg_24_init)

		wr = wr_common_pos_16 + wr_common_pos_8 + wr_uncommon_0 +  wr_common_neg_8 +  wr_common_neg_16 +  wr_common_neg_24
	
		# Compute rank the specific noise vector R
		rank_wr = compute_rank_wr(wr)
		rank_wr_list += rank_wr
			
	return rank_wr_list

def sim_3bits_to_csv(K, kappa, num_sim):
	"""
	Plot probabilities of success for all possibilities for several simulations. 
	
	Parameters:
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	
	Return: None
	"""
	rank_wr_list = sim_3bits(K, kappa, num_sim)

	num_rank = 64
	frames   = list_to_frames(num_rank, rank_wr_list)

	# Compute probabilities of success
	for rank in range(num_rank):
		if rank == 0: # 1st rank
			frames[rank] = pr_success(frames[rank], 1, num_sim)
		else:
			frames[rank] = pr_success(frames[rank], 0, num_sim)

	# create_fig(K, kappa, frames)
	# plt.savefig('./plot/3bits_num-sim=%s_K=%s_kappa=%s.png' % (num_sim, K, kappa))
	# plt.close(fig)
	# plt.show()

	write_csv(K, kappa, num_sim, frames)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa, and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='numbre of simulations')

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

		sim_3bits_to_csv(K, kappa, num_sim)

		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))