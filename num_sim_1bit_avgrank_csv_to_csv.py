#!/usr/bin/env python

"""AVERAGE RANK DPA 1-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__ 	= "1.0"

import pandas as pd

import argparse
import time
import sys

def load_csv(i, K, kappa, num_sim, rank, type_sp):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations for 
	one scalar product for either absolute or squared scalar products from CSV file.

	Parameters:
	i -- integer
	K -- string
	kappa - string
	num_sim -- integer
	rank -- integer
	type_sp -- string

	Return:
	df -- DataFrame
	"""

	file_name =  r'./csv/1bit_%s_num-sim=%s_K=%s_kappa=%s_i=%s_rank%s.csv' % (type_sp, num_sim, K, kappa, i, rank)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_csv_1bit_avgrank(i, K, kappa, num_sim, type_sp):
	"""
	Compute average rank for a given (K, kappa) tuple for num_sim simulations for 
	one scalar product for either absolute or squared scalar products
	from CSV file to save in another CSV file. 

	Parameters:
	i -- integer
	K -- string
	kappa - string
	num_sim -- integer
	type_sp -- string

	Return:
	NaN
	"""

	rank_wr_df1 = load_csv(i, K, kappa, num_sim, 1, type_sp)
	rank_wr_df2 = load_csv(i, K, kappa, num_sim, 2, type_sp)
	rank_wr_df3 = load_csv(i, K, kappa, num_sim, 3, type_sp)
	rank_wr_df4 = load_csv(i, K, kappa, num_sim, 4, type_sp)

	# Group dataframes in one
	frames = [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4]
	avgrank_df = pd.concat(frames)

	# Decimal precision 'wr'
	decimals = 7
	avgrank_df['wr'] = avgrank_df['wr'].apply(lambda x: round(x, decimals))

	# Sort by ascending order on 'wr' values
	avgrank_df = avgrank_df.iloc[avgrank_df['wr'].argsort()].reset_index(drop=True)
	avgrank_df = avgrank_df.sort_values(['wr', 'rank'], ascending=True).reset_index(drop=True)

	# Mean for 'pr_success' for each ['wr', 'rank'] tuples. 
	avgrank_df = avgrank_df.groupby(['wr', 'rank'])['pr_success'].mean().reset_index()

	# Compute avgrank = SUM ('pr_success' * 'rank') per 'wr' value
	ranks = [1, 2, 3, 4]
	condition = avgrank_df.groupby('wr')['rank'].agg(lambda x: set(ranks).issubset(set(x)))
	avgrank_df = avgrank_df['rank'].mul(avgrank_df['pr_success']).groupby(avgrank_df['wr']).sum().where(condition).reset_index(name='avgrank')

	# Write in CSV file
	avgrank_df = avgrank_df.dropna(subset=['avgrank'])
	file_name = r'./csv/1bit_%s_num-sim=%s_K=%s_kappa=%s_i=%s_avgrank.csv' % (type_sp, num_sim, K, kappa, i)
	avgrank_df.to_csv(file_name, encoding='utf-8', index=False)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, i, K, kappa and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: average rank for combinaison of absolute scalar products\n mode 2: average rank for combinaison of squared scalar products')
	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, 2]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3 or (args.mode < 1) or (args.mode > 2) or (args.i < 0) or (args.i > 2):
		print('\n**ERROR**')
		print('Required length of K and kappa: 3')
		print('Available modes: 1 (abs), 2 (sq) or 3 (sq and abs)\n')
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

		if args.mode == 1: 

			# Time option
			start_t = time.perf_counter()

			csv_to_csv_1bit_avgrank(i, K, kappa, num_sim, 'abs')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			# Time option
			start_t = time.perf_counter()

			csv_to_csv_1bit_avgrank(i, K, kappa, num_sim, 'sq')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))