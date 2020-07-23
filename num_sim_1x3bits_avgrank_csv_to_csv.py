#!/usr/bin/env python

"""AVERAGE RANK FOR 3-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__ 	= "2.0"

import pandas as pd
import argparse
import time
import sys

def load_csv(K, kappa, num_sim, rank, type_sp):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations 
	for a combinaison of either three absolute or squared scalar products. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	rank -- integer
	type_sp -- string

	Return:
	df -- DataFrame
	"""

	file_name =  r'./csv/1x3bits_%s_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (type_sp, num_sim, K, kappa, rank)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_csv_1x3bits_avgrank(K, kappa, num_sim, type_sp):
	"""
	Compute average rank for a given (K, kappa) for num_sim simulations 
	for a combinaison of either three absolute or squared scalar products. 
	from CSV file to save in another CSV file. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	type_sp -- string

	Return:
	NaN
	"""

	rank_wr_df1 = load_csv(K, kappa, num_sim, 1, type_sp)
	rank_wr_df2 = load_csv(K, kappa, num_sim, 2, type_sp)
	rank_wr_df3 = load_csv(K, kappa, num_sim, 3, type_sp)
	rank_wr_df4 = load_csv(K, kappa, num_sim, 4, type_sp)
	rank_wr_df5 = load_csv(K, kappa, num_sim, 5, type_sp)
	rank_wr_df6 = load_csv(K, kappa, num_sim, 6, type_sp)
	rank_wr_df7 = load_csv(K, kappa, num_sim, 7, type_sp)
	rank_wr_df8 = load_csv(K, kappa, num_sim, 8, type_sp)

	# Group dataframes in one
	frames = [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4, 
			  rank_wr_df5, rank_wr_df6, rank_wr_df7, rank_wr_df8]
	avgrank_df = pd.concat(frames)

	# Round 'wr'
	decimals = 7
	avgrank_df['wr'] = avgrank_df['wr'].apply(lambda x: round(x, decimals))

	# Sort by ascending order on 'wr' values
	avgrank_df = avgrank_df.iloc[avgrank_df['wr'].argsort()].reset_index(drop=True)
	avgrank_df = avgrank_df.sort_values(['wr', 'rank'], ascending=True).reset_index(drop=True)

	# Mean for 'pr_success' for each ['wr', 'rank'] tuples. 
	avgrank_df = avgrank_df.groupby(['wr', 'rank'])['pr_success'].mean().reset_index()

	# Compute avgrank = SUM ('pr_success' * 'rank') per 'wr' value
	ranks = [1, 2, 3, 4, 5, 6, 7, 8]
	condition = avgrank_df.groupby('wr')['rank'].agg(lambda x: set(ranks).issubset(set(x)))
	avgrank_df = avgrank_df['rank'].mul(avgrank_df['pr_success']).groupby(avgrank_df['wr']).sum().where(condition).reset_index(name='avgrank')

	# Write in CSV file
	avgrank_df = avgrank_df.dropna(subset=['avgrank'])
	file_name = r'./csv/1x3bits_%s_num-sim=%s_K=%s_kappa=%s_avgrank.csv' % (type_sp, num_sim, K, kappa)
	avgrank_df.to_csv(file_name, encoding='utf-8', index=False)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, K, kappa and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: average rank for combinaison of absolute scalar products\nmode 2: average rank for combinaison of squared scalar products')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()


	if len(args.K) != 3 or len(args.kappa) != 3 or (args.mode < 1) or (args.mode > 2):
		print('\n**ERROR**')
		print('Required length of K and kappa: 3')
		print('Available modes: 1 (abs) or 2 (sq)\n')

	else:
		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		if args.mode == 1: 

			# Time option
			start_t = time.perf_counter()

			csv_to_csv_1x3bits_avgrank(K, kappa, num_sim, 'abs')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			# Time option
			start_t = time.perf_counter()

			csv_to_csv_1x3bits_avgrank(K, kappa, num_sim, 'sq')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))