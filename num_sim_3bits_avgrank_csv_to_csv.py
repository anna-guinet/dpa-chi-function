#!/usr/bin/env python

"""AVERAGE RANK ON 3-BIT CHI ROW & 3-BIT K"""

__author__      = "Anna Guinet"
__email__		= "a.guinet@cs.ru.nl"
__version__ 	= "2.0"

import pandas as pd
import argparse
import time
import sys

def load_csv(K, kappa, num_sim, rank):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations
	for one scalar product for 3 bits of the register. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	rank -- integer

	Return:
	df -- DataFrame
	"""

	file_name =  r'./csv/3bits_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (num_sim, K, kappa, rank)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_csv_3bits_avgrank(K, kappa, num_sim):
	"""
	Compute average rank for a given (K, kappa) tuple for num_sim simulations
	for one scalar product for 3 bits of the register
	from CSV file to save in another CSV file. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer

	Return:
	NaN
	"""

	frames = []
	for i in range(64):
		df = load_csv(K, kappa, num_sim, i+1)
		frames.append(df)

	# Group dataframes in one
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
	ranks = [count_rank for count_rank in range(1, 65)]
	condition = avgrank_df.groupby('wr')['rank'].agg(lambda x: set(ranks).issubset(set(x)))
	avgrank_df = avgrank_df['rank'].mul(avgrank_df['pr_success']).groupby(avgrank_df['wr']).sum().where(condition).reset_index(name='avgrank')

	# Write in CSV file
	avgrank_df = avgrank_df.dropna(subset=['avgrank'])
	file_name = r'./csv/3bits_num-sim=%s_K=%s_kappa=%s_avgrank.csv' % (num_sim, K, kappa)
	avgrank_df.to_csv(file_name, encoding='utf-8', index=False)


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
		# Initial i-th bit register state value
		K = args.K

		# Key value after linear layer
		kappa = args.kappa

		# Number of simulations
		num_sim = args.num_sim

		# Time option
		start_t = time.perf_counter()

		csv_to_csv_3bits_avgrank(K, kappa, num_sim)

		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))