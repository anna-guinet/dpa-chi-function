#!/usr/bin/env python

"""LIBRARY FOR DPA ON KECCAK 3-BIT CHI ROW & 3-BIT K"""

__author__      = "Anna Guinet"
__email__		= "email@annagui.net"
__version__ 	= "2.0"

import pandas as pd
import matplotlib.pylab as plt

import argparse
import time
import sys

def load_csv(K, kappa, num_sim, rank):
	"""
	Load data from CSV file. 

	Parameters:
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	rank 	-- integer

	Return:
	df -- DataFrame
	"""
	file_name =  r'./csv/3bits_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (num_sim, K, kappa, rank)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_plot_3bits(K, kappa, num_sim):
	"""
	Plot from CSV files. 

	Parameters:
	K 		-- string
	kappa 	-- string
	num_sim -- integer

	Return: None
	"""
	num_rank = 64

	# Load data from CSV files
	frames = []
	for rank in range(num_rank):
		df = load_csv(K, kappa, num_sim, rank + 1)
		frames.append(df)

	# Plot for data from CSV files
	fig = plt.figure(figsize=(5,4))

	rank = 1
	for frame in frames:
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='Rank %s' % (rank))
		rank += 1

	# plt.legend(loc='upper right', ncol=2, fancybox=True, fontsize='x-small')
	plt.title(r'CPA for K=%s & kappa=%s' % (K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.savefig('./plot/3bits_num-sim=%s_K=%s_kappa=%s.pdf' % (num_sim, K, kappa), dpi=300)
	plt.close(fig)

	# plt.show()

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='K, kappa, and num_sim')

	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value (default: 000)')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value (default: 000)')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations to perform (default: 100)')

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

		csv_to_plot_3bits(K, kappa, num_sim)

		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))