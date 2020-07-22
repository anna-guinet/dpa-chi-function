#!/usr/bin/env python

"""AVERAGE RANK ON 3-BIT CHI ROW & 3-BIT K"""

__author__      = "Anna Guinet"
__email__		= "a.guinet@cs.ru.nl"
__version__ 	= "2.0"

import pandas as pd
import matplotlib.pylab as plt
import argparse
import time
import sys

def load_csv(K, kappa, num_sim):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations
	for one scalar product for 3 bits of the register.

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer

	Return:
	df -- DataFrame
	"""

	file_name =  r'./csv/3bits_num-sim=%s_K=%s_kappa=%s_avgrank.csv' % (num_sim, K, kappa)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_plot_3bits_avgrank(K, kappa, num_sim):
	"""
	Plot average rank for a given (K, kappa) tuple for num_sim simulations
	for one scalar product for 3 bits of the register 
	from CSV file. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer

	Return:
	NaN
	"""
	avgrank_df = load_csv(K, kappa, num_sim)

	# Display a figure
	fig = plt.figure(figsize=(6,5))
	fig.tight_layout()
	plt.plot(avgrank_df['wr'], avgrank_df['avgrank'], '-', markersize=2)

	# plt.legend(loc='upper right')
	plt.title(r'Three-bit strategy for K=%s & kappa=%s' %(K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Average rank')

	plt.savefig('./plot/3bits_num-sim=%s_K=%s_kappa=%s_avgrank.png' % (num_sim, K, kappa))
	plt.close(fig)
	
	# Show the plots
	# plt.show()

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

		csv_to_plot_3bits_avgrank(K, kappa, num_sim)

		# Time option
		end_t = time.perf_counter()
		print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))