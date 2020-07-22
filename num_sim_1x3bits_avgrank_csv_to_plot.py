#!/usr/bin/env python

"""AVERAGE RANK FOR 3-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__		= "a.guinet@cs.ru.nl"
__version__ 	= "2.0"

import pandas as pd
import matplotlib.pylab as plt
import argparse
import time
import sys

def load_csv(K, kappa, num_sim, type_sp):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations
	for a combinaison of either three absolute or squared scalar products.

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	type_sp -- string

	Return:
	df -- DataFrame
	"""

	file_name =  r'./csv/1x3bits_%s_num-sim=%s_K=%s_kappa=%s_avgrank.csv' % (type_sp, num_sim, K, kappa)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_plot_1x3bits_one_type_avgrank(K, kappa, num_sim, type_sp):
	"""
	Plot average rank for a given (K, kappa) tuple for num_sim simulations 
	for a combinaison of either three absolute or squared scalar products. 
	from CSV file. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer
	type_sp -- string

	Return:
	NaN
	"""
	avgrank_df = load_csv(K, kappa, num_sim, type_sp)

	# Display a figure
	fig = plt.figure(figsize=(6,6))
	plt.plot(avgrank_df['wr'], avgrank_df['avgrank'], '-', markersize=2)

	plt.legend(loc='upper right')
	plt.title(r'Combining three %s scalar products for K=%s & kappa=%s' %(type_sp, K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Average rank')

	# Save figure
	plt.savefig('./plot/1x3bits_%s_num-sim=%s_K=%s_kappa=%s_avgrank.png' % (type_sp, num_sim, K, kappa))
	plt.close(fig)
	
	# Show the plots
	# plt.show()

def csv_to_plot_1x3bits_two_types_avgrank(K, kappa, num_sim):
	"""
	Plot average rank for a given (K, kappa) tuple for num_sim simulations 
	for a combinaison of both three absolute and squared scalar products
	from CSV file. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer

	Return:
	NaN
	"""
	avgrank_abs_df = load_csv(K, kappa, num_sim, 'abs')
	avgrank_sq_df = load_csv(K, kappa, num_sim, 'sq')

	# Display a figure
	fig = plt.figure(figsize=(10,5))

	plt.plot(avgrank_abs_df['wr'], avgrank_abs_df['avgrank'], '-', markersize=2, color='tab:blue', label='abs')
	plt.plot(avgrank_sq_df['wr'], avgrank_sq_df['avgrank'], '--', markersize=2, color='tab:purple', label='sq')

	plt.legend(loc='upper right')
	plt.title(r'Combining three scalar products for K=%s & kappa=%s' % (K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Average rank')

	# Save figure
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_avgrank.png' % (num_sim, K, kappa))
	plt.close(fig)
	
	# Show the plots
	# plt.show()

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, K, kappa and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: average rank for absolute scalar products \nmode 2: average rank for squared scalar products \nmode 3: average rank for absolute and squared scalar products')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3 or (args.mode < 1) or (args.mode > 5):
		print('\n**ERROR**')
		print('Required length of K and kappa: 3')
		print('Available modes: 1 (abs), 2 (sq) or 3 (sq and abs)\n')

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

			csv_to_plot_1x3bits_one_type_avgrank(K, kappa, num_sim, 'abs')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1x3bits_one_type_avgrank(K, kappa, num_sim, 'sq')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 3: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1x3bits_two_types_avgrank(K, kappa, num_sim)

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')


if __name__ == '__main__':
	sys.exit(main(sys.argv))