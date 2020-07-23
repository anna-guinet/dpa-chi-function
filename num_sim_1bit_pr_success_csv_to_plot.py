#!/usr/bin/env python

"""SUCCESS PROBA DPA 1-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__ 	= "2.0"

import pandas as pd
import matplotlib.pylab as plt
import argparse
import time
import sys

def load_csv(i, K, kappa, num_sim, rank, type_sp):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations
	for either 1 absolute or squared scalar product. 

	Parameters:
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

def csv_to_plot_1bit_one_type(i, K, kappa, num_sim, type_sp):
	"""
	Plot probability of success for a given (K, kappa) tuple for num_sim simulations
	for either 1 absolute or squared scalar product
	from CSV files. 

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

	"""" Save plot for probabilities of success of all ranks """

	fig = plt.figure()
	num_rank = 0 
	for frame in [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4]:
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='Rank %s' % (num_rank + 1))
		num_rank += 1

	plt.legend(loc='upper left')
	plt.title(r'One-bit %s strategy for K_%s=%s & kappa=%s' %(type_sp, i, K[i], kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1bit_%s_num-sim=%s_K=%s_kappa=%s_i=%s.png' % (type_sp, num_sim, K, kappa, i))
	plt.close(fig)

	"""" Save plot for probabilities of success for each rank """

	fig1 = plt.figure()
	plt.title(r'One-bit %s strategy for K_%s=%s & kappa=%s | Rank 1' %(type_sp, i, K[i], kappa))
	plt.plot(rank_wr_df1['wr'], rank_wr_df1['pr_success'], '-', color='tab:blue', markersize=2, label='Rank 1')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([0, 1.08])
	plt.legend(loc='upper left')
	fig.tight_layout()
	plt.savefig('./plot/1bit_%s_num-sim=%s_K=%s_kappa=%s_i=%s_rank1.pdf' % (type_sp, num_sim, K, kappa, i))
	plt.close(fig1)

	# Plots for ranks 2 to 4
	num_rank = 2
	color_list = ('tab:orange', 'tab:green', 'tab:red')
	for frame in [rank_wr_df2, rank_wr_df3, rank_wr_df4]:
		fig = plt.figure()
		plt.title(r'One-bit %s strategy for K_%s=%s & kappa=%s | Rank %s' %(type_sp, i, K[i], kappa, num_rank))
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, color=color_list[num_rank-1], label='Rank %s' % num_rank)
		plt.xlabel('Weight w')
		plt.ylabel('Probability of success')
		plt.ylim([-0.01, 0.258])
		plt.legend(loc='upper right')
		fig.tight_layout()
		plt.savefig('./plot/1bit_%s_num-sim=%s_K=%s_kappa=%s_i=%s_rank%s.pdf' % (type_sp, num_sim, K, kappa, i, num_rank))
		plt.close(fig)
		num_rank += 1
	
	# Show the plots
	# plt.show()

def csv_to_plot_1bit_two_types(i, K, kappa, num_sim):
	"""
	Plot probabilities of success for a given (K, kappa) tuple for num_sim simulations
	for both 1 absolute and squared scalar product
	from CSV file. 

	Parameters:
	i -- integer
	K -- string
	kappa - string
	num_sim -- integer

	Return:
	NaN
	"""

	rank_wr_df1_abs = load_csv(i, K, kappa, num_sim, 1, 'abs')
	rank_wr_df2_abs = load_csv(i, K, kappa, num_sim, 2, 'abs')
	rank_wr_df3_abs = load_csv(i, K, kappa, num_sim, 3, 'abs')
	rank_wr_df4_abs = load_csv(i, K, kappa, num_sim, 4, 'abs')

	rank_wr_df1_sq = load_csv(i, K, kappa, num_sim, 1, 'sq')
	rank_wr_df2_sq = load_csv(i, K, kappa, num_sim, 2, 'sq')
	rank_wr_df3_sq = load_csv(i, K, kappa, num_sim, 3, 'sq')
	rank_wr_df4_sq = load_csv(i, K, kappa, num_sim, 4, 'sq')


	"""" Save plot for probabilities of success for all ranks """

	fig = plt.figure(figsize=(6,5))
	color_list = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')

	num_rank = 0 
	for frame in [rank_wr_df1_abs, rank_wr_df2_abs, rank_wr_df3_abs, rank_wr_df4_abs]:
		plt.plot(frame['wr'], frame['pr_success'], '-', color=color_list[num_rank], markersize=2, label='Rank %s (abs)' % (num_rank + 1))
		num_rank += 1

	num_rank = 0 
	for frame in [rank_wr_df1_sq, rank_wr_df2_sq, rank_wr_df3_sq, rank_wr_df4_sq]:
		plt.plot(frame['wr'], frame['pr_success'], '--', color=color_list[num_rank], markersize=2, label='Rank %s (sq)' % (num_rank + 1))
		num_rank += 1

	plt.legend(loc='upper left')
	plt.title(r'One-bit strategy for K_%s=%s & kappa=%s' %(i, K[i], kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1bit_num-sim=%s_K=%s_kappa=%s_i=%s.png' % (num_sim, K, kappa, i))
	plt.close(fig)

	"""" Save plot for probabilities of success for each rank """

	fig1 = plt.figure()
	plt.title(r'One-bit strategy for K_%s=%s & kappa=%s | Rank 1' % (i, K[i], kappa))
	plt.plot(rank_wr_df1_abs['wr'], rank_wr_df1_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df1_sq['wr'], rank_wr_df1_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([0, 1.08])
	plt.legend(loc='lower right')
	fig.tight_layout()
	plt.savefig('./plot/1bit_num-sim=%s_K=%s_kappa=%s_i=%s_rank1.png' % (num_sim, K, kappa, i))
	plt.close(fig1)

	fig2 = plt.figure()
	plt.title(r'One-bit strategy for K_%s=%s & kappa=%s | Rank 2' %(i, K[i], kappa))
	plt.plot(rank_wr_df2_abs['wr'], rank_wr_df2_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df2_sq['wr'], rank_wr_df2_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	fig.tight_layout()
	plt.savefig('./plot/1bit_num-sim=%s_K=%s_kappa=%s_i=%s_rank2.png' % (num_sim, K, kappa, i))
	plt.close(fig2)

	fig3 = plt.figure()
	plt.title(r'One-bit strategy for K_%s=%s & kappa=%s | Rank 3' % (i, K[i], kappa))
	plt.plot(rank_wr_df3_abs['wr'], rank_wr_df3_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df3_sq['wr'], rank_wr_df3_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	fig.tight_layout()
	plt.savefig('./plot/1bit_num-sim=%s_K=%s_kappa=%s_i=%s_rank3.png' % (num_sim, K, kappa, i))
	plt.close(fig3)

	fig4 = plt.figure()
	plt.title(r'One-bit strategy for K_%s=%s & kappa=%s | Rank 4' %(i, K[i], kappa))
	plt.plot(rank_wr_df4_abs['wr'], rank_wr_df4_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df4_sq['wr'], rank_wr_df4_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	fig.tight_layout()
	plt.savefig('./plot/1bit_num-sim=%s_K=%s_kappa=%s_i=%s_rank4.png' % (num_sim, K, kappa, i))
	plt.close(fig4)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, i, K, kappa and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: Probability of success with absolute scalar product\n mode 2: Probability of success with squared scalar product\n mode 3: Probability of success with absolute and squared scalar product')
	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, 2]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3 or (args.mode < 1) or (args.mode > 3) or (args.i < 0) or (args.i > 2):
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

			csv_to_plot_1bit_one_type(i, K, kappa, num_sim, 'abs')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1bit_one_type(i, K, kappa, num_sim, 'sq')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 3: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1bit_two_types(i, K, kappa, num_sim)

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))