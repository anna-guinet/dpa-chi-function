#!/usr/bin/env python

"""SUCCESS PROBA DPA 3-BIT CHI ROW & 1-BIT K """

__author__      = "Anna Guinet"
__email__		= "a.guinet@cs.ru.nl"
__version__ 	= "2.0"

import pandas as pd
import matplotlib.pylab as plt
import argparse
import time
import sys

def load_csv(K, kappa, num_sim, rank, type_sp):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations 
	for either a combination of 3 absolute or squared scalar products.

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

def csv_to_plot_1x3bits_one_type(K, kappa, num_sim, type_sp):
	"""
	Plot probability of success for a given (K, kappa) tuple for num_sim simulations
	for either a combination of 3 absolute or squared scalar products
	from CSV files. 

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

	"""" Save plot for probabilities of success of all ranks """

	fig = plt.figure()
	num_rank = 1
	for frame in [rank_wr_df1, rank_wr_df2, rank_wr_df3, rank_wr_df4,
				  rank_wr_df5, rank_wr_df6, rank_wr_df7, rank_wr_df8]:
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='rank %s' % (num_rank))
		num_rank += 1

	plt.legend(loc='upper right')
	plt.title(r'Combining three %s scalar products for K=%s & kappa=%s' %(type_sp, K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')

	plt.savefig('./plot/1x3bits_%s_num-sim=%s_K=%s_kappa=%s.png' % (type_sp, num_sim, K, kappa))
	plt.close(fig)
	
	"""" Save plot for probabilities of success for each rank """

	fig1 = plt.figure()
	plt.title(r'Combining three %s scalar products for K=%s & kappa=%s | Rank 1' %(type_sp, K, kappa))
	plt.plot(rank_wr_df1['wr'], rank_wr_df1['pr_success'], '-', color='tab:blue', markersize=2, label='rank 1')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([0, 1.08])
	plt.legend(loc='upper left')
	plt.savefig('./plot/1x3bits_%s_num-sim=%s_K=%s_kappa=%s_rank1.png' % (type_sp, num_sim, K, kappa))
	plt.close(fig1)

	# Plots for rank 2 to rank 8
	num_rank = 2
	for frame in [rank_wr_df2, rank_wr_df3, rank_wr_df4,
				  rank_wr_df5, rank_wr_df6, rank_wr_df7, rank_wr_df8]:
		fig = plt.figure()
		plt.title(r'Combining three %s scalar products for K=%s & kappa=%s | Rank %s' %(type_sp, K, kappa, num_rank))
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='rank %s' % num_rank)
		plt.xlabel('Weight w')
		plt.ylabel('Probability of success')
		plt.ylim([-0.01, 0.18])
		plt.legend(loc='upper right')
		plt.savefig('./plot/1x3bits_%s_num-sim=%s_K=%s_kappa=%s_rank%s.png' % (type_sp, num_sim, K, kappa, num_rank))
		plt.close(fig)
		num_rank += 1

def csv_to_plot_1x3bits_two_types(K, kappa, num_sim):
	"""
	Plot probabilities of success for a given (K, kappa) tuple for num_sim simulations
	for both a combination of 3 absolute and squared scalar products
	from CSV file. 

	Parameters:
	K -- string
	kappa - string
	num_sim -- integer

	Return:
	NaN
	"""

	rank_wr_df1_abs = load_csv(K, kappa, num_sim, 1, 'abs')
	rank_wr_df2_abs = load_csv(K, kappa, num_sim, 2, 'abs')
	rank_wr_df3_abs = load_csv(K, kappa, num_sim, 3, 'abs')
	rank_wr_df4_abs = load_csv(K, kappa, num_sim, 4, 'abs')
	rank_wr_df5_abs = load_csv(K, kappa, num_sim, 5, 'abs')
	rank_wr_df6_abs = load_csv(K, kappa, num_sim, 6, 'abs')
	rank_wr_df7_abs = load_csv(K, kappa, num_sim, 7, 'abs')
	rank_wr_df8_abs = load_csv(K, kappa, num_sim, 8, 'abs')

	rank_wr_df1_sq = load_csv(K, kappa, num_sim, 1, 'sq')
	rank_wr_df2_sq = load_csv(K, kappa, num_sim, 2, 'sq')
	rank_wr_df3_sq = load_csv(K, kappa, num_sim, 3, 'sq')
	rank_wr_df4_sq = load_csv(K, kappa, num_sim, 4, 'sq')
	rank_wr_df5_sq = load_csv(K, kappa, num_sim, 5, 'sq')
	rank_wr_df6_sq = load_csv(K, kappa, num_sim, 6, 'sq')
	rank_wr_df7_sq = load_csv(K, kappa, num_sim, 7, 'sq')
	rank_wr_df8_sq = load_csv(K, kappa, num_sim, 8, 'sq')

	"""" Save plot for probabilities of success for all ranks """

	fig = plt.figure(figsize=(6.5,5))
	color_list = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:grey')

	num_rank = 0 
	for frame in [rank_wr_df1_abs, rank_wr_df2_abs, rank_wr_df3_abs, rank_wr_df4_abs, rank_wr_df5_abs, rank_wr_df6_abs, rank_wr_df7_abs, rank_wr_df8_abs]:
		plt.plot(frame['wr'], frame['pr_success'], '-', color=color_list[num_rank], markersize=2, label='Rank %s (abs)' % (num_rank + 1))
		num_rank += 1

	num_rank = 0 
	for frame in [rank_wr_df1_sq, rank_wr_df2_sq, rank_wr_df3_sq, rank_wr_df4_sq, rank_wr_df5_sq, rank_wr_df6_sq, rank_wr_df7_sq, rank_wr_df8_sq]:
		plt.plot(frame['wr'], frame['pr_success'], '--', color=color_list[num_rank], markersize=2, label='Rank %s (sq)' % (num_rank + 1))
		num_rank += 1

	plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize = 'small')
	plt.title(r'Combining three scalar products for K=%s & kappa=%s' %(K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s.png' % (num_sim, K, kappa))
	plt.close(fig)

	"""" Save plot for probabilities of success for ranks 2, 3 """

	fig23 = plt.figure(figsize=(5,5))
	# plt.title(r'Combining three scalar products for K=%s & kappa=%s | Ranks 2 and 3' %(K, kappa))
	plt.plot(rank_wr_df2_abs['wr'], rank_wr_df2_abs['pr_success'], '-', color='tab:orange', markersize=2, label='Rank 2 (abs)')
	plt.plot(rank_wr_df2_sq['wr'], rank_wr_df2_sq['pr_success'], '-', color='tab:purple', markersize=2, label='Rank 2 (sq)')
	plt.plot(rank_wr_df3_abs['wr'], rank_wr_df3_abs['pr_success'], '-', color='tab:green', markersize=2, label='Rank 3 (abs)')
	plt.plot(rank_wr_df3_sq['wr'], rank_wr_df3_sq['pr_success'], '-', color='tab:red', markersize=2, label='Rank 3 (sq)')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	fig.tight_layout()
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank23.png' % (num_sim, K, kappa))
	plt.close(fig23)

	"""" Save plot for probabilities of success for each rank """

	fig1 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 1' %(K, kappa))
	plt.plot(rank_wr_df1_abs['wr'], rank_wr_df1_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df1_sq['wr'], rank_wr_df1_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([0, 1.08])
	plt.legend(loc='upper left')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank1.png' % (num_sim, K, kappa))
	plt.close(fig1)


	# Plots for rank 2 to rank 8
	fig2 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 2' %(K, kappa))
	plt.plot(rank_wr_df2_abs['wr'], rank_wr_df2_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df2_sq['wr'], rank_wr_df2_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank2.png' % (num_sim, K, kappa))
	plt.close(fig2)

	fig3 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 3' %(K, kappa))
	plt.plot(rank_wr_df3_abs['wr'], rank_wr_df3_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df3_sq['wr'], rank_wr_df3_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank3.png' % (num_sim, K, kappa))
	plt.close(fig3)

	fig4 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 4' %(K, kappa))
	plt.plot(rank_wr_df4_abs['wr'], rank_wr_df4_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df4_sq['wr'], rank_wr_df4_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank4.png' % (num_sim, K, kappa))
	plt.close(fig4)

	fig5 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 5' %(K, kappa))
	plt.plot(rank_wr_df5_abs['wr'], rank_wr_df5_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df5_sq['wr'], rank_wr_df5_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank5.png' % (num_sim, K, kappa))
	plt.close(fig5)

	fig6 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 6' %(K, kappa))
	plt.plot(rank_wr_df6_abs['wr'], rank_wr_df6_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df6_sq['wr'], rank_wr_df6_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.legend(loc='upper right')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank6.png' % (num_sim, K, kappa))
	plt.close(fig6)

	fig7 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 7' %(K, kappa))
	plt.plot(rank_wr_df7_abs['wr'], rank_wr_df7_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df7_sq['wr'], rank_wr_df7_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank7.png' % (num_sim, K, kappa))
	plt.close(fig7)

	fig8 = plt.figure()
	plt.title(r'Combining three scalar products for K=%s & kappa=%s | Rank 8' %(K, kappa))
	plt.plot(rank_wr_df8_abs['wr'], rank_wr_df8_abs['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(rank_wr_df8_sq['wr'], rank_wr_df8_sq['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	plt.savefig('./plot/1x3bits_num-sim=%s_K=%s_kappa=%s_rank8.png' % (num_sim, K, kappa))
	plt.close(fig8)

	# plt.show()

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, K, kappa and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: Probability of success for absolute scalar product\n mode 2: Probability of success for squared scalar product\n mode 3: Probability of success for absolute and squared scalar product')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()


	if len(args.K) != 3 or len(args.kappa) != 3 or (args.mode < 1) or (args.mode > 3):
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

			csv_to_plot_1x3bits_one_type(K, kappa, num_sim, 'abs')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1x3bits_one_type(K, kappa, num_sim, 'sq')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 3: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1x3bits_two_types(K, kappa, num_sim)

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))