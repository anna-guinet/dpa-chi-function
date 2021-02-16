"""LIBRARY FOR DPA ON KECCAK n-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__		= "email@annagui.net"
__version__ 	= "2.1"

import pandas as pd
import matplotlib.pylab as plt
import math

import argparse
import time
import sys

def load_csv(n, i, K, kappa, num_sim, rank, type_sp):
	"""
	Load data from CSV file for a given (K, kappa) tuple for num_sim simulations 
	for a combinaison of n experiments on 1 bit of the register 
	for either absolute scalar product or squared one. 

	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa  	-- string
	num_sim -- integer
	rank 	-- integer
	type_sp -- string

	Return:
	df -- DataFrame
	"""
	file_name =  r'./csv/1bit_%srow_%s_num-sim=%s_K=%s_kappa=%s_i=%s_rank%s.csv' % (n, type_sp, num_sim, K, kappa, i, rank)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_plot_1bit_one_type(n, i, K, kappa, num_sim, type_sp):
	"""
	Plot guessing entropy for a given (K, kappa) tuple for num_sim simulations for 
	a combinaison of n experiments on 1 bit of the register 
	for either absolute scalar product or squared one from CSV files. 

	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	type_sp -- string

	Return: None
	"""
	frames = []
	for rank in range(1, 5):
		frame = load_csv(n, i, K, kappa, num_sim, rank, type_sp)
		frames.append(frame)

	# Save plot for probabilities of success of all ranks
	fig = plt.figure()
	num_rank = 1
	for frame in frames:
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='Rank %s' % (num_rank))
		num_rank += 1

	plt.legend(loc='upper left')
	plt.title(r'%s-row DoM %s for K_%s=%s & kappa=%s' %(n, type_sp, i, K[i], kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1bit_%srow_%s_num-sim=%s_K=%s_kappa=%s_i=%s.pdf' % (n, type_sp, num_sim, K, kappa, i), dpi=300)
	plt.close(fig)

	# plt.show()

def csv_to_plot_1bit_two_types(n, i, K, kappa, num_sim):
	"""
	Plot probabilities of success for a given (K, kappa) tuple for num_sim simulations for 
	a combinaison of n experiments on 1 bit of the register 
	for both absolute scalar product and squared on from CSV file. 

	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	type_sp -- string

	Return: None
	"""
	frames_abs = []
	for rank in range(1, 5):
		frame = load_csv(n, i, K, kappa, num_sim, rank, 'abs')
		frames_abs.append(frame)

	frames_sq = []
	for rank in range(1, 5):
		frame = load_csv(n, i, K, kappa, num_sim, rank, 'sq')
		frames_sq.append(frame)


	# Save plot for probabilities of success for all ranks
	fig = plt.figure(figsize=(6,5))
	color_list = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red')

	num_rank = 1
	for frame in frames_abs:
		plt.plot(frame['wr'], frame['pr_success'], '-', color=color_list[num_rank - 1], 
				 markersize=2, label='Rank %s (abs)' % (num_rank))
		num_rank += 1

	num_rank = 1
	for frame in frames_sq:
		plt.plot(frame['wr'], frame['pr_success'], '--', color=color_list[num_rank - 1], 
				 markersize=2, label='Rank %s (sq)' % (num_rank))
		num_rank += 1

	plt.legend(loc='upper left')
	plt.title(r'%s-row DoM for K_%s=%s & kappa=%s' %(n, i, K[i], kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1bit_%srow_num-sim=%s_K=%s_kappa=%s_i=%s.pdf' % (n, num_sim, K, kappa, i), dpi=300)
	plt.close(fig)

	# Save plot for probabilities of success for each rank
	fig1 = plt.figure()
	plt.title(r'%s-row DoM for K_%s=%s & kappa=%s | Rank 1' % (n, i, K[i], kappa))
	plt.plot(frames_abs[0]['wr'], frames_abs[0]['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
	plt.plot(frames_sq[0]['wr'],  frames_sq[0]['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([0, 1.08])
	plt.legend(loc='lower right')
	fig1.tight_layout()
	plt.savefig('./plot/1bit_%srow_num-sim=%s_K=%s_kappa=%s_i=%s_rank1.pdf' % (n, num_sim, K, kappa, i), dpi=300)
	plt.close(fig1)

	for rank in range(1, 4):
		fig2 = plt.figure()
		plt.title(r'%s-row DoM for K_%s=%s & kappa=%s  | Rank %s' %(n, i, K[i], kappa, rank + 1))
		plt.plot(frames_abs[rank]['wr'], frames_abs[rank]['pr_success'], '-', color='tab:blue', markersize=2, label='abs')
		plt.plot(frames_sq[rank]['wr'],  frames_sq[rank]['pr_success'], '--', color='tab:purple', markersize=2, label='sq')
		plt.xlabel('Weight w')
		plt.ylabel('Probability of success')
		plt.ylim([-0.01, 0.258])
		plt.legend(loc='upper right')
		fig2.tight_layout()
		plt.savefig('./plot/1bit_%srow_num-sim=%s_K=%s_kappa=%s_i=%s_rank%s.pdf' % (n, num_sim, K, kappa, i, rank + 1), dpi=300)
		plt.close(fig2)

	# plt.show()

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, n, i, K, kappa, and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: Probability of success with absolute scalar product  | mode 2: Probability of success with squared scalar product  | mode 3: Probability of success with absolute and squared scalar product')
	parser.add_argument('n', metavar='n', type=int, default=1, help='length of chi row in {3, 5}')
	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, n-1]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations to perform')

	args = parser.parse_args()

	if (args.n != 3) and (args.n != 5):
		print('\n**ERROR**')
		print('Required length of chi row: 3 or 5 bits')

	else:
		if (args.n == 3) and (len(args.K) != 3 or len(args.kappa) != 3 or (args.i < 0) or (args.i > 2) or (args.mode < 1) or (args.mode > 3)):
				print('\n**ERROR**')
				print('Required length of K and kappa: 3')
				print('Available modes: 1 (abs), 2 (sq) or 3 (sq and abs)\n')
				print('Index studied bit i: [0, 2]\n')
		elif (args.n == 5) and (len(args.K) != 5 or len(args.kappa) != 5 or (args.i < 0) or (args.i > 4) or (args.mode < 1) or (args.mode > 3)):
				print('\n**ERROR**')
				print('Required length of K and kappa: 5')
				print('Available modes: 1 (abs), 2 (sq) or 3 (sq and abs)\n')
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

			if args.mode == 1: 

				# Time option
				start_t = time.perf_counter()

				csv_to_plot_1bit_one_type(n, i, K, kappa, num_sim, 'abs')

				# Time option
				end_t = time.perf_counter()
				print('time', end_t - start_t, '\n')

			if args.mode == 2: 

				# Time option
				start_t = time.perf_counter()

				csv_to_plot_1bit_one_type(n, i, K, kappa, num_sim, 'sq')

				# Time option
				end_t = time.perf_counter()
				print('time', end_t - start_t, '\n')

			if args.mode == 3: 

				# Time option
				start_t = time.perf_counter()

				csv_to_plot_1bit_two_types(n, i, K, kappa, num_sim)

				# Time option
				end_t = time.perf_counter()
				print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))