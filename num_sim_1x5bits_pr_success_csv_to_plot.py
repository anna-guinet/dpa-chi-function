"""LIBRARY FOR DPA ON KECCAK 5-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__		= "email@annagui.net"
__version__ 	= "1.1"

import pandas as pd
import matplotlib.pylab as plt

import argparse
import time
import sys

def load_csv(K, kappa, num_sim, rank, type_sp):
	"""
	Load data from CSV file.

	Parameters:
	K 	  	-- string
	kappa 	-- string
	num_sim -- integer
	rank 	-- integer
	type_sp -- string

	Return:
	df -- DataFrame
	"""
	file_name =  r'./csv/1x5bits_%s_num-sim=%s_K=%s_kappa=%s_rank%s.csv' % (type_sp, num_sim, K, kappa, rank)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_plot_1x5bits_one_type(K, kappa, num_sim, type_sp):
	"""
	Plot for a given (K, kappa) tuple from CSV files. 

	Parameters:
	K 		-- string
	kappa 	-- string
	num_sim -- integer
	type_sp -- string

	Return: None
	"""
	frames = []

	for rank in range(1, 33):
		frame = load_csv(K, kappa, num_sim, rank, type_sp)
		frames.append(frame)

	# Save plot for probabilities of success of all ranks
	fig = plt.figure()

	rank = 1
	for frame in frames:
		plt.plot(frame['wr'], frame['pr_success'], '-', markersize=2, label='rank %s' % (rank))
		rank += 1

	plt.legend(loc='upper right')
	plt.title(r'Combined DoM for K=%s & kappa=%s (%s)' %(K, kappa, type_sp))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.savefig('./plot/1x5bits_%s_num-sim=%s_K=%s_kappa=%s.pdf' % (type_sp, num_sim, K, kappa), dpi=300)
	plt.close(fig)
	
	# Save plot for probabilities of success for rank 1
	fig1 = plt.figure()
	plt.title(r'Combined DoM for K=%s & kappa=%s | Rank 1 (%s)' %(K, kappa, type_sp))
	plt.plot(frames[0]['wr'],frames[0]['pr_success'], '-', color='tab:blue', markersize=2, label='rank 1')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([0, 1.08])
	plt.legend(loc='upper left')
	plt.savefig('./plot/1x5bits_%s_num-sim=%s_K=%s_kappa=%s_rank1.pdf' % (type_sp, num_sim, K, kappa), dpi=300)
	plt.close(fig1)

	# Plots for rank 2 to rank 32
	for rank in range(1, 32):
		fig = plt.figure()
		plt.title(r'Combined DoM for K=%s & kappa=%s | Rank %s (%s)' %(K, kappa, rank, type_sp))
		plt.plot(frames[rank]['wr'], frames[rank]['pr_success'], '-', markersize=2, label='rank %s' % (rank + 1))
		plt.xlabel('Weight w')
		plt.ylabel('Probability of success')
		plt.ylim([-0.01, 0.18])
		plt.legend(loc='upper right')
		plt.savefig('./plot/1x5bits_%s_num-sim=%s_K=%s_kappa=%s_rank%s.pdf' % (type_sp, num_sim, K, kappa, rank), dpi=300)
		plt.close(fig)

	# plt.show()

def csv_to_plot_1x5bits_two_types(K, kappa, num_sim):
	"""
	Plot probabilities of success from CSV file. 

	Parameters:
	K 		-- string
	kappa 	-- string
	num_sim -- integer

	Return: None
	"""
	frames_sq, frames_abs = [], []

	for rank in range(1, 33):
		frame = load_csv(K, kappa, num_sim, rank, 'abs')
		frames_abs.append(frame)

	for rank in range(1, 33):
		frame = load_csv(K, kappa, num_sim, rank, 'sq')
		frames_sq.append(frame)

	# Save plot for probabilities of success for all ranks
	fig = plt.figure(figsize=(6.5,5))

	colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
			  'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']

	rank = 1
	for frame in frames_abs:
		plt.plot(frame['wr'], frame['pr_success'], '-', color=colors[(rank - 1) % 10], markersize=2, label='Rank %s (abs)' % (rank))
		rank += 1

	rank = 1
	for frame in frames_sq:
		plt.plot(frame['wr'], frame['pr_success'], '--', color=colors[(rank - 1) % 10], markersize=2, label='Rank %s (sq)' % (rank))
		rank += 1

	# plt.legend(loc='upper left', ncol=2, fancybox=True, fontsize = 'small')
	plt.title(r'Combined DoM for K=%s & kappa=%s' %(K, kappa))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1x5bits_num-sim=%s_K=%s_kappa=%s.pdf' % (num_sim, K, kappa), dpi=300)
	plt.close(fig)

	# Save plot for probabilities of success for ranks 2, 3 
	fig23 = plt.figure(figsize=(7,5))
	plt.title(r'Combined DoM for K=%s & kappa=%s | Ranks 2 and 3' %(K, kappa))
	plt.plot(frames_abs[1]['wr'], frames_abs[1]['pr_success'], '-', color='tab:orange', markersize=2, label='Rank 2 (abs)')
	plt.plot(frames_sq[1]['wr'],  frames_sq[1]['pr_success'],  '-', color='tab:purple', markersize=2, label='Rank 2 (sq)')
	plt.plot(frames_abs[2]['wr'], frames_abs[2]['pr_success'], '-', color='tab:green', markersize=2, label='Rank 3 (abs)')
	plt.plot(frames_sq[2]['wr'],  frames_sq[2]['pr_success'],  '-', color='tab:red', markersize=2, label='Rank 3 (sq)')
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	plt.ylim([-0.01, 0.258])
	plt.legend(loc='upper right')
	fig23.tight_layout()
	plt.savefig('./plot/1x5bits_num-sim=%s_K=%s_kappa=%s_rank23.pdf' % (num_sim, K, kappa), dpi=300)
	plt.close(fig23)

	# plt.show()

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='mode, K, kappa, and num_sim')

	parser.add_argument('mode', metavar='mode', type=int, default=1, 
						help='mode 1: Probability of success with absolute scalar product | mode 2: Probability of success with squared scalar product | mode 3: Probability of success with absolute and squared scalar product')
	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 5 or len(args.kappa) != 5 or (args.mode < 1) or (args.mode > 3):
		print('\n**ERROR**')
		print('Required length of K and kappa: 5')
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

			csv_to_plot_1x5bits_one_type(K, kappa, num_sim, 'abs')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1x5bits_one_type(K, kappa, num_sim, 'sq')

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 3: 

			# Time option
			start_t = time.perf_counter()

			csv_to_plot_1x5bits_two_types(K, kappa, num_sim)

			# Time option
			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))