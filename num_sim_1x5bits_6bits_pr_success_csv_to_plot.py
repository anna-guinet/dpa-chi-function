"""LIBRARY FOR DPA ON KECCAK 5-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__		= "email@annagui.net"
__version__ 	= "1.0"

import pandas as pd
import matplotlib.pylab as plt

import argparse
import time
import sys

def load_csv(K, kappa, bitK, num_sim, type_sp):
	"""
	Load data from CSV file.

	Parameters:
	K 	    -- string
	kappa   -- string
	bitK 	-- integer
	num_sim -- integer
	type_sp -- string

	Return:
	df -- DataFrame
	"""
	file_name =  r'./csv/1x5bits_6bits_%s_num-sim=%s_K=%s_kappa=%s_%sK.csv' % (type_sp, num_sim, K, kappa, bitK)

	# Load from CSV file
	df = pd.read_csv(file_name)

	return df

def csv_to_plot_6bits_one_type(K, kappa, num_sim, type_sp):
	"""
	Plot from CSV file. 

	Parameters:
	K 	   	-- string
	kappa   -- string
	num_sim -- integer
	type_sp -- string

	Return: None
	"""
	frames = []
	for bitK in range(6):
		df = load_csv(K, kappa, bitK, num_sim, type_sp)
		frames.append(df)

	# Save plot for probabilities of success for rank 1
	fig = plt.figure(figsize=(6,5))
	colors = ['tab:brown', 'tab:purple', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue']

	num_bitK = 0
	for frame in frames:
		plt.plot(frame['w'], frame['pr_success'], '-', color=colors[num_bitK], markersize=2, label='%s bits K' % (num_bitK))
		num_bitK += 1

	plt.legend(loc='upper left', ncol=1, fancybox=True, fontsize = 'small')
	plt.title('cDoM to recover kappa=%s and some K=%s bits (%s)' %(kappa, K, type_sp))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1x5bits_6bits_%s_num-sim=%s_K=%s_kappa=%s.pdf' % (type_sp, num_sim, K, kappa), dpi=300)
	plt.close(fig)
	# plt.show()

def csv_to_plot_6bits_two_types(K, kappa, num_sim):
	"""
	Plot from CSV file. 

	Parameters:
	K 	   	-- string
	kappa   -- string
	num_sim -- integer

	Return: None
	"""
	frames_abs = []
	for bitK in range(6):
		df = load_csv(K, kappa, bitK, num_sim, 'abs')
		frames_abs.append(df)

	frames_sq = []
	for bitK in range(6):
		df = load_csv(K, kappa, bitK, num_sim, 'sq')
		frames_sq.append(df)

	# Save plot for probabilities of success for rank 1
	fig = plt.figure(figsize=(5,4))
	colors = ['tab:brown', 'tab:purple', 'tab:red', 'tab:green', 'tab:orange', 'tab:blue']

	num_bitK = 0
	for frame in frames_abs:
		plt.plot(frame['w'], frame['pr_success'], '-', color=colors[num_bitK], markersize=2, label='%s bits K (abs)' % (num_bitK))
		num_bitK += 1

	num_bitK = 0
	for frame in frames_sq:
		plt.plot(frame['w'], frame['pr_success'], '--', color=colors[num_bitK], markersize=2, label='%s bits K (abs)' % (num_bitK))
		num_bitK += 1

	plt.legend(loc='upper left', ncol=1, fancybox=True, fontsize = 'small')
	plt.title('cDoM to recover kappa=%s and some K=%s bits' %(kappa, K))
	plt.xlabel('Weight w')
	plt.ylabel('Probability of success')
	fig.tight_layout()
	plt.savefig('./plot/1x5bits_6bits_num-sim=%s_K=%s_kappa=%s.pdf' % (num_sim, K, kappa), dpi=300)
	plt.close(fig)
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

			start_t = time.perf_counter()

			csv_to_plot_6bits_one_type(K, kappa, num_sim, 'abs')

			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 2: 

			start_t = time.perf_counter()

			csv_to_plot_6bits_one_type(K, kappa, num_sim, 'sq')

			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

		if args.mode == 3: 

			start_t = time.perf_counter()

			csv_to_plot_6bits_two_types(K, kappa, num_sim)

			end_t = time.perf_counter()
			print('time', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))