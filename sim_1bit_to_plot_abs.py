#!/usr/bin/env python

"""DPA ON KECCAK n-BIT CHI ROW & 1-BIT K

Save the figures for a simulation which display the outcome of a test
according to the SNR, in order to recover kappa bits. 
"""

__author__      = "Anna Guinet"
__email__       = "email@annagui.net"
__version__     = "3.1"

import numpy as np
import itertools
from decimal import *
import math
import matplotlib.pylab as plt
import random

import argparse
import time
import sys

def noise(mu):
	"""
	Compute a noise power consumption vector R for each message mu. 

	Parameter:
	mu -- list of Bool

	Return:
	R -- list of integers, length 2^n
	"""
	R = []

	# For each message mu
	for j in range(len(mu)):

		# Activity of noise part
		if (len(mu) == 32):
			d = random.gauss(0, 2)
		elif (len(mu) == 8):
			d = random.gauss(0, 1)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def gen_key_i(i, kappa, K):
	"""
	Create key value where key equals kappa, except:
	key_(i mod n) = kappa_(i mod n) XOR K_(i mod n)

	Parameters:
	i 	  -- integer in [0,n-1]
	kappa -- string
	K 	  -- string

	Return:
	key -- list of bool
	"""
	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]

	K = list(K)
	K = [bool(int(j)) for j in K]

	# Initialize new key value
	key_i = kappa.copy()

	# XOR at indice i
	key_i[i] = K[i] ^ kappa[i]

	return key_i

def gen_key(i, n):
	"""
	Generate all key values for key_(i+2 mod n) and key_(i+1 mod n), regardless the i-th bit:
	- key_(i mod n) = kappa_i XOR K = 0
	- key_(i+1 mod n) = kappa_(i+1 mod n)
	- key_(i+2 mod n) = kappa_(i+2 mod n)
	- key_(i+3 mod n) = 0 if n = 5 
	- key_(i+4 mod n) = 0 if n = 5

	Eg: for i = 1, n = 5, key = [[ 0, 0, 0, 0, 0 ], [ 0, 0, 0, 1, 0 ], [ 0, 0, 1, 0, 0 ], [ 0, 0, 1, 1, 0 ]]
	
	Parameters:
	i -- integer in [0,n-1]
	n -- integer
	
	Return:
	key -- list of lists
	"""
	# All possible values for 2 bits
	key = list(itertools.product([bool(0), bool(1)], repeat=2))

	# Transform nested tuples into nested lists
	key = [list(j) for j in key]

	# Insert 'False' as the i-th, (i+3)-th and (i+4)-th positions
	if n == 5:
		key_extend = [j.insert((i+3) % n, bool(0)) for j in key]
		key_extend = [j.insert((i+4) % n, bool(0)) for j in key]

	# Insert 'False' as the i-th position
	key_extend = [j.insert(i % n, bool(0)) for j in key]

	return key

def signal_1D(mu, i, key_i, n):
	"""
	Generate signal power consumption S values for a secret state key, for all messages mu. 

	Parameters:
	mu 	  -- 2D list of bool, size n x 2^n
	i 	  -- integer in [0,n-1]
	key_i -- list of bool
	n 	  -- integer

	Return:
	S -- list of integers, length 2^n
	"""
	S = []

	# For each message mu
	for j in range(len(mu)):
			
		# Activity of the first storage cell of the register
		d = key_i[i] ^ mu[j][i] ^ ((key_i[(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key_i[(i+2) % n] ^ mu[j][(i+2) % n]))

		# Power consumption model
		S_j = (-1)**(d)

		S.append(S_j)

	return S

def signal_2D(n, i, key, mu):
	"""
	Generate signal power consumption S values for all messages mu.
	
	'key' is a value such as key = kappa, except for bit i:
	key_i = kappa_i XOR K = 0

	Parameters:
	n   -- integer
	i   -- integer in [0,n-1]
	key -- list of bool
	mu  -- 2D list of bool

	Return:
	S -- 2D list of integers
	"""	
	S = []

	# For each key value
	for l in range(len(key)):

		S_l = []

		# For each message mu
		for j in range(len(mu)):

			# Activity of the first storage cell of the register
			d = key[l][i] ^ mu[j][i] ^ ((key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n]))

			S_lj = (-1)**(d)

			S_l.append(S_lj)

		S += [S_l]

	return S

def kappa_idx(n):
	"""
	Provide scalar products indexes for kappa values.

	Parameter:
	n -- integer

	Return:
	list_kappa_idx -- list of lists
	"""
	list_kappa_idx = []

	if n == 5:
		list_kappa_idx0 = [['*00**', 0, False, False],
						   ['*01**', 1, False, True],
						   ['*10**', 2, True, False],
						   ['*11**', 3, True, True]]

		list_kappa_idx1 = [['**00*', 0, False, False],
						   ['**01*', 1, False, True],
						   ['**10*', 2, True, False],
						   ['**11*', 3, True, True]]

		list_kappa_idx2 = [['***00', 0, False, False],
						   ['***01', 1, False, True],
						   ['***10', 2, True, False],
						   ['***11', 3, True, True]]

		list_kappa_idx3 = [['0***0', 0, False, False],
						   ['0***1', 1, True, False],
						   ['1***0', 2, False, True],
						   ['1***1', 3, True, True]]

		list_kappa_idx4 = [['00***', 0, False, False],
						   ['01***', 1, False, True],
						   ['10***', 2, True, False],
						   ['11***', 3, True, True]]

		list_kappa_idx = [list_kappa_idx0, list_kappa_idx1, 
						  list_kappa_idx2, list_kappa_idx3, 
						  list_kappa_idx4]

	if n == 3:
		list_kappa_idx0 = [['*00', 0, False, False],
						   ['*01', 1, False, True],
						   ['*10', 2, True, False],
						   ['*11', 3, True, True]]

		list_kappa_idx1 = [['0*0', 0, False, False],
						   ['0*1', 1, True, False],
						   ['1*0', 2, False, True],
						   ['1*1', 3, True, True]]

		list_kappa_idx2 = [['00*', 0, False, False],
						  ['01*', 1, False, True],
						  ['10*', 2, True, False],
						  ['11*', 3, True, True]]

		list_kappa_idx = [list_kappa_idx0, list_kappa_idx1, list_kappa_idx2]

	return list_kappa_idx

def find_abs_w0(a, norm):
	"""
	Find the point when for the scalar product of the solution guess equals 0.
	
	Parameters:
	a    -- Decimal
	norm -- integer
	
	Return:
	w0 -- Decimal
	"""	
	w0 = 0

	if (a - Decimal(norm)) != 0:
		# w0 = a / a - norm
		w0 = a / (a - Decimal(norm))

	return w0
		
def fct_abs_solution(w, a, p_i, norm):
	"""
	Function for the scalar product of the solution guess.
	
	Parameters:
	w 	 -- float
	a 	 -- Decimal
	p_i  -- integer
	norm -- integer
	
	Return:
	y -- Decimal
	"""
	w = Decimal(w)
	constant = Decimal(norm) * Decimal(p_i)
	a = a * Decimal(p_i)

	# if a > 0, then y = (value - a) * w + a
	# if a < 0, then y = (a - value) * w - a
	y = (constant - a) * w + a
	y = abs(y)

	return y / Decimal(norm)
		
def fct_abs_nonsol(w, b, norm):
	"""
	Function for the scalar product of an nonsol guess.
	
	Parameters:
	w 	 -- float
	b 	 -- Decimal
	norm -- integer
	
	Return:
	y -- Decimal
	"""	
	b = abs(b)
		
	# y = -|b| * x + |b|
	y = - b * Decimal(w) + b
		
	return y / Decimal(norm)

def find_wr(a, b, norm):
	"""
	Find the point when for the scalar product of the solution
	quals the scalar product of a nonsol.
	
	Parameters:
	a 	 -- Decimal
	b 	 -- Decimal
	norm -- integer
	
	Return:
	y -- Decimal
	"""	
	b = abs(b)

	w1 = 0
	w2 = 0

	if (Decimal(norm) + (b - a)) != 0:

		# w1 = (|b| - a)/ (norm - (|b| - a))
		w1 = (b - a) / (Decimal(norm) + (b - a))
	
	if (b + a - Decimal(norm)) != 0:

		# w2 = (|b| + a)/ (|b| + a - norm)
		w2 = (b + a) / (b + a - Decimal(norm))

	return w1, w2

def append_wr(value, wr):
	"""
	Append value to wr if in [0,1]. 

	Parameters:
	value 	 -- Decimal
	wr 		 -- list of Decimal

	Return: None
	"""
	if (value != None) and (value > 0) and (value < 1):
		wr.append(value)

def xor_secret_i(K, kappa, i):
	""" 
	Compute power consumption at bit i = kappa_i XOR K_i
	
	Parameters:
	K 	  -- string
	kappa -- string
	i 	  -- integer
	
	Return:
	p_i -- float
	"""
	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = list(K)
	K = [bool(int(j)) for j in K]
	
	# Initialize new kappa values
	kappa_copy = kappa.copy()
	K_copy = K.copy()
	
	# XOR at indice i
	d = K_copy[i] ^ kappa_copy[i]
	
	# Power consumption at i-th bit
	p_i = (-1)**d

	return p_i

def test_loop(K, kappa, mu, i, key_i, n, R, S_ref, ax1):
	"""
	Control test.

	"""
	w_list = []

	# Signal power consumption for a key
	S = signal_1D(mu, i, key_i, n)

	# Parameters for loop
	scalar_00, scalar_01, scalar_10, scalar_11 = [], [], [], []

	for j in range(1001):
		# Weight of signal part
		w = j / 1000
		
		scalar = []
			
		# Global power consumption P = (1 - w) * R + w * S
		P = [((Decimal(1 - w) * Decimal(r))  + (Decimal(w) * Decimal(s))) for r, s in zip(R, S)]

		# Scalar product <S-ref, P>
		scalar = np.dot(S_ref, P)
		scalar_abs = abs(scalar)
		
		w_list.append(w)

		scalar_00.append(scalar_abs[0] / len(mu))
		scalar_01.append(scalar_abs[1] / len(mu))
		scalar_10.append(scalar_abs[2] / len(mu))
		scalar_11.append(scalar_abs[3] / len(mu))

	ax1.plot(w_list, scalar_00, '--', color='black',  markersize=1, label='00 loop')
	ax1.plot(w_list, scalar_01, ':',  color='black',  markersize=1, label='01 loop')
	ax1.plot(w_list, scalar_10, '-.', color='black',  markersize=1, label='10 loop')
	ax1.plot(w_list, scalar_11, '-',  color='black',  markersize=1, label='11 loop')

def find_rank(wr, w0):
	"""
	Return the list of ranks for the solution kappa.

	Parameters:
	wr -- list of Decimal
	w0 -- Decimal

	Return:
	rank -- list of integer
	wr -- list of Decimal
	"""
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if wr:
		# Count number of rank increment ('wr2') and rank decrement (wr1')
		counter_1 = sum(1 for w in wr if w > w0)
		counter_2 = sum(-1 for w in wr if w < w0)
		num_wr = counter_1 + counter_2
		
		rank_init = 1 + num_wr
		rank.append(rank_init)
		
		for w in wr:
		
			# Rank increases
			if w > w0:
				rank.append(rank[-1] - 1)
				
			# Rank decreases
			if w < w0:
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, wr

def compute_rank_wr(wr, w0):
	"""
	Compute intervals for each ranks in order to plot them.

	Parameters:
	wr -- list of Decimal
	w0 -- Decimal

	Return:
	wr 	 -- list of Decimal
	rank -- list of integer
	"""
	# Transform Decimal into sorted float
	getcontext().prec = 8
	wr = [float(w) for w in wr]
	wr = sorted(wr)

	rank, wr = find_rank(wr, w0)

	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr   = [i for i in zip(wr, wr)]
		
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr   = list(itertools.chain.from_iterable(wr))
		
	# Insert edges for plotting
	wr.insert(0, 0)
	wr.append(1)

	return wr, rank

def sim_1bit(n, i, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n 		-- integer
	i 		-- integer
	K 		-- string
	kappa 	-- string
	num_sim -- integer

	Return:
	NaN
	"""
	# Message
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	# Noise power consumption
	R = noise(mu)

	# Secret values for i-th bit
	key_i = gen_key_i(i, kappa, K)
	key = gen_key(i, n)

	# Signal reference vectors
	S_ref = signal_2D(n, i, key, mu)

	# Global power consumption P_init = R
	P_init = [Decimal(r) for r in R]

	# Scalar product <S-ref, P>
	scalar_init = np.dot(S_ref, P_init)

	# ------------------------------------------------------------------------------------------- #

	# List of indexes for scalar product
	list_kappa_idx = kappa_idx(n)

	# Retrieve idx of kappa's from solution kappa
	solution_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[2] == key_i[(i+1) % n]) and (sublist[3] == key_i[(i+2) % n])]
	solution_idx = list(itertools.chain.from_iterable(solution_idx))

	nonsol_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[0] != solution_idx[0])]

	# Find initial scalar product values
	solution_init = scalar_init[solution_idx[1]]

	nonsol_init = []
	for j in range(3):
		nonsol_init.append(scalar_init[nonsol_idx[j][1]])
	
	# Determine the sign of initial solution value depending on the activity of the register at bit i
	p_i = xor_secret_i(K, kappa, i)
	solution_init = solution_init * p_i

	# ------------------------------------------------------------------------------------------- #

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": [2,1]}, sharex=True)

	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)

	# List of all intersections with the solution kappa function
	wr = []

	# Interval of w for abscissas
	interval = [0, 1]

	# Find the fiber such as f(w0) = 0 
	w0 = find_abs_w0(solution_init, len(mu))

	for j in range(3):
		nonsol_abs = [fct_abs_nonsol(w, nonsol_init[j], len(mu)) for w in interval]
		ax1.plot(interval, nonsol_abs, '-', color='tab:red', markersize=1, label='%s' % nonsol_idx[j][0])

	if (solution_init > 0):
		solution_abs = [fct_abs_solution(w, solution_init, p_i, len(mu)) for w in interval]
		ax1.plot(interval, solution_abs, '-', color='tab:blue',  markersize=1, label='%s' % solution_idx[0])

	else:
		interval_dec = [0, w0]
		interval_inc = [w0, 1]
		solution_abs_dec = [fct_abs_solution(w, solution_init, p_i, len(mu)) for w in interval_dec]
		solution_abs_inc = [fct_abs_solution(w, solution_init, p_i, len(mu)) for w in interval_inc]
		ax1.plot(interval_dec, solution_abs_dec, '-', color='tab:blue',  markersize=1, label='%s' % solution_idx[0])
		ax1.plot(interval_inc, solution_abs_inc, '-', color='tab:blue',  markersize=1)

		ax1.axvline(x=w0, linestyle=':', color='tab:blue', markersize=1, label='w_0')

	# Intersections between solution function and nonsol functions
	for j in range(3):
		wr1_nonsol, wr2_nonsol = find_wr(solution_init, nonsol_init[j], len(mu))
		append_wr(wr1_nonsol, wr)
		append_wr(wr2_nonsol, wr)

	# ------------------------------------------------------------------------------------------- #

	# Test for 1000 w points
	# test_loop(K, kappa, mu, i, key_i, n, R, S_ref, ax1)

	wr, rank = compute_rank_wr(wr, w0)

	ax2.plot(wr, rank, color='tab:blue')
	
	ax1.legend(loc='upper left', ncol=2, fancybox=True, fontsize = 'small')
	ax1.set_title(r'Single DoM with K_%s=%s & kappa=%s' %(i, K[i], kappa))
	ax1.set_ylabel('Absolute scalar product |<S_ref,P>|')
	ax2.set_ylabel('Rank solution kappa')
	ax2.set_xlabel('Weight w')
	# ax2.invert_yaxis()
	ax2.set_ylim([4.5, 0])

	fig.tight_layout()
	plt.show()
	# plt.savefig('./plot/sim_1bit_%rowabs_K=%s_kappa=%s_i=%s_#%s.pdf' % (n, K, kappa, i, num_sim), dpi=300)
	# plt.savefig('./plot/sim_1bit_%srow_abs_K=%s_kappa=%s_i=%s_#%s.png' % (n, K, kappa, i, num_sim))
	# plt.close(fig)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='n, i, K, kappa, and num_sim')

	parser.add_argument('n', metavar='n', type=int, default=1, help='length of chi row in {3, 5}')
	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, n-1]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='n-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='n-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if (args.n != 3) and (args.n != 5):
		print('\n**ERROR**')
		print('Required length of chi row: 3 or 5 bits')

	else:
		if (args.n == 3) and (len(args.K) != 3 or len(args.kappa) != 3 or (args.i < 0) or (args.i > 2)):
				print('\n**ERROR**')
				print('Required length of K and kappa: 3')
				print('Index studied bit i: [0, 2]\n')
		elif (args.n == 5) and (len(args.K) != 5 or len(args.kappa) != 5 or (args.i < 0) or (args.i > 4)):
				print('\n**ERROR**')
				print('Required length of K and kappa: 5')
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

			# Time option
			start_t = time.perf_counter()
			
			for j in range(num_sim):
				sim_1bit(n, i, K, kappa, j)

			# Time option
			end_t = time.perf_counter()
			print('\ntime', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))