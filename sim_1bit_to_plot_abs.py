#!/usr/bin/env python

"""DPA SIMULATION 3-BIT CHI ROW & 1-BIT K"""

__author__      = "Anna Guinet"
__email__       = "a.guinet@cs.ru.nl"
__version__     = "3.0"

import numpy as np
import itertools
from decimal import *
import math
import matplotlib.pylab as plt
import random
import argparse
import time
import sys

def noise(n=3):
	"""
	Compute a noise power consumption R vector for each possible message mu. 
	
	Parameter:
	n -- integer (defaut 3)
	
	Return:
	R -- list of integers, length 2^n
	"""
	R = []

	# For each message mu
	for j in range(2**n):
		
		# Activity of noise part
		d = random.gauss(0, 1)

		# R = SUM_i (-1)^d_i
		R.append(d)

	return R

def gen_key_i(i, kappa='000', K='0'):
	""" 
	Create key value where key = kappa, except for bit i: key_i = kappa_i XOR K
	
	Parameters:
	kappa -- string (default '000')
	K -- string (default '0')
	i -- integer in [0,n-1]
	
	Return:
	key_i -- list of bool
	"""

	# Transform string into list of booleans
	kappa = list(kappa)
	kappa = [bool(int(j)) for j in kappa]
	
	K = bool(int(K))
	
	# Initialize new key value
	key_i = kappa.copy()
	
	# XOR at indice i
	key_i[i] = K ^ kappa[i]
	
	return key_i

def gen_key(i):
	""" 
	Create key value where key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	Eg: for i = 1, key = [[ 0, 0, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ], [ 1, 0, 1 ]]
	
	Note: We do not take into account key_i as it adds only symmetry the signal power consumption.
	
	Parameters:
	i -- integer in [0,n-1]
	
	Return:
	key -- 2D list of bool, size n x 2^(n-1)
	"""
	key_0 = [[bool(0), bool(0), bool(0)], 
			 [bool(0), bool(0), bool(1)], 
			 [bool(0), bool(1), bool(0)], 
			 [bool(0), bool(1), bool(1)]]

	key_1 = [[bool(0), bool(0), bool(0)], 
			 [bool(0), bool(0), bool(1)], 
			 [bool(1), bool(0), bool(0)], 
			 [bool(1), bool(0), bool(1)]]

	key_2 = [[bool(0), bool(0), bool(0)], 
			 [bool(0), bool(1), bool(0)], 
			 [bool(1), bool(0), bool(0)], 
			 [bool(1), bool(1), bool(0)]]

	key = [key_0, key_1, key_2]
	
	return key[i]

def signal_1D(mu, i, key_i, n=3):
	"""
	Generate signal power consumption S values for a secret state key, for all messages mu. 
	
	Parameters:
	mu -- 2D list of bool, size n x 2^n
	i -- integer in [0,n-1]
	key_i -- list of bool
	n -- integer (default 3)
	
	Return:
	S -- list of integers, length 2^n
	"""
	
	S = []

	# For each message mu
	for j in range(2**n):
			
		# Activity of the first storage cell of the register
		d = key_i[i] ^ mu[j][i] ^ ((key_i[(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key_i[(i+2) % n] ^ mu[j][(i+2) % n]))

		# Power consumption model
		S_j = (-1)**(d)
	
		S.append(S_j)

	return S

def signal_2D(mu, i, key, n=3):
	"""
	Generate signal power consumption S values for a 3-bit kappa and a 1-bit K, for all messages mu.
	
	'key' is a value such as key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	
	Parameters:
	mu -- 2D list of bool, size n x 2^n
	i -- integer in [0,n-1]
	key -- list of bool
	n -- integer (default 3)
	
	Return:
	S -- 2D list of integers, size 2^n x 2^(n-1)
	"""	
	S = []
	
	# For each key value
	#for l in range(2**n):
	for l in range(2**(n-1)):
	
		S_l = []

		# For each message mu value
		for j in range(2**n):
			
			# Activity of the first storage cell of the register
			d = key[l][i] ^ mu[j][i] ^ ((key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n]))
				
			S_lj = (-1)**(d)
				
			S_l.append(S_lj)
		
		S += [S_l]
	
	return S

def find_abs_w0(a):
	"""
	Find the point when for the scalar product of the solution guess equals 0.
	
	Parameter:
	a -- Decimal
	
	Return:
	w0 -- Decimal
	"""	
	w0 = 0

	if (a - Decimal(8)) != 0:
		# w0 = a / a - 8
		w0 = a / (a - Decimal(8))

	return w0
		
def fct_abs_solution(w, a, p_i):
	"""
	Function for the scalar product of the solution guess.
	
	Parameter:
	w -- float
	a -- Decimal
	p_i -- integer
	
	Return:
	y -- Decimal
	"""
	w = Decimal(w)
	constant = Decimal(8) * Decimal(p_i)
	a = a * Decimal(p_i)

	# if a > 0, then y = (value - a) * w + a
	# if a < 0, then y = (a - value) * w - a
	y = (constant - a) * w + a
	y = abs(y)
		
	return y / Decimal(8)
		
def fct_abs_nonsol(w, b):
	"""
	Function for the scalar product of an nonsol guess.
	
	Parameter:
	w -- float
	b -- Decimal
	
	Return:
	y -- Decimal
	"""	

	b = abs(b)
		
	# y = -|b| * x + |b|
	y = - b * Decimal(w) + b
		
	return y / Decimal(8)

def find_wr(a, b):
	"""
	Find the point when for the scalar product of the solution guess equals the scalar product of an nonsol guess.
	
	Parameter:
	a -- Decimal
	b -- Decimal
	
	Return:
	y -- Decimal
	"""	
	b = abs(b)

	w1 = 0
	w2 = 0

	if (Decimal(8) + (b - a)) != 0:
		# w1 = (|b| - a)/ (8 - (|b| - a))
		w1 = (b - a) / (Decimal(8) + (b - a))
	
	if (b + a - Decimal(8)) != 0:
		# w2 = (|b| + a)/ (|b| + a - 8)
		w2 = (b + a) / (b + a - Decimal(8))

	return w1, w2

def xor_secret_i(K, kappa, i):
	""" 
	Compute power consumption at bit i = kappa_i XOR K_i
	
	Parameters:
	kappa -- string
	K -- string
	i -- integer
	
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

def find_rank(wr, w0):
	"""
	Return the list of ranks for the solution kappa.
	
	Parameter:
	wr -- list of Decimal
	w0 -- Decimal
	
	Return:
	rank -- list of integer
	wr -- list of Decimal
	"""

	# List of ranks
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

def sim_1bit(n, i, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 

	Parameters:
	n -- integer
	i -- integer
	K -- string
	kappa -- string
	num_sim -- integer

	Return:
	NaN
	"""
	
	# Signal message mu
	mu = list(itertools.product([bool(0), bool(1)], repeat=n))

	getcontext().prec = 10

	# Noise power consumption
	R = noise(n)

	# Secret values for i-th bit
	key_i = gen_key_i(i, kappa, K)
	key = gen_key(i)

	# All possible signal power consumption values
	S_ref = signal_2D(mu, i, key, n)

	""" Initial values of scalar product """

	scalar_init = []

	# Global power consumption P_init
	P_init = [Decimal(r) for r in R]

	# Scalar product <S-ref, P>
	scalar_init = np.dot(S_ref, P_init)

	# Parameters for functions
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

	solution_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[2] == key_i[(i+1) % n]) and (sublist[3] == key_i[(i+2) % n])]
	solution_idx = list(itertools.chain.from_iterable(solution_idx))

	nonsol_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[0] != solution_idx[0])]
	nonsol0_idx = nonsol_idx[0]
	nonsol1_idx = nonsol_idx[1]
	nonsol2_idx = nonsol_idx[2]

	# Find initial scalar product values
	solution_init = scalar_init[solution_idx[1]]

	nonsol0_init = scalar_init[nonsol0_idx[1]]
	nonsol1_init = scalar_init[nonsol1_idx[1]]
	nonsol2_init = scalar_init[nonsol2_idx[1]]

	# Determine the sign of initial solution value depending on the activity of the register at bit i
	p_i = xor_secret_i(K, kappa, i)
	solution_init = solution_init * p_i

	""" Find the functions according to the kappas and the intersections with solution function """

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": [2,1]}, sharex=True)

	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
	
	# List of all intersections with the solution kappa function
	wr = []

	# interval of w for abscissas
	interval = [0, 1]

	# Find the fiber such as f(w0) = 0 
	w0 = find_abs_w0(solution_init)

	nonsol0_abs = [fct_abs_nonsol(w, nonsol0_init) for w in interval]
	nonsol1_abs = [fct_abs_nonsol(w, nonsol1_init) for w in interval]
	nonsol2_abs = [fct_abs_nonsol(w, nonsol2_init) for w in interval]
	ax1.plot(interval, nonsol0_abs, '-', color='tab:red', markersize=1, label='%s' % nonsol0_idx[0])
	ax1.plot(interval, nonsol1_abs, '-', color='tab:orange', markersize=1, label='%s' % nonsol1_idx[0])
	ax1.plot(interval, nonsol2_abs, '-', color='tab:green', markersize=1, label='%s' % nonsol2_idx[0])

	if (solution_init > 0):
		solution_abs = [fct_abs_solution(w, solution_init, p_i) for w in interval]
		ax1.plot(interval, solution_abs, '-', color='tab:blue',  markersize=1, label='%s' % solution_idx[0])
	
	else:
		interval_dec = [0, w0]
		interval_inc = [w0, 1]
		solution_abs_dec = [fct_abs_solution(w, solution_init, p_i) for w in interval_dec]
		solution_abs_inc = [fct_abs_solution(w, solution_init, p_i) for w in interval_inc]
		ax1.plot(interval_dec, solution_abs_dec, '-', color='tab:blue',  markersize=1, label='%s' % solution_idx[0])
		ax1.plot(interval_inc, solution_abs_inc, '-', color='tab:blue',  markersize=1)

		ax1.axvline(x=w0, linestyle=':', color='tab:blue', markersize=1, label='w_0')


	# intersections between solution function and nonsol functions
	wr1_nonsol0, wr2_nonsol0 = find_wr(solution_init, nonsol0_init)
	wr1_nonsol1, wr2_nonsol1 = find_wr(solution_init, nonsol1_init)
	wr1_nonsol2, wr2_nonsol2 = find_wr(solution_init, nonsol2_init)
		
	if (wr1_nonsol0 > 0) and (wr1_nonsol0 < 1):
		wr.append(wr1_nonsol0)
		# ax1.axvline(x=wr1_nonsol0, linestyle='--', color='tab:red', markersize=1, label='wr1')
		
	if (wr2_nonsol0 > 0) and (wr2_nonsol0 < 1):
		wr.append(wr2_nonsol0)
		# ax1.axvline(x=wr2_nonsol0, linestyle=':', color='tab:red', markersize=1, label='wr2')
		
	if (wr1_nonsol1 > 0) and (wr1_nonsol1 < 1):
		wr.append(wr1_nonsol1)
		# ax1.axvline(x=wr1_nonsol1, linestyle='--', color='tab:orange', markersize=1, label='wr1')
			
	if (wr2_nonsol1 > 0) and (wr2_nonsol1 < 1):
		wr.append(wr2_nonsol1)
		# ax1.axvline(x=wr2_nonsol1, linestyle=':', color='tab:orange', markersize=1, label='wr2')
		
	if (wr1_nonsol2 > 0) and (wr1_nonsol2 < 1):
		wr.append(wr1_nonsol2)
		# ax1.axvline(x=wr1_nonsol2, linestyle='--', color='tab:green', markersize=1, label='wr1')
			
	if (wr2_nonsol2 > 0) and (wr2_nonsol2 < 1):
		wr.append(wr2_nonsol2)
		# ax1.axvline(x=wr2_nonsol2, linestyle=':', color='tab:green', markersize=1, label='wr2')

	""" Tests loop """

	# Results of scalar products
	w_list = []

	# Signal power consumption for a key
	S = signal_1D(mu, i, key_i, n)
		
	# Parameters for loop
	scalar_00 = []
	scalar_01 = []
	scalar_10 = []
	scalar_11 = []

	for j in range(101):

		# Weight of signal part
		w = j / 100
		
		scalar = []
			
		# Global power consumption P = (1 - w) * R + w * S
		P = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S)]

		# Scalar product <S-ref, P>
		scalar = np.dot(S_ref, P)
		scalar_abs = abs(scalar)
		
		w_list.append(w)

		scalar_00.append(scalar_abs[0])
		scalar_01.append(scalar_abs[1])
		scalar_10.append(scalar_abs[2])
		scalar_11.append(scalar_abs[3])

		# if (j == 0) or (j == 100):
		# 	print('00 || j = %s  || '% j, scalar[0])
		# 	print('01 || j = %s  || '% j, scalar[1])
		# 	print('10 || j = %s  || '% j, scalar[2])
		# 	print('11 || j = %s  || '% j, scalar[3])
		# 	print()

	# ax1.plot(w_list, scalar_00, '-.', color='black',  markersize=1, label='00 loop')
	# ax1.plot(w_list, scalar_01, ':', color='black',  markersize=1, label='01 loop')
	# ax1.plot(w_list, scalar_10, '--', color='black',  markersize=1, label='10 loop')
	# ax1.plot(w_list, scalar_11, '-.', color='black',  markersize=1, label='11 loop')

	""" Find rank of solution kappa	"""

	# Transform Decimal into float
	getcontext().prec = 5
	wr = [float(w) for w in wr]

	# Sort by w
	wr = sorted(wr)

	rank, wr = find_rank(wr, w0)

	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	wr = [i for i in zip(wr, wr)]
		
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	wr = list(itertools.chain.from_iterable(wr))
		
	# Insert edges for plotting
	wr.insert(0, 0)
	wr.append(1)

	ax2.plot(wr, rank, color='tab:blue')
	
	ax1.legend(loc='upper left')
	ax1.set_title(r'One-bit strategy for K_%s=%s & kappa=%s' %(i, K[i], kappa))
	ax1.set_ylabel('Scalar product |<S_ref,P>|')
	ax2.set_ylabel('Rank solution kappa')
	ax2.set_xlabel('Weight w')
	# ax2.invert_yaxis()
	ax2.set_ylim([4.5, 0])
	fig.tight_layout()
	
	# plt.show()
	plt.savefig('./plot/sim_1bit_abs_K=%s_kappa=%s_i=%s_#%s.png' % (K, kappa, i, num_sim))
	plt.close(fig)

def main(unused_command_line_args):

	parser = argparse.ArgumentParser(description='i, K, kappa and num_sim')

	parser.add_argument('i', metavar='i', type=int, default=1, help='studied bit of first chi row in [0, 2]')
	parser.add_argument('K', metavar='K', type=str, default='000', help='3-length bit value')
	parser.add_argument('kappa', metavar='kappa', type=str, default='000', help='3-length bit value')
	parser.add_argument('num_sim', metavar='num_sim', type=int, default=100, help='number of simulations')

	args = parser.parse_args()

	if len(args.K) != 3 or len(args.kappa) != 3 or (args.i < 0) or (args.i > 2):
		print('\n**ERROR**')
		print('Required length of K and kappa: 3')
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

		# Time option
		start_t = time.perf_counter()
		
		for j in range(num_sim):
			sim_1bit(n, i, K, kappa, j)

		# Time option
		end_t = time.perf_counter()
		print('\ntime', end_t - start_t, '\n')

if __name__ == '__main__':
	sys.exit(main(sys.argv))