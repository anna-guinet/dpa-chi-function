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

def gen_key(i, n=3):
	""" 
	Create key value where key = kappa, except for bit i: key_i = kappa_i XOR K = 0
	Eg: for i = 1, key = [[ 0, 0, 0 ], [ 0, 0, 1 ], [ 1, 0, 0 ], [ 1, 0, 1 ]]
	
	Note: We do not take into account key_i as it adds only symmetry the signal power consumption.
	
	Parameters:
	i -- integer in [0,n-1]
	n -- integer (default 3)
	
	Return:
	key -- 2D list of bool, size n x 2^(n-1)
	"""

	# All possible values for n-1 bits
	key = list(itertools.product([bool(0), bool(1)], repeat=2))

	# Trasnform nested tuples into nested lists
	key = [list(j) for j in key]

	# Insert 'False' as the i-th position
	key_extend = [j.insert(i % n, bool(0)) for j in key]
	
	return key

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
			#d = mu[j][i] ^ ((key[l][(i+1) % n] ^ mu[j][(i+1) % n] ^ bool(1)) & (key[l][(i+2) % n] ^ mu[j][(i+2) % n]))
				
			S_lj = (-1)**(d)
				
			S_l.append(S_lj)
		
		S += [S_l]
	
	return S

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

def fct_sq_correct(w, a):
	"""
	Function for the scalar product of the correct guess.
	
	Parameter:
	w -- float
	a -- Decimal
	p_i -- integer
	
	Return:
	y -- Decimal
	"""
	sq_a = a**2

	w = Decimal(w)
	sq_w = Decimal(w)**2

	alpha = Decimal(8 - a)**2
	beta = Decimal(2) * a * Decimal(8 - a)
	gamma = sq_a
		
	# y = (8 - a)^2 * w^2  + 2 * a * (8 - a) * w + a^2
	y = (alpha * sq_w) + (beta * w) + gamma

	return y / Decimal(8**2)

def find_sq_w0(a):
	"""
	Find the point when for the scalar product of the correct guess equals 0.
	
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
	
def fct_sq_incorrect(w, b):
	"""
	Function for the scalar product of an incorrect guess.
	
	Parameter:
	w -- float
	b -- Decimal
	
	Return:
	y -- Decimal
	"""	
	sq_b = b**2
	
	# y = b^2 * (w - 1)^2
	y = sq_b * (Decimal(1 - w)**2)
		
	return y / Decimal(8**2)
		
def find_inter(a, b):
	"""
	Find the point when for the scalar product of the correct guess equals the scalar product of an incorrect guess.
	
	Parameter:
	a -- Decimal
	b -- Decimal
	
	Return:
	y -- Decimal
	"""	
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

def find_rank(inter, w0):
	"""
	Return the list of ranks for the correct kappa.
	
	Parameter:
	inter -- list of Decimal
	w0 -- Decimal
	
	Return:
	rank -- list of integer
	inter -- list of Decimal
	"""

	# List of ranks
	rank = []

	# If the list is not empty, retrieve the rank in [0,1]
	if inter:

		# Count number of rank increment ('inter2') and rank decrement (inter1')
		counter_1 = sum(1 for w in inter if w > w0)
		counter_2 = sum(-1 for w in inter if w < w0)
		num_inter = counter_1 + counter_2
		
		rank_init = 1 + num_inter
		
		rank.append(rank_init)
		
		for w in inter:
		
			# Rank increases
			if w > w0:
				rank.append(rank[-1] - 1)
				
			# Rank decreases
			if w < w0:
				rank.append(rank[-1] + 1)
	else:
		rank = [1]
		
	return rank, inter

def sim_1bit(n, i, K, kappa, num_sim):
	"""
	Compute simulation scalar products and plot outcomes. 
	
	Parameter:
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
	key_i = gen_key_i(i, K, kappa)
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

	correct_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[2] == key_i[(i+1) % n]) and (sublist[3] == key_i[(i+2) % n])]
	correct_idx = list(itertools.chain.from_iterable(correct_idx))

	incorrect_idx = [sublist for sublist in list_kappa_idx[i] if (sublist[0] != correct_idx[0])]
	incorrect0_idx = incorrect_idx[0]
	incorrect1_idx = incorrect_idx[1]
	incorrect2_idx = incorrect_idx[2]

	# Find initial scalar product values
	correct_init = scalar_init[correct_idx[1]]

	incorrect0_init = scalar_init[incorrect0_idx[1]]
	incorrect1_init = scalar_init[incorrect1_idx[1]]
	incorrect2_init = scalar_init[incorrect2_idx[1]]

	# Determine the sign of initial correct value depending on the activity of the register at bit i
	p_i = xor_secret_i(K, kappa, i)
	correct_init = correct_init * p_i

	""" Find the functions according to the kappas and the intersections with correct function """

	# Display a figure with two subplots
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5,5), gridspec_kw={"height_ratios": [2,1]}, sharex=True)

	# Make subplots close to each other and hide x ticks for all but bottom plot.
	fig.subplots_adjust(hspace=0.2)
	plt.setp([ax.get_xticklabels() for ax in fig.axes[:-1]], visible=False)
	
	# List of all intersections with the correct kappa function
	inter = []

	# Find the fiber such as f(w0) = 0 
	w0 = find_sq_w0(correct_init)

	# Abscissas
	precision = 1000
	interval = [(w / precision) for w in range(precision)]

	incorrect0_sq = [fct_sq_incorrect(w, incorrect0_init) for w in interval]
	incorrect1_sq = [fct_sq_incorrect(w, incorrect1_init) for w in interval]
	incorrect2_sq = [fct_sq_incorrect(w, incorrect2_init) for w in interval]
	ax1.plot(interval, incorrect0_sq, '-', color='tab:red', markersize=1, label='%s' % incorrect0_idx[0])
	ax1.plot(interval, incorrect1_sq, '-', color='tab:orange', markersize=1, label='%s' % incorrect1_idx[0])
	ax1.plot(interval, incorrect2_sq, '-', color='tab:green', markersize=1, label='%s' % incorrect2_idx[0])

	if (correct_init > 0):
		correct_sq = [fct_sq_correct(w, correct_init) for w in interval]
		ax1.plot(interval, correct_sq, '-', color='tab:blue',  markersize=1, label='%s' % correct_idx[0])
	
	else:
		interval_dec = [w for w in interval if w <= w0]
		interval_inc = [w for w in interval if w >= w0]
		correct_sq_dec = [fct_sq_correct(w, correct_init) for w in interval_dec]
		correct_sq_inc = [fct_sq_correct(w, correct_init) for w in interval_inc]
		ax1.plot(interval_dec, correct_sq_dec, '-', color='tab:blue',  markersize=1, label='%s' % correct_idx[0])
		ax1.plot(interval_inc, correct_sq_inc, '-', color='tab:blue',  markersize=1)

		ax1.axvline(x=w0, linestyle=':', color='tab:blue', label='w_0', markersize=1)

	# Intersections between correct function and incorrect functions
	inter1_incorrect0, inter2_incorrect0 = find_inter(correct_init, incorrect0_init)
	inter1_incorrect1, inter2_incorrect1 = find_inter(correct_init, incorrect1_init)
	inter1_incorrect2, inter2_incorrect2 = find_inter(correct_init, incorrect2_init)
		
	if (inter1_incorrect0 > 0) and (inter1_incorrect0 < 1):
		inter.append(inter1_incorrect0)
		# ax1.axvline(x=inter1_incorrect0, linestyle='--', color='tab:red', markersize=1, label='inter1')
		
	if (inter2_incorrect0 > 0) and (inter2_incorrect0 < 1):
		inter.append(inter2_incorrect0)
		# ax1.axvline(x=inter2_incorrect0, linestyle=':', color='tab:red', markersize=1, label='inter2')
		
	if (inter1_incorrect1 > 0) and (inter1_incorrect1 < 1):
		inter.append(inter1_incorrect1)
		# ax1.axvline(x=inter1_incorrect1, linestyle='--', color='tab:orange', markersize=1, label='inter1')
			
	if (inter2_incorrect1 > 0) and (inter2_incorrect1 < 1):
		inter.append(inter2_incorrect1)
		# ax1.axvline(x=inter2_incorrect1, linestyle=':', color='tab:orange', markersize=1, label='inter2')
		
	if (inter1_incorrect2 > 0) and (inter1_incorrect2 < 1):
		inter.append(inter1_incorrect2)
		# ax1.axvline(x=inter1_incorrect2, linestyle='--', color='tab:green', markersize=1, label='inter1')
			
	if (inter2_incorrect2 > 0) and (inter2_incorrect2 < 1):
		inter.append(inter2_incorrect2)
		# ax1.axvline(x=inter2_incorrect2, linestyle=':', color='tab:green', markersize=1, label='inter2')


	""" Tests loop

	# Results of scalar products
	w_list = []

	# Signal power consumption for a key
	S = signal_1D(mu, i, key_i, n)
		
	# Parameters for loop
	scalar_00 = []
	scalar_01 = []
	scalar_10 = []
	scalar_11 = []

	test = []

	for j in range(101):

		# Weight of signal part
		w = j / 100
		
		scalar = []
			
		# Global power consumption P = (1 - w) * R + w * S
		P = [((Decimal(1 - w) * Decimal(r)) + (Decimal(w) * Decimal(s))) for r, s in zip(R, S)]

		# Scalar product <S-ref, P>^2
		scalar = np.dot(S_ref, P)
		scalar_sq = scalar**2
		
		w_list.append(w)

		scalar_00.append(Decimal(scalar_sq[0]) / Decimal(8**2))
		scalar_01.append(Decimal(scalar_sq[1]) / Decimal(8**2))
		scalar_10.append(Decimal(scalar_sq[2]) / Decimal(8**2))
		scalar_11.append(Decimal(scalar_sq[3]) / Decimal(8**2))

	ax1.plot(w_list, scalar_00, ':', color='black',  markersize=1, label='00 loop')
	ax1.plot(w_list, scalar_01, '--', color='black',  markersize=1, label='01 loop')
	ax1.plot(w_list, scalar_10, '--', color='black',  markersize=1, label='10 loop')
	ax1.plot(w_list, scalar_11, '-.', color='black',  markersize=1, label='11 loop')
	"""

	""" Find rank of correct kappa	"""

	# Transform Decimal into float
	getcontext().prec = 5
	inter = [float(w) for w in inter]

	# Sort by w
	inter = sorted(inter)

	rank, inter = find_rank(inter, w0)

	# Expand the lists for plotting
	rank = [r for r in zip(rank, rank)]
	inter = [i for i in zip(inter, inter)]
		
	# Un-nest the previous lists
	rank = list(itertools.chain.from_iterable(rank))
	inter = list(itertools.chain.from_iterable(inter))
		
	# Insert edges for plotting
	inter.insert(0, 0)
	inter.append(1)

	ax2.plot(inter, rank, color='tab:blue')

	ax1.legend(loc='upper left')
	ax1.set_title(r'One-bit strategy for K_%s=%s & kappa=%s' %(i, K[i], kappa))
	ax1.set_ylabel('Scalar product <S_ref,P>^2')
	ax2.set_ylabel('Rank kappa solution')
	ax2.set_xlabel('Weight w')
	# ax2.invert_yaxis()
	ax2.set_ylim([5, 0])
	
	fig.tight_layout()
	# plt.show()
	plt.savefig('./plot/sim_1bit_sq_K=%s_kappa=%s_i=%s_#%s.png' % (K, kappa, i, num_sim))
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