# Investigating Differential Power Analysis on the Chi Function

Written in Python 3.7.2 by Anna Guinet.

## Input parameters

The scripts take in command line the following parameters:
- ```n``` an integer in {3, 5}. This is the length of the studied chi row.
- ```i``` an integer in {0, 1, 2}. This is the i-th studied bit of the first chi row.
- ```K``` a string of length three or five bits (e.g., 00101). This is the initial the secret state.
- ```kappa``` a string of length three or five bits (e.g., 01110). This is an intermediate value. Kappa is the first bits of the output of the linear layer.
- ```num_sim``` an integer (e.g., 1000). This is the number of simulations.
- ```mode``` an integer in {1, 2} or {1, 2, 3} depending on the script. This is an option to compile the code for absolute (1), squared (2) or both (3) options.

## Plot scalar product functions for one simulation

In order to plot scalar product functions for one simulation, compile one of the script ```sim_*``` with Python.

The one-simulation scripts take the following arguments:
- for the DoM strategy
  - ```sim_1bit_to_plot_abs.py [-h] n i K kappa and num_sim```  with absolute scalar products.
  - ```sim_1bit_to_plot_sq.py [-h] n i K kappa and num_sim```  with squared scalar products.

- for the cDoM strategy
  - ```sim_1x5bits_to_plot_abs.py [-h] K kappa num_sim```  to combine absolute scalar products.
  - ```sim_1x5bits_to_plot_sq.py [-h] K kappa num_sim``` to combine squared scalar products.

- for the CPA strategy
  - ```sim_3bits_3row_to_plot.py [-h] K kappa num_sim``` for a three-bit chi row.

## Plot success probabilities per rank for multiple simulations

In order to plot success probabilities for all the rank with a given number of simulations, compile one of the script ```num_sim_*_pr_success_*``` with Python.

First, the following scripts save success probabilities per rank in a CSV file:
- for the DoM strategy
	- ```num_sim_1bit_pr_success_to_csv_abs.py [-h] n i K kappa num_sim```  with absolute scalar products.
	- ```num_sim_1bit_pr_success_to_csv_sq.py [-h] n i K kappa num_sim```  with squared scalar products.

- for the cDoM strategy
	- ```num_sim_1x3bits_pr_success_to_csv_abs.py [-h] K kappa num_sim```  to combine absolute scalar products.
	- ```num_sim_1x3bits_pr_success_to_csv_sq.py [-h] K kappa num_sim```  to combine squared scalar products.

- for the CPA strategy
	- ```num_sim_3bits_pr_success_to_csv.py [-h] K kappa num_sim```  for a three-bit chi row.

Then, run the following scripts to plot the results from the CSV file:
- ```num_sim_1bit_pr_success_csv_to_plot.py [-h] mode n i K kappa num_sim```  for the DoM strategy with absolute (mode: 1) or squared (mode: 2) scalar products, or both (mode: 3).
- ```num_sim_1x3bits_pr_success_csv_to_plot.py [-h] mode K kappa num_sim```  for the cDoM strategy with absolute (mode: 1) or squared (mode: 2) scalar products, or both (mode: 3) options.

- ```num_sim_3bits_pr_success_csv_to_plot.py [-h] K kappa num_sim ``` for a three-bit chi row.


# Licence

The code is under the Creative Commons Zero v1.0 Universal licence.
