import pandas as pd
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
from pylab import cm
import numpy as np

import input_parsing as parse
import plot_stuff as mplt  

def GigaVal_per_s(val, time):
	return val * 1e-9 / time

def dgemm_flops(M,N,K):
	return M*N*(2*K+1)

font=8
plt.rc('font', family='serif', serif='Times')
plt.rc('text', usetex=True)
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
plt.rc('axes', labelsize=font)

# width as measured in inkscape
width = 3.487
height = width / 1.618


machine_names= ['testbed-I_Tesla-K40','testbed-II_Tesla-V100']
funcs=['Dgemm', 'Sgemm']
version='final'
exchept_T_list = [512,768,666]
for func in funcs:
	for machine in machine_names:

		validation_data = parse.read_validation_values("..",machine,func,version)	
		validation_pred = parse.read_validation_predictions("..",machine,func,version)	
		validation_mixed = parse.create_validation_set(validation_data, validation_pred, func)
		
		#Square set
		locs= [[1,1,1]]
		mid_sizes= [16384]
		ctrs_fat = [2]
		ctrs_thin = []
		square_validation_set = parse.validation_set_split_BLAS3("Square",locs,mid_sizes,ctrs_fat,ctrs_thin,validation_mixed)
		square_validation_set = parse.cleanT(square_validation_set,exchept_T_list)
		print(square_validation_set)

		fig, ax = plt.subplots()
		fig.subplots_adjust(left=.15, bottom=.17, right=.99, top=.97)

		plt.plot(square_validation_set['T'][square_validation_set['Noreuse_t'] < 3*square_validation_set['CoCopeLia_t']], square_validation_set['Noreuse_t'][square_validation_set['Noreuse_t'] < 3*square_validation_set['CoCopeLia_t']],
				 '^', markersize=4, color=mplt.cp3[0], label='cuBLASXt')
		plt.plot(square_validation_set['T'], square_validation_set['CoCopeLia_t'],
				 'o', markersize=4, color=mplt.cp3[1], label='CoCopeLia wrapper')
		plt.plot(square_validation_set['T'][square_validation_set['Noreuse_t'] < 3*square_validation_set['CoCopeLia_t']], square_validation_set['CoCopelia_nr'][square_validation_set['Noreuse_t'] < 3*square_validation_set['CoCopeLia_t']],
				 linewidth=1, color=mplt.cp3[0], label='BTS-Model')
		plt.plot(square_validation_set['T'], square_validation_set['CoCopelia'],
				 linewidth=1, color=mplt.cp3[1], label='DR-Model')
		plt.plot(square_validation_set['T'], square_validation_set['werkhoven'],
				 linewidth=1, color=mplt.cp3[2], label='CSO-Model [10]')

		plt.legend(fontsize=font, loc='upper right', fancybox = False, ncol=1)

		ax.set_ylabel('Time (s)')
		#plt.yscale('log')
		ax.set_xlabel('Tile Dim (T)')

		#ax.set_ylim(1,10)

		fig.set_size_inches(width, height)
		fig.savefig('Plots/validation_plot_Model_time_sys-%s_func-%s_dim_%d-%d-%d_loc-%d-%d-%d.pdf' % (machine, func, mid_sizes[0], mid_sizes[0], mid_sizes[0], locs[0][0], locs[0][1], locs[0][2]))
		plt.close()

